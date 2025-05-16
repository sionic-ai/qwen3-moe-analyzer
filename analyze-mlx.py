#!/usr/bin/env python3
"""
collect_router_stats_mlx.py

스크립트 설명:

MLX 기반 Qwen3 MoE 모델의 MoE 레이어 (Qwen3MoeSparseMoeBlock) 동작을 수정하여
각 레이어별 전문가(expert) 활성 빈도를 수집하고,
지수이동평균(EMA)을 적용하여 통계 파일을 생성합니다.
또한, 전체 모델에 대한 집계 통계 및 표 형식 출력을 제공합니다.

사용법:
python collect_router_stats_mlx.py [--model MODEL] [--tokens N] [--decay D]
[--output DIR] "prompt1" "prompt2" ...

예시:
python collect_router_stats_mlx.py \
    --model mlx-community/Qwen3-235B-A22B-8bit \
    --tokens 128 \
    --decay 0.99999 \
    --output stats_mlx/ \
    "Translate to French: Hello" "Explain recursion in simple terms."

필수 설치: pip install tabulate
"""
import argparse
import os
from collections import Counter, defaultdict
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock # Assuming this is the correct import path
try:
    from tabulate import tabulate
except ImportError:
    print("Please install tabulate: pip install tabulate")
    exit(1)

# Global statistics collectors
router_hits_global = defaultdict(Counter)
ema_stats_global = defaultdict(lambda: defaultdict(float))
current_decay_global = 0.99999  # Default, will be updated by args

# Store the original __call__ method
_original_qwen3moesparsempeblock_call = None

def patched_qwen3moesparsempeblock_call(self, x: mx.array):
    """
    Patched __call__ method for Qwen3MoeSparseMoeBlock to collect stats.
    """
    gates = self.gate(x)
    gates = mx.softmax(gates, axis=-1, precise=True)

    k = self.top_k
    inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
    scores = mx.take_along_axis(gates, inds, axis=-1)

    if self.norm_topk_prob:
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)

    if hasattr(self, 'layer_idx_for_stats'):
        layer_id = self.layer_idx_for_stats
        expert_indices_flat = inds.reshape(-1).tolist()
        total_routing_events = len(expert_indices_flat)

        if total_routing_events > 0:
            expert_counts_this_call = Counter(int(i) for i in expert_indices_flat)
            for exp, cnt in expert_counts_this_call.items():
                pct = cnt / total_routing_events
                ema_stats_global[layer_id][exp] = ema_stats_global[layer_id][exp] * current_decay_global + pct * (1 - current_decay_global)
                router_hits_global[layer_id][exp] += cnt
    
    y = self.switch_mlp(x, inds)
    y = (y * scores[..., None]).sum(axis=-2)
    return y

def save_and_print_stats(output_dir, decay_val):
    os.makedirs(output_dir, exist_ok=True)

    if not router_hits_global:
        print("[INFO] No router hits were recorded. No stats files or tables will be generated.")
        return

    total_model_lifetime_token_selections = 0
    model_wide_expert_hits = Counter()
    model_wide_expert_active_layers_count = defaultdict(int)
    model_wide_expert_ema_sum = defaultdict(float)

    print("\n--- Per-Layer Expert Activation Statistics ---")
    # Sort layers by index for consistent output
    sorted_layer_indices = sorted(router_hits_global.keys())

    for layer_idx in sorted_layer_indices:
        expert_counts_in_layer = router_hits_global[layer_idx]
        fname = os.path.join(output_dir, f"routing_stats_layer_{layer_idx}.txt")
        
        table_data_layer = []
        layer_ema_stats = ema_stats_global.get(layer_idx, {})
        
        # Get all expert IDs that have either a hit or an EMA value for this layer
        all_expert_ids_for_layer = set(expert_counts_in_layer.keys()) | set(layer_ema_stats.keys())
        
        current_layer_token_selections = sum(expert_counts_in_layer.values())
        total_model_lifetime_token_selections += current_layer_token_selections

        # Sort experts by EMA descending for this layer's table and file
        sorted_experts_for_layer_display = sorted(
            all_expert_ids_for_layer,
            key=lambda exp_id: -layer_ema_stats.get(exp_id, 0.0)
        )

        with open(fname, "w") as f:
            f.write(f"Layer {layer_idx}, EMA (decay={decay_val}):\n")
            for exp_id in sorted_experts_for_layer_display:
                ema_val = layer_ema_stats.get(exp_id, 0.0)
                hits = expert_counts_in_layer.get(exp_id, 0)
                table_data_layer.append([exp_id, f"{ema_val*100:.2f}%", hits])
                f.write(f"Expert {exp_id}: {ema_val*100:.2f}%\n")

                # Aggregate for model-wide stats
                model_wide_expert_hits[exp_id] += hits
                # An expert is active in a layer if it has recorded hits or a non-zero EMA in that layer
                if hits > 0 or ema_val > 0: # Check ema_val directly
                    model_wide_expert_active_layers_count[exp_id] += 1
                    model_wide_expert_ema_sum[exp_id] += ema_val # Summing EMA for averaging later

            f.write(f"Lifetime Token Selections: {current_layer_token_selections}\n")
        
        print(f"\nLayer {layer_idx} (EMA decay={decay_val}, Lifetime Token Selections: {current_layer_token_selections}):")
        if table_data_layer:
            print(tabulate(table_data_layer, headers=["Expert ID", "EMA (%)", "Hits"], tablefmt="grid"))
        else:
            print("No expert activity recorded for this layer.")
        print(f"[INFO] Saved stats for Layer {layer_idx} -> {fname}")

    # --- Aggregated Model-Wide Statistics ---
    print("\n\n--- Aggregated Model-Wide Expert Activation Statistics ---")
    
    aggregated_table_data = []
    all_model_expert_ids = set(model_wide_expert_hits.keys()) | set(model_wide_expert_active_layers_count.keys())

    # Sort experts by total model-wide hits (desc), then by avg EMA (desc)
    sorted_model_experts = sorted(
        all_model_expert_ids,
        key=lambda eid: (
            -model_wide_expert_hits.get(eid, 0), 
            -(model_wide_expert_ema_sum.get(eid, 0.0) / model_wide_expert_active_layers_count.get(eid, 1))
            if model_wide_expert_active_layers_count.get(eid, 0) > 0 else 0
        )
    )

    for exp_id in sorted_model_experts:
        total_hits = model_wide_expert_hits.get(exp_id, 0)
        num_active_layers = model_wide_expert_active_layers_count.get(exp_id, 0)
        
        avg_ema_val_pct = 0.0
        if num_active_layers > 0:
            avg_ema_val_pct = (model_wide_expert_ema_sum.get(exp_id, 0.0) / num_active_layers) * 100
        
        overall_selection_percentage = 0.0
        if total_model_lifetime_token_selections > 0:
             overall_selection_percentage = (total_hits / total_model_lifetime_token_selections) * 100

        aggregated_table_data.append([
            exp_id, 
            f"{avg_ema_val_pct:.2f}%", 
            num_active_layers,
            total_hits,
            f"{overall_selection_percentage:.2f}%"
        ])

    headers_aggregated = ["Expert ID", "Avg. EMA (%)", "Active Layers", "Total Hits", "Overall Sel. (%)"]
    if aggregated_table_data:
        print(f"\nTotal Lifetime Token Selections (all MoE layers): {total_model_lifetime_token_selections}")
        print(tabulate(aggregated_table_data, headers=headers_aggregated, tablefmt="grid"))
    else:
        print("No aggregated expert activity to display.")

    # Save aggregated stats to a file
    agg_fname = os.path.join(output_dir, "routing_stats_aggregated.txt")
    with open(agg_fname, "w") as f:
        f.write(f"Aggregated Model-Wide Expert Activation Statistics (EMA decay={decay_val})\n")
        f.write(f"Total Lifetime Token Selections (all MoE layers): {total_model_lifetime_token_selections}\n\n")
        f.write(tabulate(aggregated_table_data, headers=headers_aggregated, tablefmt="pipe")) # Use 'pipe' for text file
        f.write("\n")
    print(f"[INFO] Saved aggregated stats -> {agg_fname}")


def collect_stats_mlx(model_name, prompts, max_tokens, decay, output_dir):
    global current_decay_global, _original_qwen3moesparsempeblock_call
    current_decay_global = decay

    if _original_qwen3moesparsempeblock_call is None:
        _original_qwen3moesparsempeblock_call = Qwen3MoeSparseMoeBlock.__call__
        Qwen3MoeSparseMoeBlock.__call__ = patched_qwen3moesparsempeblock_call
        print("[INFO] Patched Qwen3MoeSparseMoeBlock.__call__ method.")

    print(f"[INFO] Loading model: {model_name}")
    model, tokenizer = load(model_name)
    print(f"[INFO] Model {model_name} loaded.")

    moe_layer_count = 0
    # Assuming model structure is model.model.layers for MLX HF models
    # Or model.layers if model directly holds the list of Qwen3MoeDecoderLayer
    target_layers_list = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'): # Common for AutoModel type classes
        target_layers_list = model.model.layers
    elif hasattr(model, 'layers'): # If model class itself has .layers (like raw Qwen3MoeModel)
         target_layers_list = model.layers
    else:
        print("[WARNING] Could not find model.model.layers or model.layers structure to inject layer_idx.")

    if target_layers_list:
        for i, layer_module in enumerate(target_layers_list):
            if hasattr(layer_module, 'mlp') and isinstance(layer_module.mlp, Qwen3MoeSparseMoeBlock):
                layer_module.mlp.layer_idx_for_stats = i
                # print(f"[DEBUG] Tagged MoE block in layer {i} (via mlp attribute)")
                moe_layer_count +=1
            elif isinstance(layer_module, Qwen3MoeSparseMoeBlock): # Direct MoE layer
                 layer_module.layer_idx_for_stats = i
                #  print(f"[DEBUG] Tagged MoE block (direct layer) {i}")
                 moe_layer_count +=1
    
    if moe_layer_count > 0:
        print(f"[INFO] Found and tagged {moe_layer_count} MoE blocks for stats collection.")
    else:
        print("[ERROR] No Qwen3MoeSparseMoeBlock instances found in the model. Cannot collect stats.")
        print("Please ensure the model is a Qwen3 MoE model and the path to Qwen3MoeSparseMoeBlock is correct.")
        # Optionally, restore here if no MoE blocks are found and exit
        # if _original_qwen3moesparsempeblock_call is not None:
        #     Qwen3MoeSparseMoeBlock.__call__ = _original_qwen3moesparsempeblock_call
        return

    for prompt_text in prompts:
        print(f"\n[INFO] Generating for prompt: \"{prompt_text[:50]}...\"")
        formatted_prompt = prompt_text
        if tokenizer.chat_template:
            try:
                messages = [{"role": "user", "content": prompt_text}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                print(f"[WARNING] Failed to apply chat template: {e}. Using raw prompt.")
        
        _ = generate(model, tokenizer, prompt=formatted_prompt, max_tokens=max_tokens, verbose=False)
        mx.eval() 

    # Call the new function to save and print all stats
    save_and_print_stats(output_dir, decay)

    if _original_qwen3moesparsempeblock_call is not None:
        Qwen3MoeSparseMoeBlock.__call__ = _original_qwen3moesparsempeblock_call
        _original_qwen3moesparsempeblock_call = None # Avoid re-patching if called again in same session
        print("[INFO] Restored original Qwen3MoeSparseMoeBlock.__call__ method.")

def main():
    parser = argparse.ArgumentParser(
        description="Collect MoE router activation stats for MLX Qwen3 MoE models with tabular output."
    )
    parser.add_argument(
        "--model", default="mlx-community/Qwen3-235B-A22B-8bit",
        help="Hugging Face model name or path for MLX (e.g., mlx-community/Qwen3-7B-Chat-MoE-A2.7T-Int4)"
    )
    parser.add_argument(
        "--tokens", type=int, default=128,
        help="Max new tokens per prompt for generation"
    )
    parser.add_argument(
        "--decay", type=float, default=0.99999,
        help="EMA decay factor (e.g., 0.99999)"
    )
    parser.add_argument(
        "--output", default="stats_mlx",
        help="Output directory for routing stats files"
    )
    parser.add_argument(
        "prompts", nargs='+',
        help="List of prompts to generate and collect stats on"
    )
    args = parser.parse_args()
    
    if not args.prompts:
        parser.error("Please provide at least one prompt.")

    # Example with a smaller, more manageable MoE model for testing if Qwen3-235B is too large
    # If you want to test quickly, you might temporarily change the default model in parser
    # to something like "mlx-community/Mistral-7B-Instruct-v0.2-moe-8bit" if it exists and is smaller,
    # or any other MoE model available in MLX community.
    # For the specific "mlx-community/Qwen3-235B-A22B-8bit", ensure your system can handle it.
    
    collect_stats_mlx(
        args.model, args.prompts, args.tokens, args.decay, args.output
    )

if __name__ == "__main__":
    main()