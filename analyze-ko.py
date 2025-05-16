#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_router_stats_ko.py

vLLM 기반 Qwen MoE 모델(FusedMoE)에 훅을 걸어
1) 전체 프롬프트에 대한 전문가 활성화 통계(EMA, 호출 횟수)
2) 한글 프롬프트에 한정한 전문가 활성화 통계(EMA, 호출 횟수)

를 따로 수집합니다.

사용법:
  # 한글 프롬프트를 파일(ko_prompts.txt)에 한 줄씩 저장한 뒤:
  python collect_router_stats_ko.py \
    --model kalomaze/Qwen3-16B-A3B \
    --tokens 128 \
    --decay 0.99999 \
    --output stats \
    --prompts_file ko_prompts.txt

  # 혹은 명령줄 인자로 직접 프롬프트를 나열할 수도 있습니다:
  python collect_router_stats_ko.py \
    --model kalomaze/Qwen3-16B-A3B \
    --tokens 128 \
    --decay 0.99999 \
    --output stats \
    "안녕하세요" "오늘 날씨 어때?" "한국 역사 알려줘"
"""
import argparse
import os
import re
from collections import Counter, defaultdict

from vllm import LLM, SamplingParams
from vllm.model_executor.layers.fused_moe.layer import FusedMoE

def collect_stats(model_name, prompts, max_tokens, decay, output_dir):
    # ─── 통계 저장구조 ────────────────────────────────────────────────────────
    router_hits_all = defaultdict(Counter)
    ema_all         = defaultdict(lambda: defaultdict(float))
    router_hits_ko  = defaultdict(Counter)
    ema_ko          = defaultdict(lambda: defaultdict(float))

    # 현재 처리 중인 프롬프트를 hook에서 참조하기 위한 공유 dict
    current_prompt = {"text": ""}

    # ─── MoE forward 훅 정의 ──────────────────────────────────────────────────
    def moe_hook(module, inputs, outputs):
        # outputs[2] 에 선택된 expert 인덱스가 담겨 있음
        indices = outputs[2].detach().cpu().view(-1)
        total   = len(indices)
        ctr     = Counter(int(i) for i in indices)
        layer_id = module.layer_ids[0] if hasattr(module, "layer_ids") else None

        prompt_text = current_prompt["text"]
        is_ko = bool(re.search(r"[가-힣]", prompt_text))

        for exp, cnt in ctr.items():
            pct = cnt / total
            # 전체 통계
            ema_all[layer_id][exp]    = ema_all[layer_id][exp]    * decay + pct * (1 - decay)
            router_hits_all[layer_id][exp] += cnt
            # 한글 전용 통계
            if is_ko:
                ema_ko[layer_id][exp]    = ema_ko[layer_id][exp]    * decay + pct * (1 - decay)
                router_hits_ko[layer_id][exp] += cnt

    # ─── 모델 로드 및 훅 등록 ──────────────────────────────────────────────────
    model = LLM(
        model_name,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        tensor_parallel_size=1
    )
    for m in model.engine.model.modules():
        if isinstance(m, FusedMoE):
            m.register_forward_hook(moe_hook)

    # ─── 출력 디렉토리 준비 ───────────────────────────────────────────────────
    os.makedirs(output_dir,            exist_ok=True)
    os.makedirs(os.path.join(output_dir, "all"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ko"),  exist_ok=True)

    # ─── 프롬프트별 generate 호출 ─────────────────────────────────────────────
    for prompt in prompts:
        print(f"[INFO] Prompt: {prompt}")
        current_prompt["text"] = prompt
        _ = model.generate(prompt, SamplingParams(max_tokens=max_tokens))

    # ─── 전체 통계 파일 쓰기 ─────────────────────────────────────────────────
    for layer, ctr in router_hits_all.items():
        fname = os.path.join(output_dir, "all", f"routing_stats_layer_{layer}_all.txt")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(f"# [전체 프롬프트]\n")
            f.write(f"Layer {layer}, EMA(decay={decay}):\n")
            for exp, ema_val in sorted(ema_all[layer].items(), key=lambda x: -x[1]):
                f.write(f"Expert {exp}: {ema_val*100:.2f}%\n")
            f.write(f"Lifetime Tokens: {sum(ctr.values())}\n")
        print(f"[INFO] Saved all stats -> {fname}")

    # ─── 한글 전용 통계 파일 쓰기 ────────────────────────────────────────────
    for layer, ctr in router_hits_ko.items():
        fname = os.path.join(output_dir, "ko", f"routing_stats_layer_{layer}_ko.txt")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(f"# [한글 프롬프트만]\n")
            f.write(f"Layer {layer}, EMA(decay={decay}):\n")
            for exp, ema_val in sorted(ema_ko[layer].items(), key=lambda x: -x[1]):
                f.write(f"Expert {exp}: {ema_val*100:.2f}%\n")
            f.write(f"Lifetime Tokens: {sum(ctr.values())}\n")
        print(f"[INFO] Saved ko stats  -> {fname}")

def main():
    parser = argparse.ArgumentParser(
        description="Collect MoE router stats for all vs. Korean prompts"
    )
    parser.add_argument(
        "--model", default="kalomaze/Qwen3-16B-A3B",
        help="Hugging Face model name or path"
    )
    parser.add_argument(
        "--tokens", type=int, default=128,
        help="Max new tokens per prompt"
    )
    parser.add_argument(
        "--decay", type=float, default=0.99999,
        help="EMA decay factor"
    )
    parser.add_argument(
        "--output", default="stats",
        help="Output directory"
    )
    parser.add_argument(
        "--prompts_file", type=str,
        help="한국어 프롬프트를 줄단위로 저장한 텍스트 파일 경로"
    )
    parser.add_argument(
        "prompts", nargs="*",
        help="직접 입력할 프롬프트들 (–-prompts_file 우선)"
    )
    args = parser.parse_args()

    # prompts_file 우선, 아니면 positional prompts
    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as pf:
            prompts = [line.strip() for line in pf if line.strip()]
    else:
        prompts = args.prompts

    if not prompts:
        parser.error("프롬프트가 제공되지 않았습니다. --prompts_file 또는 positional prompts 사용하세요.")

    collect_stats(
        args.model,
        prompts,
        args.tokens,
        args.decay,
        args.output,
    )

if __name__ == "__main__":
    main()
    
    
    
안녕하세요
오늘 날씨 어때?
한국 역사에 대해 설명해줘.
