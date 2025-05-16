# Qwen3 MoE Router Statistics & Forced Expert Generation Tools

This repository provides scripts for:
- Collecting expert router statistics (Exponential Moving Average - EMA & hit counts) from Qwen Mixture-of-Experts (MoE) models.
- Generating text with forced expert pools or analyzing expert subsets in MLX-based Qwen3 MoE models.

Blog post: https://blog.sionic.ai/qwen3-moe-strategy

## 1. Setup & Installation (Python 3.10)

Prepare a Python 3.10 environment, create a virtual environment, and install dependencies.

```bash
# Example for Ubuntu: install Python 3.10
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

# Create and activate a virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Script Usage Examples

Below are examples of how to run each script:

| Script File                   | Description                                                                                                                      | Example                                                                                                                                       |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| **`analyze-ko.py`**           | Collects layer-wise expert activation statistics (EMA & hit counts) for all prompts vs. Korean-only prompts using a vLLM-based Qwen MoE model. | ```bash
python analyze-ko.py \
  --model kalomaze/Qwen3-16B-A3B \
  --tokens 128 --decay 0.99999 \
  --output stats --prompts_file ko_prompts.txt
```                                                                                                                                         |
| **`analyze-mlx.py`**          | Patches an MLX Qwen3 MoE model to gather per-layer expert statistics (EMA & hit counts) and prints tables.                         | ```bash
python analyze-mlx.py \
  --model mlx-community/Qwen3-235B-A22B-8bit \
  --tokens 128 --decay 0.99999 --output stats_mlx \
  "Hello" "How's the weather?" "Tell me about Korean history"
```                                                                                                                                         |

## 3. Summary of Analysis Scripts

The table below summarizes key functions and output files of the analysis scripts:

| Script            | Key Features                                                                                                                               | Output Files                                                                      |
|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| **`analyze-ko.py`** | - Hooks into a vLLM-based Qwen MoE model to collect expert activation counts and EMA for all vs. Korean-only prompts.                       | - `stats/all/routing_stats_layer_{layer}_all.txt`<br>- `stats/ko/routing_stats_layer_{layer}_ko.txt` |
| **`analyze-mlx.py`**| - Patches MLX Qwen3 MoE layers to collect per-layer expert activation counts and EMA.<br>- Generates console tables and writes layer-wise and aggregated stats files. | - `stats_mlx/routing_stats_layer_{layer}.txt`<br>- `stats_mlx/routing_stats_aggregated.txt`     |

## 4. Example Analysis Outputs

### 4.1. Per-Layer Expert Statistics (Example from `stats/stats_Qwen3MoeSparseMoeBlock.txt`)

This section shows an example output from one of the analysis scripts, displaying statistics for a single MoE layer.

```shell
(qwen-moe) sionic@vessl-a100-sxm-02:~/sigrid/qwen-moe$ cat stats/stats_Qwen3MoeSparseMoeBlock.txt
Layer: Qwen3MoeSparseMoeBlock
  Avg forward time: 45.36 ms
  Avg gating+stats time: 0.45 ms
EMA(decay=0.99999):
  Expert 22: 0.14%
  Expert 104: 0.13%
  Expert 75: 0.12%
  Expert 33: 0.12%
  Expert 101: 0.11%
  Expert 1: 0.10%
  Expert 7: 0.10%
  Expert 9: 0.10%
  Expert 25: 0.09%
  Expert 20: 0.09%
  Expert 18: 0.09%
  Expert 42: 0.09%
  Expert 97: 0.09%
  Expert 86: 0.09%
  Expert 125: 0.09%
  Expert 56: 0.08%
  Expert 38: 0.08%
  Expert 24: 0.08%
  Expert 120: 0.08%
  Expert 44: 0.08%
  Expert 102: 0.07%
  Expert 0: 0.07%
  Expert 31: 0.07%
  Expert 23: 0.07%
  Expert 64: 0.07%
  Expert 3: 0.07%
  Expert 82: 0.07%
  Expert 13: 0.06%
  Expert 35: 0.06%
  Expert 4: 0.06%
  Expert 96: 0.06%
  Expert 83: 0.06%
  Expert 17: 0.06%
  Expert 93: 0.06%
  Expert 71: 0.06%
  Expert 29: 0.06%
  Expert 45: 0.06%
  Expert 116: 0.06%
  Expert 50: 0.06%
  Expert 69: 0.05%
  Expert 48: 0.05%
  Expert 55: 0.05%
  Expert 10: 0.05%
  Expert 91: 0.05%
  Expert 16: 0.05%
  Expert 110: 0.05%
  Expert 100: 0.05%
  Expert 107: 0.05%
  Expert 94: 0.05%
  Expert 115: 0.05%
  Expert 52: 0.05%
  Expert 118: 0.05%
  Expert 121: 0.05%
  Expert 27: 0.05%
  Expert 5: 0.05%
  Expert 84: 0.05%
  Expert 77: 0.05%
  Expert 14: 0.05%
  Expert 59: 0.04%
  Expert 60: 0.04%
  Expert 108: 0.04%
  Expert 40: 0.04%
  Expert 6: 0.04%
  Expert 54: 0.04%
  Expert 2: 0.04%
  Expert 106: 0.04%
  Expert 73: 0.04%
  Expert 51: 0.04%
  Expert 98: 0.04%
  Expert 30: 0.04%
  Expert 32: 0.04%
  Expert 61: 0.04%
  Expert 37: 0.04%
  Expert 72: 0.04%
  Expert 90: 0.04%
  Expert 99: 0.04%
  Expert 112: 0.04%
  Expert 53: 0.04%
  Expert 66: 0.04%
  Expert 85: 0.04%
  Expert 88: 0.04%
  Expert 15: 0.04%
  Expert 36: 0.04%
  Expert 62: 0.03%
  Expert 26: 0.03%
  Expert 89: 0.03%
  Expert 113: 0.03%
  Expert 21: 0.03%
  Expert 68: 0.03%
  Expert 39: 0.03%
  Expert 41: 0.03%
  Expert 43: 0.03%
  Expert 87: 0.03%
  Expert 114: 0.03%
  Expert 49: 0.03%
  Expert 127: 0.03%
  Expert 105: 0.03%
  Expert 63: 0.03%
  Expert 95: 0.03%
  Expert 8: 0.03%
  Expert 124: 0.03%
  Expert 92: 0.03%
  Expert 57: 0.03%
  Expert 28: 0.02%
  Expert 123: 0.02%
  Expert 78: 0.02%
  Expert 122: 0.02%
  Expert 65: 0.02%
  Expert 109: 0.02%
  Expert 80: 0.02%
  Expert 79: 0.02%
  Expert 74: 0.02%
  Expert 58: 0.02%
  Expert 76: 0.02%
  Expert 119: 0.02%
  Expert 46: 0.02%
  Expert 103: 0.02%
  Expert 111: 0.02%
  Expert 67: 0.02%
  Expert 12: 0.02%
  Expert 11: 0.02%
  Expert 34: 0.02%
  Expert 126: 0.01%
  Expert 117: 0.01%
  Expert 19: 0.01%
  Expert 81: 0.01%
  Expert 70: 0.01%
  Expert 47: 0.01%
Total tokens per expert:
  Expert 0: 587
  Expert 1: 822
  Expert 2: 350
  Expert 3: 545
  Expert 4: 521
  Expert 5: 380
  Expert 6: 347
  Expert 7: 799
  Expert 8: 224
  Expert 9: 781
  Expert 10: 440
  Expert 11: 132
  Expert 12: 137
  Expert 13: 525
  Expert 14: 371
  Expert 15: 290
  Expert 16: 432
  Expert 17: 491
  Expert 18: 715
  Expert 19: 98
  Expert 20: 730
  Expert 21: 259
  Expert 22: 1140
  Expert 23: 560
  Expert 24: 670
  Expert 25: 737
  Expert 26: 286
  Expert 27: 380
  Expert 28: 199
  Expert 29: 465
  Expert 30: 317
  Expert 31: 582
  Expert 32: 320
  Expert 33: 960
  Expert 34: 130
  Expert 35: 523
  Expert 36: 290
  Expert 37: 311
  Expert 38: 681
  Expert 39: 250
  Expert 40: 351
  Expert 41: 245
  Expert 42: 709
  Expert 43: 253
  Expert 44: 612
  Expert 45: 458
  Expert 46: 162
  Expert 47: 80
  Expert 48: 443
  Expert 49: 233
  Expert 50: 446
  Expert 51: 336
  Expert 52: 404
  Expert 53: 298
  Expert 54: 354
  Expert 55: 442
  Expert 56: 686
  Expert 57: 204
  Expert 58: 175
  Expert 59: 361
  Expert 60: 353
  Expert 61: 315
  Expert 62: 283
  Expert 63: 227
  Expert 64: 558
  Expert 65: 198
  Expert 66: 295
  Expert 67: 147
  Expert 68: 257
  Expert 69: 453
  Expert 70: 92
  Expert 71: 467
  Expert 72: 312
  Expert 73: 341
  Expert 74: 174
  Expert 75: 963
  Expert 76: 163
  Expert 77: 369
  Expert 78: 189
  Expert 79: 177
  Expert 80: 184
  Expert 81: 98
  Expert 82: 537
  Expert 83: 496
  Expert 84: 375
  Expert 85: 299
  Expert 86: 708
  Expert 87: 248
  Expert 88: 294
  Expert 89: 277
  Expert 90: 307
  Expert 91: 433
  Expert 92: 210
  Expert 93: 479
  Expert 94: 408
  Expert 95: 221
  Expert 96: 508
  Expert 97: 710
  Expert 98: 331
  Expert 99: 306
  Expert 100: 411
  Expert 101: 869
  Expert 102: 599
  Expert 103: 156
  Expert 104: 1027
  Expert 105: 225
  Expert 106: 334
  Expert 107: 412
  Expert 108: 354
  Expert 109: 186
  Expert 110: 423
  Expert 111: 148
  Expert 112: 304
  Expert 113: 269
  Expert 114: 243
  Expert 115: 408
  Expert 116: 457
  Expert 117: 105
  Expert 118: 384
  Expert 119: 167
  Expert 120: 639
  Expert 121: 381
  Expert 122: 188
  Expert 123: 192
  Expert 124: 209
  Expert 125: 699
  Expert 126: 117
  Expert 127: 223
```

**Interpretation of the Layer Statistics:**

| Metric                              | Value                                     | Interpretation                                                                                                |
|-------------------------------------|-------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| **Forward Pass Time**               | Avg 45.36 ms (gating+stats 0.45 ms)       | Only ≈1% of the total MoE block computation is for router & statistics collection.                          |
| **Total Tokens Processed**          | 49,920 tokens                             | Distributed among 128 experts in this layer.                                                                  |
| **Mean / Median Tokens per Expert** | 390 / 349 tokens                          | Ideally, each expert would handle ≈390 tokens for perfect balance.                                            |
| **Max / Min Tokens per Expert**     | 1,140 / 80 tokens                         | The most utilized expert (No. 22) processed 14.3 times more tokens than the least utilized expert (No. 47). |
| **Std. Dev. / Coeff. of Variation** | 215 / 0.55                                | The standard deviation is 55% of the mean, indicating **significant load imbalance** among experts.         |
| **Gini Coefficient**                | 0.30                                      | Measures token distribution inequality (0=perfect equality, 1=max inequality). 0.30 suggests **moderate imbalance**. |
| **Top 5 / 10 Experts' Share**       | 9.9% / 17.7% of total tokens              | The top 5 experts handled 9.9% of tokens, and the top 10 handled 17.7%.                                       |
