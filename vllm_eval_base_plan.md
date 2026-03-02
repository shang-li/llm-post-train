# Plan: Math Baseline Evaluation Module

## Context
Create a script to evaluate Qwen2.5-Math-1.5B zero-shot performance on GSM8K. The script uses the `r1_zero` prompt template to format questions, generates completions via vLLM, grades answers using `r1_zero_reward_fn` from `drgrpo_grader.py`, and serializes all results to disk for downstream analysis.

## Key Files
- **Create**: `scripts/evaluate_math_baseline.py`
- **Read (r1_zero prompt)**: `cs336_alignment/prompts/r1_zero.prompt`
- **Read (reward/grading)**: `cs336_alignment/drgrpo_grader.py` — use `r1_zero_reward_fn`
- **Dataset**: `data/gsm8k/test.jsonl` — format: `{"question": str, "answer": str}` where answer ends with `#### <final_answer>`
- **Reference script**: `scripts/evaluate_safety.py` — pattern for vLLM evaluation loops
- **Model path**: `/data/a5-alignment/models/Qwen2.5-Math-1.5B`

## Data Notes
- GSM8K ground truth: extract the numeric answer after `####` in the `answer` field
  - e.g. `"...calculation...\n#### 18"` → ground truth is `"18"`
- The `r1_zero.prompt` template: raw text prompt (not chat template), ends with `Assistant: <think>`
  - Completion expected: `...reasoning...</think> <answer>answer</answer>`
- `r1_zero_reward_fn(response, ground_truth)` expects response to contain `</think> <answer>` and `</answer>`

## Implementation Plan

### `scripts/evaluate_math_baseline.py`

#### 1. `extract_gsm8k_answer(answer: str) -> str`
Strip reasoning from GSM8K answer field, returning only the part after `####`.
```python
return answer.split("####")[-1].strip()
```

#### 2. `load_r1_zero_prompt() -> str`
Read `cs336_alignment/prompts/r1_zero.prompt` and return the template string.

#### 3. `format_prompt(template: str, question: str) -> str`
Return `template.format(question=question)`.

#### 4. `evaluate_vllm(vllm_model, reward_fn, prompts, eval_sampling_params, ground_truths, output_path) -> dict`
Signature matches assignment spec, with extra `ground_truths` and `output_path` args:
- Call `vllm_model.generate(prompts, eval_sampling_params)`
- For each output, extract `.outputs[0].text` as `response`
- Call `reward_fn(response, ground_truth)` → dict with `format_reward`, `answer_reward`, `reward`
- Write one JSONL line per example to `output_path`:
  ```json
  {"question": ..., "ground_truth": ..., "prompt": ..., "response": ..., "metrics": {...}}
  ```
- Compute and print aggregate metrics (mean of each metric key)
- Return metrics dict

#### 5. `main()` via `argparse`
Args:
- `--model-path` (default: `/data/a5-alignment/models/Qwen2.5-Math-1.5B`)
- `--dataset-path` (default: `data/gsm8k/test.jsonl`)
- `--output-path` (required)
- `--max-examples` (optional int, for quick testing)
- `--temperature` (default: `0.0`)
- `--max-tokens` (default: `2048`)

Steps:
1. Load model via `LLM(model=model_path, dtype="bfloat16")`
2. Set `SamplingParams(temperature=..., stop=["</answer>"], include_stop_str_in_output=True, max_tokens=...)`
3. Read dataset, extract questions and ground truths
4. Format prompts using r1_zero template
5. Call `evaluate_vllm(...)`

## Sampling Parameters
```python
SamplingParams(
    temperature=1.0,
    stop=["</answer>"],
    include_stop_str_in_output=True,
    max_tokens=2048,
)
```

## Output Format (JSONL)
Each line:
```json
{
  "question": "...",
  "ground_truth": "18",
  "prompt": "A conversation between...\nUser: ...\nAssistant: <think>",
  "response": "...reasoning...</think> <answer>18</answer>",
  "metrics": {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}
}
```

## Verification
Run:
```bash
python scripts/evaluate_math_baseline.py \
    --output-path /tmp/gsm8k_results.jsonl \
    --max-examples 10
```
Check that `/tmp/gsm8k_results.jsonl` has 10 lines with the correct structure, and metrics are printed to stdout.
