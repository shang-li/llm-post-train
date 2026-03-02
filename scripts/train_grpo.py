#!/usr/bin/env python3
"""GRPO training with PPO-clip and group-relative advantages on Qwen2.5-Math-1.5B / GSM8K.

Algorithm per optimizer step
  1. Sample N_PROMPTS questions from the training set.
  2. Generate GROUP_SIZE rollouts per prompt via model.generate() (temp=1.0).
  3. Score rollouts with r1_zero_reward_fn; group-normalize to get advantages.
  4. Compute old_log_probs from the rollout policy (no-grad).
  5. Update in GRAD_ACCUM_STEPS microbatches via grpo_microbatch_train_step
     (loss_type="grpo_clip").
  6. Log scalars to TensorBoard.

After training
  - Free HF model; load final checkpoint with vLLM (greedy, full test set).
  - Save eval JSONL to cs336_alignment/eval_results/.
  - Print + save a side-by-side comparison table against all prior reports
    in that directory.

Usage
    uv run python scripts/train_grpo.py --output-dir models/grpo --num-steps 200
"""

import argparse
import gc
import json
import logging
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Callable

import torch
import torch.nn.utils as nn_utils
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from vllm import LLM, SamplingParams
from xopen import xopen

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import (
    compute_group_normalized_rewards,
    get_response_log_probs,
    grpo_microbatch_train_step,
    log_generations,
    tokenize_prompt_and_output,
)

logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH  = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
PROMPT_PATH = Path(__file__).parent.parent / "cs336_alignment" / "prompts" / "r1_zero.prompt"
TRAIN_PATH  = Path(__file__).parent.parent / "data" / "gsm8k" / "train.jsonl"
TEST_PATH   = Path(__file__).parent.parent / "data" / "gsm8k" / "test.jsonl"
EVAL_DIR    = Path(__file__).parent.parent / "cs336_alignment" / "eval_results"

# ── GRPO hyper-parameters ──────────────────────────────────────────────────────
# Rollout / advantage
GROUP_SIZE       = 8      # rollouts per prompt (G)
N_PROMPTS        = 2      # prompts per step   (N)
ROLLOUT_BATCH    = GROUP_SIZE * N_PROMPTS   # 16 total rollouts per step
NORMALIZE_STD    = True   # divide centered rewards by within-group std
ADV_EPS          = 1e-8   # denominator safety for group std

# Optimisation
MICROBATCH_SIZE  = 4      # sub-batch for gradient accumulation (safe for ~23 GB VRAM)
GRAD_ACCUM_STEPS = ROLLOUT_BATCH // MICROBATCH_SIZE   # 4
LR               = 5e-7   # GRPO requires lower LR than SFT
WEIGHT_DECAY     = 0.01
MAX_GRAD_NORM    = 1.0
WARMUP_RATIO     = 0.05
CLIPRANGE        = 0.2    # PPO clip ε

# Generation
MAX_NEW_TOKENS   = 512    # per rollout / per evaluation response

# Logging / checkpointing
CKPT_EVERY       = 100    # save checkpoint every N opt steps
EVAL_EVERY       = 50     # run log_generations every N opt steps
LOG_GEN_N        = 16     # test examples for in-loop log_generations


# ── data helpers ───────────────────────────────────────────────────────────────

def _extract_answer(answer_str: str) -> str:
    return answer_str.split("####")[-1].strip()


def load_gsm8k(path: Path, prompt_template: str) -> list[dict]:
    """Load GSM8K as list of {prompt_str, ground_truth} dicts."""
    examples = []
    with xopen(path) as f:
        for line in f:
            ex = json.loads(line)
            gt = _extract_answer(ex["answer"])
            examples.append({
                "prompt_str":   prompt_template.format(question=ex["question"]),
                "ground_truth": gt,
            })
    return examples


# ── rollout generation ─────────────────────────────────────────────────────────

def generate_rollouts(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    group_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[str]:
    """Generate group_size completions per prompt (no-grad, temp=1.0).

    Returns a flat list of length len(prompts) * group_size ordered as:
        [p0r0, p0r1, …, p0r(G-1), p1r0, …, p(N-1)r(G-1)]
    """
    model.eval()
    all_responses: list[str] = []

    with torch.no_grad():
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = enc["input_ids"].to(device)          # (1, prompt_len)
            prompt_len = input_ids.shape[1]

            # Expand to a batch of group_size for parallel sampling
            input_ids_rep = input_ids.repeat(group_size, 1)  # (G, prompt_len)
            attn_mask = torch.ones(group_size, prompt_len, dtype=torch.long, device=device)

            out_ids = model.generate(
                input_ids=input_ids_rep,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )  # (G, prompt_len + gen_len)

            for i in range(group_size):
                gen_tokens = out_ids[i, prompt_len:]
                text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                all_responses.append(text)

    model.train()
    return all_responses


# ── old log-prob computation ───────────────────────────────────────────────────

def compute_old_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    microbatch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute log p_θ(y|x) for the rollout policy before the parameter update.

    Processed in microbatches to stay within VRAM. Returns CPU tensor of shape
    (rollout_batch, seq_len).
    """
    model.eval()
    chunks: list[torch.Tensor] = []
    n = input_ids.shape[0]
    with torch.no_grad():
        for s in range(0, n, microbatch_size):
            e = min(s + microbatch_size, n)
            lp = get_response_log_probs(
                model,
                input_ids[s:e].to(device),
                labels[s:e].to(device),
            )
            chunks.append(lp["log_probs"].cpu())
    model.train()
    return torch.cat(chunks, dim=0)  # (n, seq_len)


# ── vLLM evaluation ───────────────────────────────────────────────────────────

def _run_vllm_eval(
    vllm_model: LLM,
    reward_fn: Callable,
    prompts: list[str],
    sampling_params: SamplingParams,
    ground_truths: list[str],
    output_path: Path,
) -> dict[str, float]:
    logger.info("vLLM: generating %d completions …", len(prompts))
    raw_outputs = vllm_model.generate(prompts, sampling_params)
    all_metrics: list[dict] = []
    with xopen(str(output_path), "w") as fout:
        for i, out in enumerate(raw_outputs):
            response = out.outputs[0].text
            metrics  = reward_fn(response, ground_truths[i])
            all_metrics.append(metrics)
            fout.write(json.dumps({
                "prompt":       prompts[i],
                "response":     response,
                "ground_truth": ground_truths[i],
                "metrics":      metrics,
            }) + "\n")
    aggregated = {k: mean(m[k] for m in all_metrics) for k in all_metrics[0]}
    for k, v in sorted(aggregated.items()):
        logger.info("  eval/%s = %.4f", k, v)
    return aggregated


# ── comparison report ─────────────────────────────────────────────────────────

def build_comparison_report(
    eval_dir: Path,
    display_names: dict[str, str] | None = None,
) -> str:
    """Read all JSONL files in eval_dir, aggregate metrics, return markdown table.

    display_names maps JSONL stem → human-readable name.  Unmapped stems are
    shown as-is.
    """
    display_names = display_names or {}
    all_results: dict[str, dict[str, float]] = {}

    for jsonl_path in sorted(eval_dir.glob("*.jsonl")):
        metrics_list: list[dict] = []
        try:
            with xopen(str(jsonl_path)) as f:
                for line in f:
                    rec = json.loads(line)
                    if "metrics" in rec:
                        metrics_list.append(rec["metrics"])
        except Exception as exc:
            logger.warning("Could not read %s: %s", jsonl_path, exc)
            continue
        if not metrics_list:
            continue
        name = display_names.get(jsonl_path.stem, jsonl_path.stem)
        all_results[name] = {
            k: mean(m[k] for m in metrics_list)
            for k in metrics_list[0]
        }

    if not all_results:
        return "(no evaluation results found)"

    metric_keys = sorted(next(iter(all_results.values())).keys())
    col_w  = max(len(k) for k in metric_keys) + 2
    name_w = max(len(n) for n in all_results) + 2

    header = f"{'Model':<{name_w}}" + "".join(f"  {k:>{col_w}}" for k in metric_keys)
    sep    = "-" * len(header)
    rows   = [
        f"{name:<{name_w}}" + "".join(f"  {metrics[k]:>{col_w}.4f}" for k in metric_keys)
        for name, metrics in all_results.items()
    ]
    return "\n".join([
        "## Evaluation Comparison — GSM8K test set",
        "",
        header,
        sep,
        *rows,
        "",
    ])


# ── training ──────────────────────────────────────────────────────────────────

def train(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    prompt_template = PROMPT_PATH.read_text()

    # ── tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    eos_token = tokenizer.eos_token  # <|endoftext|> for Qwen2.5-Math

    # ── data ───────────────────────────────────────────────────────────────────
    logger.info("Loading GSM8K …")
    train_examples = load_gsm8k(TRAIN_PATH, prompt_template)
    test_examples  = load_gsm8k(TEST_PATH,  prompt_template)
    logger.info("  train=%d  test=%d", len(train_examples), len(test_examples))
    train_prompts = [ex["prompt_str"]   for ex in train_examples]
    train_gts     = [ex["ground_truth"] for ex in train_examples]

    # ── model ──────────────────────────────────────────────────────────────────
    logger.info("Loading model from %s …", args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    ).to(device)
    model.gradient_checkpointing_enable()
    model.train()

    # ── optimizer & scheduler ──────────────────────────────────────────────────
    total_steps  = args.num_steps
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    # Recompute batch sizes from CLI args (may differ from module-level constants)
    rollout_batch    = args.group_size * args.n_prompts
    grad_accum_steps = rollout_batch // MICROBATCH_SIZE

    logger.info(
        "Training: %d steps  (warmup=%d)  lr=%.1e  G=%d  N=%d  rollout_batch=%d  grad_accum=%d  ε=%.2f",
        total_steps, warmup_steps, args.lr, args.group_size, args.n_prompts,
        rollout_batch, grad_accum_steps, args.cliprange,
    )

    writer   = SummaryWriter(log_dir=str(out_dir / "tb_logs"))
    rng      = random.Random(42)

    # ── step loop ──────────────────────────────────────────────────────────────
    for step in range(1, total_steps + 1):

        # 1. sample a mini-batch of prompts ────────────────────────────────────
        indices        = rng.sample(range(len(train_examples)), args.n_prompts)
        batch_prompts  = [train_prompts[i] for i in indices]
        batch_gts      = [train_gts[i]     for i in indices]

        # 2. generate rollouts ─────────────────────────────────────────────────
        # order: [p0r0,…,p0r(G-1), p1r0,…, p(N-1)r(G-1)]
        rollout_responses = generate_rollouts(
            model, tokenizer, batch_prompts, args.group_size, MAX_NEW_TOKENS, device
        )
        repeated_prompts = [p for p in batch_prompts for _ in range(args.group_size)]
        repeated_gts     = [g for g in batch_gts     for _ in range(args.group_size)]

        # 3. rewards → group-normalised advantages ─────────────────────────────
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_gts,
            group_size=args.group_size,
            advantage_eps=ADV_EPS,
            normalize_by_std=NORMALIZE_STD,
        )
        # advantages: (ROLLOUT_BATCH,) → (ROLLOUT_BATCH, 1) for per-token broadcast
        advantages_2d = advantages.unsqueeze(1).to(device)

        # 4. tokenise rollout batch ────────────────────────────────────────────
        # Append EOS to mark response end (mirrors SFT training format)
        output_strs = [r + eos_token for r in rollout_responses]
        tokenized = tokenize_prompt_and_output(
            prompt_strs=repeated_prompts,
            output_strs=output_strs,
            tokenizer=tokenizer,
        )
        input_ids     = tokenized["input_ids"]      # (ROLLOUT_BATCH, seq_len)
        labels        = tokenized["labels"]          # (ROLLOUT_BATCH, seq_len)
        response_mask = tokenized["response_mask"]   # (ROLLOUT_BATCH, seq_len)

        # 5. old log-probs from the rollout policy (no-grad) ───────────────────
        old_log_probs = compute_old_log_probs(
            model, input_ids, labels, MICROBATCH_SIZE, device
        )  # (ROLLOUT_BATCH, seq_len) on CPU

        # 6. gradient-accumulation training loop ───────────────────────────────
        model.train()
        optimizer.zero_grad()
        step_losses:     list[float] = []
        step_clip_fracs: list[float] = []

        for mb_idx in range(grad_accum_steps):
            s = mb_idx * MICROBATCH_SIZE
            e = s + MICROBATCH_SIZE

            mb_input_ids     = input_ids[s:e].to(device)
            mb_labels        = labels[s:e].to(device)
            mb_response_mask = response_mask[s:e].to(device)
            mb_advantages    = advantages_2d[s:e]             # already on device
            mb_old_log_probs = old_log_probs[s:e].to(device)  # (mb, seq_len)

            lp_dict = get_response_log_probs(model, mb_input_ids, mb_labels)

            loss, meta = grpo_microbatch_train_step(
                policy_log_probs=lp_dict["log_probs"],
                response_mask=mb_response_mask,
                gradient_accumulation_steps=grad_accum_steps,
                loss_type="grpo_clip",
                advantages=mb_advantages,
                old_log_probs=mb_old_log_probs,
                cliprange=args.cliprange,
            )
            step_losses.append(loss.item())
            if "is_clipped" in meta:
                step_clip_fracs.append(meta["is_clipped"].item())

        nn_utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        # 7. TensorBoard logging ───────────────────────────────────────────────
        avg_loss = mean(step_losses)
        writer.add_scalar("train/loss",        avg_loss,                   step)
        writer.add_scalar("train/lr",          scheduler.get_last_lr()[0], step)
        writer.add_scalar("train/mean_reward", reward_meta["mean_reward"], step)
        writer.add_scalar("train/std_reward",  reward_meta["std_reward"],  step)
        writer.add_scalar("train/max_reward",  reward_meta["max_reward"],  step)
        writer.add_scalar("train/min_reward",  reward_meta["min_reward"],  step)
        if step_clip_fracs:
            writer.add_scalar("train/clip_frac", mean(step_clip_fracs), step)

        if step % 10 == 0:
            logger.info(
                "step %4d | loss=%7.4f | reward=%.3f (std=%.3f, max=%.2f) | lr=%.2e",
                step, avg_loss,
                reward_meta["mean_reward"], reward_meta["std_reward"],
                reward_meta["max_reward"],
                scheduler.get_last_lr()[0],
            )

        # 8. periodic log_generations eval (HF model on small test subset) ─────
        if step % EVAL_EVERY == 0:
            logger.info("log_generations eval at step %d …", step)
            torch.cuda.empty_cache()
            sample = test_examples[:LOG_GEN_N]
            gen_metrics = log_generations(
                model=model,
                tokenizer=tokenizer,
                prompts=[ex["prompt_str"]   for ex in sample],
                ground_truths=[ex["ground_truth"] for ex in sample],
                reward_fn=r1_zero_reward_fn,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=1.0,
            )
            for k, v in gen_metrics.items():
                writer.add_scalar(f"gen/{k}", v, step)
            logger.info(
                "  gen @step%d: %s",
                step, {k: f"{v:.4f}" for k, v in gen_metrics.items()},
            )

        # 9. periodic checkpoint ───────────────────────────────────────────────
        if step % CKPT_EVERY == 0 or step == total_steps:
            ckpt_dir = out_dir / f"grpo-step{step}"
            logger.info("Saving checkpoint → %s", ckpt_dir)
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))

    # ══ post-training vLLM evaluation ════════════════════════════════════════
    final_ckpt = out_dir / f"grpo-step{total_steps}"
    logger.info("Training complete.  Checkpoint: %s", final_ckpt)

    logger.info("Freeing HF model; loading vLLM for full-test evaluation …")
    # Delete optimizer/scheduler before the model: they hold references to
    # model.parameters() that would prevent GPU memory from being freed.
    del optimizer, scheduler
    del model
    gc.collect()
    torch.cuda.empty_cache()

    vllm_model = LLM(model=str(final_ckpt), dtype="bfloat16")
    eval_params = SamplingParams(
        temperature=0.0,                   # greedy for reproducibility
        stop=["</answer>"],
        include_stop_str_in_output=True,
        max_tokens=MAX_NEW_TOKENS,
    )

    eval_stem       = f"gsm8k_results_grpo{total_steps}steps_t0"
    eval_output_path = EVAL_DIR / f"{eval_stem}.jsonl"
    eval_metrics = _run_vllm_eval(
        vllm_model=vllm_model,
        reward_fn=r1_zero_reward_fn,
        prompts=[ex["prompt_str"]   for ex in test_examples],
        sampling_params=eval_params,
        ground_truths=[ex["ground_truth"] for ex in test_examples],
        output_path=eval_output_path,
    )
    for k, v in eval_metrics.items():
        writer.add_scalar(f"eval/{k}", v, total_steps)
    logger.info("Final eval: %s", {k: f"{v:.4f}" for k, v in eval_metrics.items()})

    del vllm_model
    gc.collect()
    torch.cuda.empty_cache()

    # ══ comparison report ════════════════════════════════════════════════════
    display_names = {
        "gsm8k_results_t0":           "Base model  (greedy, t=0)",
        "gsm8k_results_t1":           "Base model  (sampled, t=1)",
        "gsm8k_results_sft1epoch_t0": "SFT 1 epoch (greedy, t=0)",
        eval_stem:                    f"GRPO {total_steps} steps (greedy, t=0)",
    }
    report      = build_comparison_report(EVAL_DIR, display_names)
    report_path = EVAL_DIR / "comparison_report.md"
    report_path.write_text(report + f"\n_Generated after {total_steps} GRPO steps._\n")
    logger.info("Comparison report → %s", report_path)
    print("\n" + report)

    writer.close()
    logger.info("Done.  Checkpoints in %s", out_dir)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(
        description="GRPO fine-tuning of Qwen2.5-Math-1.5B on GSM8K"
    )
    parser.add_argument(
        "--output-dir", default="models/grpo",
        help="Root directory for checkpoints and TensorBoard logs",
    )
    parser.add_argument(
        "--model-path", default=MODEL_PATH,
        help="Path to base / pre-trained model",
    )
    parser.add_argument(
        "--num-steps", type=int, default=200,
        help="Number of optimizer steps",
    )
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE,
                        help="Rollouts per prompt (G)")
    parser.add_argument("--n-prompts",  type=int, default=N_PROMPTS,
                        help="Prompts per rollout batch (N)")
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--cliprange",  type=float, default=CLIPRANGE,
                        help="PPO clip epsilon ε")
    args = parser.parse_args()

    # Validate that rollout batch is divisible by microbatch size
    rollout_batch = args.group_size * args.n_prompts
    if rollout_batch % MICROBATCH_SIZE != 0:
        parser.error(
            f"group_size × n_prompts = {rollout_batch} must be divisible "
            f"by MICROBATCH_SIZE={MICROBATCH_SIZE}"
        )

    logger.info("Command: %s", " ".join(sys.argv))
    train(args)


if __name__ == "__main__":
    main()
