#!/usr/bin/env python3
"""SFT fine-tuning of Qwen2.5-Math-1.5B on GSM8K using the r1_zero format.

Each training example is formatted as:
    prompt  = r1_zero template (ends with '<think>')
    output  = '{chain_of_thought} </think> <answer> {answer} </answer><|endoftext|>'

The response_mask covers only the output tokens, so the SFT loss is
computed exclusively on the model's chain-of-thought + answer.

Usage:
    uv run python scripts/train_sft.py --output-dir models --num-epochs 1
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from statistics import mean
from typing import Callable

import torch
import torch.nn.utils as nn_utils
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from vllm import LLM, SamplingParams
from xopen import xopen

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import (
    get_response_log_probs,
    log_generations,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)

logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH   = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
PROMPT_PATH  = Path(__file__).parent.parent / "cs336_alignment" / "prompts" / "r1_zero.prompt"
TRAIN_PATH   = Path(__file__).parent.parent / "data" / "gsm8k" / "train.jsonl"
TEST_PATH    = Path(__file__).parent.parent / "data" / "gsm8k" / "test.jsonl"

# ── hyper-parameters ───────────────────────────────────────────────────────────
EFFECTIVE_BATCH  = 16
MICROBATCH_SIZE  = 4                          # safe for L4 (23 GB)
GRAD_ACCUM_STEPS = EFFECTIVE_BATCH // MICROBATCH_SIZE   # 4
LR               = 2e-5
WEIGHT_DECAY     = 0.01
MAX_GRAD_NORM    = 1.0
WARMUP_RATIO     = 0.05
MAX_NEW_TOKENS   = 1024
LOG_GEN_N        = 16   # examples for in-loop log_generations (HF model)
MAX_SEQ_LEN      = 512  # drop ~1% of examples longer than this to avoid OOM


# ── data helpers ───────────────────────────────────────────────────────────────

def _extract_answer(answer_str: str) -> str:
    return answer_str.split("####")[-1].strip()


def _extract_cot(answer_str: str) -> str:
    return answer_str.split("####")[0].strip()


def load_gsm8k(
    path: Path,
    prompt_template: str,
    eos_token: str,
    tokenizer,
    max_seq_len: int | None = None,
) -> list[dict]:
    """Return list of {prompt_str, output_str, ground_truth} dicts.

    If max_seq_len is set, examples whose full tokenized length exceeds it
    are dropped (avoids GPU OOM on rare very-long chain-of-thought examples).
    """
    examples, dropped = [], 0
    with xopen(path) as f:
        for line in f:
            ex = json.loads(line)
            gt  = _extract_answer(ex["answer"])
            cot = _extract_cot(ex["answer"])
            prompt_str = prompt_template.format(question=ex["question"])
            output_str = f" {cot} </think> <answer> {gt} </answer>{eos_token}"
            if max_seq_len is not None:
                full_len = len(tokenizer(
                    prompt_str + output_str, add_special_tokens=False
                )["input_ids"])
                if full_len > max_seq_len:
                    dropped += 1
                    continue
            examples.append({
                "prompt_str":   prompt_str,
                "output_str":   output_str,
                "ground_truth": gt,
            })
    if dropped:
        logger.info("  dropped %d examples exceeding max_seq_len=%d", dropped, max_seq_len)
    return examples


class SFTDataset(Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def make_collate_fn(tokenizer):
    def collate(batch):
        tokenized = tokenize_prompt_and_output(
            prompt_strs=[ex["prompt_str"]  for ex in batch],
            output_strs=[ex["output_str"]  for ex in batch],
            tokenizer=tokenizer,
        )
        return tokenized
    return collate


# ── vLLM evaluation (inlined to avoid import path issues) ──────────────────────

def _run_vllm_eval(
    vllm_model: LLM,
    reward_fn: Callable,
    prompts: list[str],
    sampling_params: SamplingParams,
    ground_truths: list[str],
    output_path: Path,
) -> dict[str, float]:
    logger.info("vLLM: generating %d completions ...", len(prompts))
    raw_outputs = vllm_model.generate(prompts, sampling_params)
    all_metrics = []
    with xopen(str(output_path), "w") as fout:
        for i, out in enumerate(raw_outputs):
            response = out.outputs[0].text
            metrics  = reward_fn(response, ground_truths[i])
            all_metrics.append(metrics)
            fout.write(json.dumps({
                "prompt": prompts[i], "response": response,
                "ground_truth": ground_truths[i], "metrics": metrics,
            }) + "\n")
    aggregated = {k: mean(m[k] for m in all_metrics) for k in all_metrics[0]}
    for k, v in sorted(aggregated.items()):
        logger.info("  eval/%s = %.4f", k, v)
    return aggregated


# ── training ───────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = PROMPT_PATH.read_text()

    # ── tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # Qwen2.5-Math: pad_token == eos_token == <|endoftext|>; no BOS.
    eos_token = tokenizer.eos_token

    # ── datasets ───────────────────────────────────────────────────────────────
    logger.info("Loading GSM8K ...")
    train_examples = load_gsm8k(TRAIN_PATH, prompt_template, eos_token, tokenizer, MAX_SEQ_LEN)
    test_examples  = load_gsm8k(TEST_PATH,  prompt_template, eos_token, tokenizer)
    logger.info("  train=%d  test=%d", len(train_examples), len(test_examples))

    train_loader = DataLoader(
        SFTDataset(train_examples),
        batch_size=MICROBATCH_SIZE,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer),
        drop_last=True,   # keeps microbatch counts clean for grad accumulation
    )

    # ── model ──────────────────────────────────────────────────────────────────
    logger.info("Loading model from %s ...", MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16
    ).to(device)
    model.gradient_checkpointing_enable()   # trade compute for activation memory
    model.train()

    # ── optimizer & scheduler ──────────────────────────────────────────────────
    microbatches_per_epoch  = len(train_loader)
    opt_steps_per_epoch     = microbatches_per_epoch // GRAD_ACCUM_STEPS
    total_opt_steps         = opt_steps_per_epoch * args.num_epochs
    warmup_steps            = max(1, int(total_opt_steps * WARMUP_RATIO))
    logger.info(
        "Optimizer steps: %d/epoch × %d epochs = %d total  (warmup=%d)",
        opt_steps_per_epoch, args.num_epochs, total_opt_steps, warmup_steps,
    )

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_opt_steps,
    )

    writer      = SummaryWriter(log_dir=str(out_dir / "tb_logs"))
    opt_step    = 0    # counts optimizer (not microbatch) steps

    # ── epoch loop ─────────────────────────────────────────────────────────────
    for epoch in range(1, args.num_epochs + 1):
        logger.info("=== Epoch %d / %d ===", epoch, args.num_epochs)
        model.train()
        optimizer.zero_grad()

        loss_accum  = 0.0
        loss_count  = 0

        for mb_idx, batch in enumerate(train_loader):
            input_ids     = batch["input_ids"].to(device)
            labels        = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)

            log_probs_dict = get_response_log_probs(
                model, input_ids, labels, return_token_entropy=False
            )

            loss, _ = sft_microbatch_train_step(
                policy_log_probs=log_probs_dict["log_probs"],
                response_mask=response_mask,
                gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            )
            loss_accum += loss.item()
            loss_count += 1

            # ── optimizer step every GRAD_ACCUM_STEPS microbatches ─────────────
            if (mb_idx + 1) % GRAD_ACCUM_STEPS == 0:
                nn_utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

                avg_loss = loss_accum / loss_count
                loss_accum, loss_count = 0.0, 0

                writer.add_scalar("train/loss", avg_loss, opt_step)
                writer.add_scalar("train/lr",   scheduler.get_last_lr()[0], opt_step)

                if opt_step % 50 == 0:
                    logger.info(
                        "  step %4d | loss=%.4f | lr=%.2e",
                        opt_step, avg_loss, scheduler.get_last_lr()[0],
                    )

        logger.info("Epoch %d: training complete (%d opt steps total).", epoch, opt_step)

        # ── save checkpoint first (before any inference that could OOM) ────────
        ckpt_dir = out_dir / f"sft-{epoch}epochs"
        logger.info("Saving checkpoint → %s", ckpt_dir)
        model.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))

        # ── log_generations on small test subset (HF model) ───────────────────
        logger.info("Logging generations on %d test examples ...", LOG_GEN_N)
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
            writer.add_scalar(f"gen/{k}", v, epoch)
        logger.info("  gen metrics: %s", {k: f"{v:.4f}" for k, v in gen_metrics.items()})

        # ── evaluate with vLLM (free HF model first to fit in VRAM) ───────────
        logger.info("Freeing HF model; loading vLLM for evaluation ...")
        del model
        gc.collect()
        torch.cuda.empty_cache()

        vllm_model = LLM(model=str(ckpt_dir), dtype="bfloat16")
        eval_params = SamplingParams(
            temperature=0.0,               # greedy for reproducibility
            stop=["</answer>"],
            include_stop_str_in_output=True,
            max_tokens=MAX_NEW_TOKENS,
        )
        eval_metrics = _run_vllm_eval(
            vllm_model=vllm_model,
            reward_fn=r1_zero_reward_fn,
            prompts=[ex["prompt_str"]   for ex in test_examples],
            sampling_params=eval_params,
            ground_truths=[ex["ground_truth"] for ex in test_examples],
            output_path=out_dir / f"eval_epoch{epoch}.jsonl",
        )
        for k, v in eval_metrics.items():
            writer.add_scalar(f"eval/{k}", v, epoch)
        logger.info("Eval epoch %d: %s", epoch, {k: f"{v:.4f}" for k, v in eval_metrics.items()})

        del vllm_model
        gc.collect()
        torch.cuda.empty_cache()

        # ── reload HF model if more epochs remain ─────────────────────────────
        if epoch < args.num_epochs:
            logger.info("Reloading model from checkpoint for next epoch ...")
            model = AutoModelForCausalLM.from_pretrained(
                str(ckpt_dir), torch_dtype=torch.bfloat16
            ).to(device)
            model.gradient_checkpointing_enable()
            model.train()
            optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            # Note: scheduler state is not restored; re-creating is intentional
            # for simplicity when training additional epochs interactively.

    writer.close()
    logger.info("Training complete. Checkpoints saved to %s", out_dir)


def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description="SFT training for Qwen2.5-Math-1.5B on GSM8K")
    parser.add_argument("--output-dir", default="models",
                        help="Root directory for checkpoints and TensorBoard logs")
    parser.add_argument("--num-epochs", type=int, default=1)
    args = parser.parse_args()
    logger.info("Command: %s", " ".join(sys.argv))
    train(args)


if __name__ == "__main__":
    main()
