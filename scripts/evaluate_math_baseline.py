"""Evaluate Qwen2.5-Math-1.5B zero-shot performance on GSM8K.

Usage:
    python scripts/evaluate_math_baseline.py \
        --output-path /tmp/gsm8k_results.jsonl \
        --max-examples 10
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from statistics import mean
from typing import Callable

from vllm import LLM, SamplingParams
from xopen import xopen

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "cs336_alignment" / "prompts" / "r1_zero.prompt"
MODEL_PATH = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
DATASET_PATH = Path(__file__).parent.parent / "data" / "gsm8k" / "test.jsonl"


def extract_gsm8k_answer(answer: str) -> str:
    """Extract the final numeric answer from a GSM8K answer string (after ####)."""
    return answer.split("####")[-1].strip()


def load_prompt_template(path: Path) -> str:
    return path.read_text()


def format_prompt(template: str, question: str) -> str:
    return template.format(question=question)


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    eval_sampling_params: SamplingParams,
    ground_truths: list[str],
    output_path: str,
    questions: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate a language model on a list of prompts, compute evaluation metrics,
    and serialize results to disk.

    Args:
        vllm_model: Loaded vLLM LLM instance.
        reward_fn: Function mapping (response, ground_truth) -> metrics dict.
        prompts: Formatted input prompts.
        eval_sampling_params: vLLM SamplingParams for generation.
        ground_truths: Ground-truth answers parallel to prompts.
        output_path: Path to write output JSONL.
        questions: Optional original question strings for serialization.

    Returns:
        Dict of aggregated metric means.
    """
    logger.info(f"Generating completions for {len(prompts)} prompts...")
    raw_outputs = vllm_model.generate(prompts, eval_sampling_params)

    all_metrics = []
    with xopen(output_path, "w") as fout:
        for i, output in enumerate(raw_outputs):
            prompt = prompts[i]
            response = output.outputs[0].text
            ground_truth = ground_truths[i]
            metrics = reward_fn(response, ground_truth)
            all_metrics.append(metrics)

            record = {
                "ground_truth": ground_truth,
                "prompt": prompt,
                "response": response,
                "metrics": metrics,
            }
            if questions is not None:
                record["question"] = questions[i]

            fout.write(json.dumps(record) + "\n")

    aggregated = {
        key: mean(m[key] for m in all_metrics)
        for key in all_metrics[0]
    }
    for key, value in sorted(aggregated.items()):
        logger.info(f"{key}: {value:.4f}")
        print(f"{key}: {value:.4f}")

    return aggregated


def main():
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-Math-1.5B on GSM8K.")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--dataset-path", default=str(DATASET_PATH))
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))

    # Load prompt template
    template = load_prompt_template(PROMPT_PATH)

    # Load dataset
    examples = []
    with xopen(args.dataset_path) as f:
        for line in f:
            examples.append(json.loads(line))
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    logger.info(f"Loaded {len(examples)} examples from {args.dataset_path}")

    questions = [ex["question"] for ex in examples]
    ground_truths = [extract_gsm8k_answer(ex["answer"]) for ex in examples]
    prompts = [format_prompt(template, q) for q in questions]

    # Load model
    logger.info(f"Loading model from {args.model_path} ...")
    llm = LLM(model=args.model_path, dtype="bfloat16")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        max_tokens=args.max_tokens,
    )

    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        eval_sampling_params=sampling_params,
        ground_truths=ground_truths,
        output_path=args.output_path,
        questions=questions,
    )

    logger.info("finished running %s", sys.argv[0])


if __name__ == "__main__":
    main()
