from __future__ import annotations

import logging
from typing import Any, Callable

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    # Tokenize prompts and outputs separately (no special tokens added)
    prompt_ids_list = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
    output_ids_list = tokenizer(output_strs, add_special_tokens=False)["input_ids"]

    # Concatenate per-example and record lengths
    full_ids_list = [p + o for p, o in zip(prompt_ids_list, output_ids_list)]
    prompt_lens = [len(p) for p in prompt_ids_list]
    output_lens = [len(o) for o in output_ids_list]
    max_len = max(len(f) for f in full_ids_list)

    # Right-pad all sequences to max_len
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    batch_size = len(full_ids_list)
    full_padded = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(full_ids_list):
        full_padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    # input_ids: all tokens except the last one → (batch_size, max_len - 1)
    input_ids = full_padded[:, :-1]

    # labels: shifted by 1 (all tokens except the first) → (batch_size, max_len - 1)
    labels = full_padded[:, 1:]

    # response_mask: 1 for response token positions in `labels`, 0 for prompt/padding
    # In labels (= full[1:]), response tokens occupy indices [P-1, P, ..., P+O-2]
    response_mask = torch.zeros(batch_size, max_len - 1, dtype=torch.long)
    for i, (p_len, o_len) in enumerate(zip(prompt_lens, output_lens)):
        response_start = p_len - 1        # index of first response token in labels
        response_end = p_len + o_len - 1  # exclusive end
        if response_start < response_end:
            response_mask[i, response_start:response_end] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute per-token entropy of next-token predictions over the vocab dimension.

    Args:
        logits: torch.Tensor of shape (batch_size, sequence_length, vocab_size).

    Returns:
        torch.Tensor of shape (batch_size, sequence_length).
    """
    log_probs = F.log_softmax(logits, dim=-1)   # numerically stable via logsumexp
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities log p_θ(x_t | x_{<t}).

    Args:
        model: HuggingFace causal LM.
        input_ids: torch.Tensor of shape (batch_size, sequence_length).
        labels: torch.Tensor of shape (batch_size, sequence_length), shifted input_ids.
        return_token_entropy: if True, also return per-token entropy.

    Returns:
        dict with:
            "log_probs": (batch_size, sequence_length) — conditional log-probs indexed by labels.
            "token_entropy": (batch_size, sequence_length) — present only if return_token_entropy=True.
    """
    logits = model(input_ids).logits                            # (B, T, V)
    log_probs_all = F.log_softmax(logits, dim=-1)               # (B, T, V)
    log_probs = log_probs_all.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, T)

    result: dict[str, torch.Tensor] = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)       # (B, T)
    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum masked elements along a dimension and divide by normalize_constant.

    Args:
        tensor: torch.Tensor to sum and normalize.
        mask: same shape as tensor; only positions where mask==1 contribute.
        dim: dimension to sum along. If None, sum over all elements.
        normalize_constant: value to divide the sum by.

    Returns:
        torch.Tensor of the normalized masked sum.
    """
    masked = tensor * mask
    summed = masked.sum() if dim is None else masked.sum(dim=dim)
    return summed / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the SFT cross-entropy loss for a microbatch and backprop gradients.

    Loss = -masked_sum / (batch_size × normalize_constant × gradient_accumulation_steps).
    This is equivalent to a mean over the batch, divided by normalize_constant and
    gradient_accumulation_steps, so gradients average correctly across microbatches.

    Args:
        policy_log_probs: (batch_size, sequence_length) per-token log-probs.
        response_mask: (batch_size, sequence_length) 1 for response tokens.
        gradient_accumulation_steps: number of microbatches per optimizer step.
        normalize_constant: additional denominator for the masked sum.

    Returns:
        (loss, metadata): loss is the gradient-adjusted scalar (what backward was called on).
    """
    batch_size = policy_log_probs.shape[0]
    loss = -masked_normalize(
        policy_log_probs,
        response_mask,
        normalize_constant=normalize_constant * batch_size * gradient_accumulation_steps,
    )
    loss.backward()
    return loss, {}


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """Mean of tensor over masked positions.

    Args:
        tensor: data tensor.
        mask: same shape as tensor; 1 = include, 0 = exclude.
        dim: dimension to average along. If None, average over all masked elements.

    Returns:
        Masked mean with the same shape semantics as tensor.mean(dim).
    """
    masked = tensor * mask
    if dim is None:
        return masked.sum() / mask.sum()
    return masked.sum(dim=dim) / mask.sum(dim=dim)


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token policy gradient loss.

    Args:
        raw_rewards_or_advantages: (batch_size, 1) scalar per sequence.
        policy_log_probs: (batch_size, sequence_length) per-token log-probs.

    Returns:
        (batch_size, sequence_length) per-token loss.
    """
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute per-token GRPO-Clip loss.

    Args:
        advantages: (batch_size, 1) per-example advantages A.
        policy_log_probs: (batch_size, sequence_length) log-probs of current policy.
        old_log_probs: (batch_size, sequence_length) log-probs of old policy.
        cliprange: clip parameter ε.

    Returns:
        (loss, metadata): loss is (batch_size, sequence_length) per-token clipped loss.
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - cliprange, 1 + cliprange) * advantages
    loss = -torch.min(unclipped, clipped)
    metadata = {"is_clipped": clipped < unclipped}
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dispatch to the appropriate policy gradient loss function.

    Args:
        policy_log_probs: (batch_size, sequence_length).
        loss_type: one of "no_baseline", "reinforce_with_baseline", "grpo_clip".
        raw_rewards: (batch_size, 1), required for "no_baseline".
        advantages: (batch_size, 1), required for "reinforce_with_baseline" and "grpo_clip".
        old_log_probs: (batch_size, sequence_length), required for "grpo_clip".
        cliprange: float, required for "grpo_clip".

    Returns:
        (loss, metadata) where loss is (batch_size, sequence_length).
    """
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    elif loss_type == "grpo_clip":
        assert advantages is not None and old_log_probs is not None and cliprange is not None
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: str,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Execute a forward-and-backward pass on a microbatch.

    Computes the per-token policy gradient loss, aggregates via masked mean,
    scales by gradient_accumulation_steps, then calls loss.backward().

    Args:
        policy_log_probs: (batch_size, sequence_length) per-token log-probs.
        response_mask: (batch_size, sequence_length) 1 for response tokens.
        gradient_accumulation_steps: number of microbatches per optimizer step.
        loss_type: one of "no_baseline", "reinforce_with_baseline", "grpo_clip".
        raw_rewards: (batch_size, 1), required for "no_baseline".
        advantages: (batch_size, 1), required for "reinforce_with_baseline" / "grpo_clip".
        old_log_probs: (batch_size, sequence_length), required for "grpo_clip".
        cliprange: clip parameter ε, required for "grpo_clip".

    Returns:
        (loss, metadata): loss is the gradient-adjusted scalar; metadata contains
            aggregated statistics from the underlying loss call.
    """
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    loss = masked_mean(per_token_loss, response_mask) / gradient_accumulation_steps
    loss.backward()

    agg_metadata: dict[str, torch.Tensor] = {}
    for k, v in metadata.items():
        if isinstance(v, torch.Tensor):
            agg_metadata[k] = masked_mean(v.float(), response_mask)

    return loss, agg_metadata


def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute per-group normalized rewards (GRPO advantages).

    Args:
        reward_fn: Callable(response, ground_truth) -> dict with "reward" key.
        rollout_responses: list of rollout strings, length = rollout_batch_size.
        repeated_ground_truths: list of ground-truth strings, length = rollout_batch_size.
        group_size: number of rollouts per question.
        advantage_eps: epsilon added to std to avoid division by zero.
        normalize_by_std: if True, divide centered rewards by (std + advantage_eps).

    Returns:
        (advantages, raw_rewards, metadata)
    """
    rollout_batch_size = len(rollout_responses)

    raw_rewards_list = [
        reward_fn(resp, gt)["reward"]
        for resp, gt in zip(rollout_responses, repeated_ground_truths)
    ]
    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)

    n_groups = rollout_batch_size // group_size
    rewards_grouped = raw_rewards.view(n_groups, group_size)

    group_means = rewards_grouped.mean(dim=1, keepdim=True)
    advantages_grouped = rewards_grouped - group_means

    if normalize_by_std:
        group_stds = rewards_grouped.std(dim=1, keepdim=True, unbiased=True)
        advantages_grouped = advantages_grouped / (group_stds + advantage_eps)

    advantages = advantages_grouped.view(rollout_batch_size)

    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std().item(),
        "max_reward": raw_rewards.max().item(),
        "min_reward": raw_rewards.min().item(),
    }

    return advantages, raw_rewards, metadata


def log_generations(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    max_new_tokens: int = 512,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Generate responses from the model and log per-example and aggregate statistics.

    Logs for each example:
      1. The input prompt
      2. The model's generated response
      3. The ground-truth answer
      4. Reward info (format_reward, answer_reward, reward)
      5. Average token entropy of the response
      6. Response length

    Args:
        model: HuggingFace causal LM (already on the correct device).
        tokenizer: Matching tokenizer.
        prompts: List of formatted input prompt strings.
        ground_truths: Parallel list of ground-truth answer strings.
        reward_fn: Callable(response, ground_truth) -> dict with at least
            "reward", "format_reward", "answer_reward" keys.
        max_new_tokens: Maximum tokens to generate per prompt.
        temperature: Sampling temperature (1.0 = unmodified logits).

    Returns:
        dict with aggregate scalar metrics:
            "avg_reward", "avg_format_reward", "avg_answer_reward",
            "avg_token_entropy", "avg_response_len",
            "avg_response_len_correct", "avg_response_len_incorrect".
        Also logs each example via the module logger at INFO level.
    """
    model.eval()
    device = next(model.parameters()).device

    all_rewards: list[dict[str, float]] = []
    response_lens: list[int] = []
    correct_lens: list[int] = []
    incorrect_lens: list[int] = []
    all_entropies: list[float] = []

    # Process one example at a time to avoid OOM from stacking large score tensors.
    # (batch_size=16 × max_new_tokens=1024 × vocab=151K in bf16 ≈ 5 GB)
    for i, (prompt, gt) in enumerate(zip(prompts, ground_truths)):
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature != 0.0),
                temperature=temperature if temperature != 0.0 else None,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_ids = outputs.sequences[0, prompt_len:]   # (gen_len,)
        non_pad  = (gen_ids != tokenizer.pad_token_id).sum().item()
        response = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Entropy: stack per-step scores → (gen_len, V), compute in one shot, free immediately
        if outputs.scores:
            scores_2d = torch.stack(outputs.scores, dim=0)          # (gen_len, V)
            entropy_per_tok = compute_entropy(scores_2d.unsqueeze(0)).squeeze(0)  # (gen_len,)
            avg_ent = entropy_per_tok[:non_pad].mean().item() if non_pad > 0 else 0.0
            del scores_2d, entropy_per_tok
        else:
            avg_ent = 0.0

        reward_info = reward_fn(response, gt)
        all_rewards.append(reward_info)
        response_lens.append(non_pad)
        is_correct = reward_info.get("answer_reward", 0.0) > 0.0
        (correct_lens if is_correct else incorrect_lens).append(non_pad)
        all_entropies.append(avg_ent)

        logger.info(
            "\n--- Generation %d ---"
            "\nPROMPT:       %s"
            "\nRESPONSE:     %s"
            "\nGROUND TRUTH: %s"
            "\nREWARDS:      %s"
            "\nAVG ENTROPY:  %.4f"
            "\nRESP LEN:     %d",
            i, prompt, response, gt, reward_info, avg_ent, non_pad,
        )

    def _safe_mean(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else float("nan")

    metrics = {
        "avg_reward":              _safe_mean([r["reward"] for r in all_rewards]),
        "avg_format_reward":       _safe_mean([r["format_reward"] for r in all_rewards]),
        "avg_answer_reward":       _safe_mean([r["answer_reward"] for r in all_rewards]),
        "avg_token_entropy":       _safe_mean(all_entropies),
        "avg_response_len":        _safe_mean([float(l) for l in response_lens]),
        "avg_response_len_correct":   _safe_mean([float(l) for l in correct_lens]),
        "avg_response_len_incorrect": _safe_mean([float(l) for l in incorrect_lens]),
    }

    logger.info("Generation metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    model.train()
    return metrics
