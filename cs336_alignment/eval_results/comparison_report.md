## Evaluation Comparison — GSM8K test set

Model                                           format_reward   answer_reward          reward
---------------------------------------------------------------------------------------------
GRPO 200 steps   (SFT init,  t=0) [new]                0.9947          0.7165          0.7165
SFT 1 epoch      (greedy, t=0)                         0.9962          0.7142          0.7142
GRPO 200 steps   (base init, t=0)                      0.5133          0.1759          0.1759
Base model       (greedy, t=0)                         0.5011          0.1797          0.1797
Base model       (sampled, t=1)                        0.1964          0.0258          0.0258

Notes
-----
- All evals use greedy decoding (temperature=0) unless noted.
- answer_reward = exact-match on numeric answer; format_reward = correct <think>/<answer> tags.
- GRPO (base init): trained 200 steps from Qwen2.5-Math-1.5B base; barely moves above baseline.
- GRPO (SFT init): trained 200 steps from the SFT-1epoch checkpoint; slightly outperforms SFT.
