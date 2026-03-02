"""Quick test of Qwen2.5-Math-1.5B via vLLM."""

from vllm import LLM, SamplingParams

MODEL_PATH = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"

PROMPTS = [
    "What is 2 + 2?",
    "Solve for x: 2x + 5 = 13",
    "What is the derivative of x^2?",
]

def main():
    print(f"Loading model from {MODEL_PATH} ...")
    llm = LLM(model=MODEL_PATH, dtype="bfloat16")

    sampling_params = SamplingParams(
        temperature=1.0,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        max_tokens=256
    )

    print("Running inference...\n")
    outputs = llm.generate(PROMPTS, sampling_params)

    for output in outputs:
        prompt = output.prompt
        response = output.outputs[0].text
        print(f"Prompt:   {prompt}")
        print(f"Response: {response.strip()}")
        print("-" * 60)


if __name__ == "__main__":
    main()
