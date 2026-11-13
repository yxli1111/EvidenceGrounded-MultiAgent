import os

from dotenv import load_dotenv

from src.core.llm_client import OpenRouterLLMClient


def main():
    load_dotenv()

    model_id = os.getenv("OPENROUTER_MODEL")

    if not model_id:
        print("Please configure OPENROUTER_MODEL in .env")
        return

    client = OpenRouterLLMClient()

    response = client.generate(
        stage="smoke_test",
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: Hello DSA4213"
            }
        ],
        model=model_id,
        temperature=0,
        max_tokens=64,
    )

    print("Model:", response.usage.model)
    print("Response:", response.text)
    print("Input tokens:", response.usage.input_tokens)
    print("Output tokens:", response.usage.output_tokens)
    print("Latency:", response.usage.latency_seconds)


if __name__ == "__main__":
    main()
