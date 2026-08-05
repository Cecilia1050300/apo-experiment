import os

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("找不到 OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model="gpt-5.5",
        input="Reply with exactly: OpenAI API connection successful",
        reasoning={"effort": "none"},
    )

    print(response.output_text)


if __name__ == "__main__":
    main()
