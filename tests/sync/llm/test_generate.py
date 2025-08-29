
from llemon.sync import LLM


def test_generate(llm: LLM):
    response = llm.generate("What's 2 + 2? Answer with a single digit and no punctuation.")
    assert response.text == "4"
