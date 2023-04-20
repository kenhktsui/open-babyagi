from typing import List
import json
import httpx
from httpx import TimeoutException
import backoff


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        TimeoutException
    ),
    max_tries=3,
    factor=1.5,
    max_time=300
)
def huggingface_chat_completion_create_retrying(messages: List[dict],
                                                host: str,
                                                port: int,
                                                temperature: float = 0.5,
                                                max_tokens: int = 100,
                                                **kwargs):
    """
    Helper function for creating a completion.
    `args` and `kwargs` match what is accepted by `openai.Completion.create`.
    """
    data = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **kwargs
    }
    result = httpx.post(f'http://{host}:{port}/chat/completions',
                        data=json.dumps(data),
                        headers={'Content-type': 'application/json'},
                        timeout=60.0)
    result.raise_for_status()
    result = json.loads(result.text)
    return result
