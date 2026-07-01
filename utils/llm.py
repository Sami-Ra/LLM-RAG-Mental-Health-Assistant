from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
    APIError,
)
from config import OPENAI_API_KEY, LLM_MODEL, TEMPERATURE, MAX_TOKENS


class LLMService:
    """Handles communication with the OpenAI Chat Completions API."""

    def __init__(self) -> None:
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def generate_response(self, messages: list[dict]) -> str:
        """
        Send a messages list to OpenAI and return the assistant's reply.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-format messages (system + user turns).

        Returns
        -------
        str
            The assistant's response content.

        Raises
        ------
        ValueError
            For authentication failures (invalid API key).
        RuntimeError
            For connection, timeout, rate-limit, or other API errors.
        """
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=messages,
            )
            return response.choices[0].message.content

        except AuthenticationError:
            raise ValueError(
                "Invalid OpenAI API key. Check the OPENAI_API_KEY value in your .env file."
            )
        except RateLimitError:
            raise RuntimeError(
                "OpenAI rate limit exceeded. Please wait a moment and try again."
            )
        except APITimeoutError:
            raise RuntimeError(
                "The request to OpenAI timed out. Please try again."
            )
        except APIConnectionError:
            raise RuntimeError(
                "Could not connect to OpenAI. Check your internet connection."
            )
        except APIError as e:
            raise RuntimeError(f"OpenAI API error: {e}")
