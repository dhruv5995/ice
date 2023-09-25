from typing import Optional

from genai.credentials import Credentials
from genai.model import Model
from genai.schemas import GenerateParams
from structlog.stdlib import get_logger

from ice.agents.base import Agent
from ice.agents.base import Stop
from ice.settings import settings

log = get_logger()


class WatsonXAgent(Agent):
    """An agent that uses IBM's WatsonX GenAI to generate answers and predictions."""

    def __init__(self, model: str, params: Optional[dict] = None) -> None:
        self.model = model
        self.params = params or {}

    async def complete(
        self,
        *,
        prompt: str,
        stop: Stop = None,
        verbose: bool = False,
        default: str = "",
        max_tokens: int = 256
    ) -> str:
        """Generate an answer to a question given some context."""
        if verbose:
            print(prompt)

        completion = await self._complete(prompt=prompt)

        return completion

    async def _complete(self, prompt: str):
        credentials = Credentials(settings.GENAI_KEY, api_endpoint=settings.GENAI_API)

        model = Model(
            model=self.model,
            credentials=credentials,
            params=GenerateParams(**self.params),
        )

        completion = model.generate(prompts=[prompt])

        return completion[0].generated_text
