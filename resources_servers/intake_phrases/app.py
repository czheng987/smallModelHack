"""NeMo Gym resources server for intake phrasing.

Each server vote is one rollout. verify() scores that vote. compute_metrics()
turns the rollouts into pass@1, which is how many of the servers agreed on an
accurate split. Intake keeps the phrases only when that pass rate is at least 90%.

If the Gym checkout or its dependencies are missing, score_votes still returns
a result so intake can start.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, ClassVar, Optional
from unittest.mock import MagicMock

from phrase_check import PASS_RATE, accepted_split, consensus, drop_bare_phrases, passes, verify_split

# The Gym repo is a submodule. Put it on the path so intake can import it
# without a separate editable install.
_GYM_ROOT = Path(__file__).resolve().parents[2] / "Gym"
if (_GYM_ROOT / "nemo_gym").is_dir() and str(_GYM_ROOT) not in sys.path:
    sys.path.insert(0, str(_GYM_ROOT))

try:
    from pydantic import Field

    from nemo_gym.base_resources_server import (
        BaseResourcesServerConfig,
        BaseVerifyRequest,
        BaseVerifyResponse,
        ReverifyMode,
        SimpleResourcesServer,
    )
    from nemo_gym.openai_utils import NeMoGymResponse
    from nemo_gym.server_utils import ServerClient
except ModuleNotFoundError:
    Field = None
    SimpleResourcesServer = None


_gym_warning_printed = False


def score_votes(message: str, votes: list[Optional[tuple[list[str], list[str]]]]):
    """Score one student message. Uses NeMo Gym when that package can be imported."""
    global _gym_warning_printed
    if SimpleResourcesServer is None:
        if not _gym_warning_printed:
            print(
                "NeMo Gym is not installed. Run: git submodule update --init && "
                "pip install -r requirements.txt"
            )
            _gym_warning_printed = True
        return accepted_split(message, votes)
    return _score_with_gym(message, votes)


def run_gym_servers(message, schedule, resource, count):
    """Run count NeMo Gym servers on one phrase split. Does not call a model."""
    vote = (list(schedule or []), list(resource or []))
    return score_votes(message, [vote] * count)


if SimpleResourcesServer is not None:

    class IntakePhraseResourcesServerConfig(BaseResourcesServerConfig):
        REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    class IntakeVerifyRequest(BaseVerifyRequest):
        student_message: str = ""
        schedule: list[str] = Field(default_factory=list)
        resource: list[str] = Field(default_factory=list)
        answered: bool = True

    class IntakeVerifyResponse(BaseVerifyResponse):
        student_message: str = ""
        schedule: list[str] = Field(default_factory=list)
        resource: list[str] = Field(default_factory=list)
        answered: bool = True

    def _response_for(schedule, resource) -> NeMoGymResponse:
        """A Gym response whose text is the phrase split, so verify has a rollout to score."""
        text = '{"schedule": ' + str(list(schedule)) + ', "resource": ' + str(list(resource)) + "}"
        return NeMoGymResponse(
            id="intake-phrase",
            created_at=0,
            model="intake",
            object="response",
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            output=[
                {
                    "type": "message",
                    "id": "intake-phrase-message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
        )

    class IntakePhraseResourcesServer(SimpleResourcesServer):
        """Scores phrase votes and reports pass@1 for their agreement."""

        config: IntakePhraseResourcesServerConfig
        last_schedule: list[str] = Field(default_factory=list)
        last_resource: list[str] = Field(default_factory=list)

        async def verify(self, body: IntakeVerifyRequest) -> IntakeVerifyResponse:
            schedule, resource = drop_bare_phrases(body.schedule, body.resource)
            if not body.answered:
                reward = 0.0
                schedule, resource = [], []
            else:
                reward = 1.0 if passes(verify_split(body.student_message, schedule, resource)) else 0.0
            payload = body.model_dump()
            payload.update(reward=reward, schedule=schedule, resource=resource)
            return IntakeVerifyResponse(**payload)

        def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, float]:
            """pass@1 is the share of servers that agreed on an accurate split."""
            if not tasks:
                self.last_schedule, self.last_resource = [], []
                return {"pass@1": 0.0}
            rates = []
            chosen_schedule: list[str] = []
            chosen_resource: list[str] = []
            for rollouts in tasks:
                if not rollouts:
                    rates.append(0.0)
                    continue
                student_message = rollouts[0].get("student_message") or ""
                kept = []
                for item in rollouts:
                    if item.get("reward", 0) >= 1.0 and item.get("answered", False):
                        kept.append((item.get("schedule") or [], item.get("resource") or []))
                    else:
                        kept.append(None)
                schedule, resource, rate = consensus(kept)
                if not passes(verify_split(student_message, schedule, resource)):
                    schedule, resource, rate = [], [], 0.0
                rates.append(rate)
                chosen_schedule, chosen_resource = schedule, resource
            self.last_schedule = chosen_schedule
            self.last_resource = chosen_resource
            return {"pass@1": sum(rates) / len(rates)}

    def load_server() -> IntakePhraseResourcesServer:
        """Build the resources server in this process. Intake calls verify directly."""
        return IntakePhraseResourcesServer(
            config=IntakePhraseResourcesServerConfig(
                host="127.0.0.1",
                port=8002,
                entrypoint="app.py",
                name="intake_phrases",
            ),
            server_client=MagicMock(spec=ServerClient),
        )

    def _score_with_gym(message: str, votes):
        """One NeMo Gym server per vote. The model is not called here."""
        servers = [load_server() for _ in votes]

        async def _verify_all():
            async def _one(server, vote):
                if vote is None:
                    schedule, resource, answered = [], [], False
                else:
                    schedule, resource, answered = list(vote[0]), list(vote[1]), True
                body = IntakeVerifyRequest(
                    responses_create_params={"input": message},
                    response=_response_for(schedule, resource),
                    student_message=message,
                    schedule=schedule,
                    resource=resource,
                    answered=answered,
                )
                return await server.verify(body)

            return await asyncio.gather(*[_one(server, vote) for server, vote in zip(servers, votes)])

        responses = asyncio.run(_verify_all())
        metrics = servers[0].compute_metrics([[item.model_dump() for item in responses]])
        rate = float(metrics["pass@1"])
        if rate >= PASS_RATE:
            return True, list(servers[0].last_schedule), list(servers[0].last_resource), rate
        return False, [], [], rate

    if __name__ == "__main__":
        IntakePhraseResourcesServer.run_webserver()
