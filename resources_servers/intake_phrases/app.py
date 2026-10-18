"""NeMo Gym resources server for intake phrasing.

Each server vote is one rollout. verify() scores that vote. compute_metrics()
turns the rollouts into pass@1, which is how many of the servers agreed on an
accurate split. Intake keeps the phrases only when that pass rate is at least 90%.
"""

import asyncio
from typing import Any, ClassVar, Optional
from unittest.mock import MagicMock

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
from phrase_check import PASS_RATE, consensus, drop_bare_phrases, passes, verify_split


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
            message = rollouts[0].get("student_message") or ""
            votes = []
            for item in rollouts:
                if item.get("reward", 0) >= 1.0 and item.get("answered", False):
                    votes.append((item.get("schedule") or [], item.get("resource") or []))
                else:
                    votes.append(None)
            schedule, resource, rate = consensus(votes)
            if not passes(verify_split(message, schedule, resource)):
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


def score_votes(message: str, votes: list[Optional[tuple[list[str], list[str]]]]):
    """Score one student message with the NeMo Gym verifier. Returns ok, phrases, pass@1."""
    server = load_server()

    async def _verify_all():
        responses = []
        for vote in votes:
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
            responses.append(await server.verify(body))
        return responses

    responses = asyncio.run(_verify_all())
    metrics = server.compute_metrics([[item.model_dump() for item in responses]])
    rate = float(metrics["pass@1"])
    if rate >= PASS_RATE:
        return True, list(server.last_schedule), list(server.last_resource), rate
    return False, [], [], rate


if __name__ == "__main__":
    IntakePhraseResourcesServer.run_webserver()
