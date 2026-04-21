from typing import Literal, Optional, Dict, Any
from openenv.core import (
    Action as OpenEnvAction,
    Observation as OpenEnvObservation,
    State as OpenEnvState,
)
from pydantic import BaseModel, Field


class Reward(BaseModel):
    value: float


class OpsSIMObservation(OpenEnvObservation):
    """Observation returned by the OpsSim-AI environment."""
    task_type: Literal["easy", "medium", "hard", "cascade"] = "easy"
    user_message: Optional[str] = None
    config: Optional[dict] = None
    available_actions: Optional[list[str]] = None
    user_messages: Optional[list[str]] = None
    system_metrics: Optional[dict] = None
    system_state: Optional[dict] = None
    alerts: Optional[list[str]] = None
    playbook_text: Optional[str] = None
    logs: Optional[str] = None
    step_count: int = 0


class OpsSIMAction(OpenEnvAction):
    """Action taken by the agent in the OpsSim-AI environment."""
    action_type: str = Field(..., description="The action to execute")
    target: Optional[str] = Field(default=None, description="Optional target for the action")


class OpsSIMState(OpenEnvState):
    """Internal state of the OpsSim-AI environment."""
    task_type: str = "easy"
    state_data: Dict[str, Any] = Field(default_factory=dict)


# Backward-compatible aliases so existing code keeps working
Observation = OpsSIMObservation
Action = OpsSIMAction
