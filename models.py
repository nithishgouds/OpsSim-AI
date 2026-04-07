from pydantic import BaseModel
from typing import Literal, Optional

class Reward(BaseModel):
    value: float

class Observation(BaseModel):
    task_type: Literal["easy", "medium", "hard"]
    user_message: Optional[str] = None
    config: Optional[dict] = None
    available_actions: Optional[list[str]] = None
    user_messages: Optional[list[str]] = None
    system_metrics: Optional[dict] = None
    system_state: Optional[dict] = None
    alerts: Optional[list[str]] = None
    playbook_text: Optional[str] = None
    logs: Optional[str] = None
    step_count: int

class Action(BaseModel):
    action_type: str
    target: Optional[str] = None
