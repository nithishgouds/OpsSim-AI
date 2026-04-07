from pydantic import BaseModel
from typing import Literal, Optional

class Observation(BaseModel):
    task_type: Literal["easy", "medium", "hard"]

    # EASY
    user_message: Optional[str] = None
    config: Optional[dict] = None
    available_actions: Optional[list[str]] = None

    # MEDIUM
    user_messages: Optional[list[str]] = None
    system_metrics: Optional[dict] = None

    # HARD
    system_state: Optional[dict] = None
    alerts: Optional[list[str]] = None
    playbook_text: Optional[str] = None

    # COMMON
    logs: Optional[str] = None
    step_count: int

    
class Action(BaseModel):
    action_type: str
    target: Optional[str] = None  # needed for HARD task
