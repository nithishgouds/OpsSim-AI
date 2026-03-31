from pydantic import BaseModel
from typing import Literal

class Observation(BaseModel):
    logs: str # logs 
    user_messages: str # messages given by user
    error_type_hint: Literal["db_timeout", "cpu_overload", "bad_deploy", "unknown"] # LLM's guess
    confidence: float # How confident LLM is on hint (0-1)
    latency_level: float # How much latency system is experiencing
    error_rate: float # failure rate (0-1)
    cpu_usage: float # how intensive the task to run is (0-1)
    recent_deploy: bool # Has there been a recent deploy?
    user_impact: float # Impact on Users
    step_count: int # current timestep

class Action(BaseModel):
    action_type : Literal[
        "assign_low_priority", # set priority to low
        "assign_high_priority", # set priority to high

        "restart_db", # to fix timeout
        "scale_up_service", # fix cpu_overload
        "rollback_deploy", # fix bad_deploy

        "investigate", # inquire more, but at a high cost
        "do_nothing" # no action
    ]

class Reward(BaseModel):
    value: float # final reward value