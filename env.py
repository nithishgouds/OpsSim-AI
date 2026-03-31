"""
DevOpsEnv

Implements a multi-task OpenEnv-compatible environment with 3 tasks:
- EASY: Fix configuration from natural language
- MEDIUM: Detect hidden bug from complaints
- HARD: Policy-based system stabilization

Key Methods:
- reset() → initializes environment per task
- step(action) → applies action, uses LLM hint, computes reward
- state() → returns internal state
"""

import random
from models import Observation, Action, Reward
from llm_parser import LLMParser


class DevOpsEnv:
    def __init__(self, seed=42, max_steps = 5, task_type = "easy"):
        self.rng = random.Random(seed)  # deterministic randomness
        self.llm = LLMParser()          # LLM interface
        self.max_steps = max_steps      # max steps per episode
        self.task_type = task_type      # pick task
        self.reset()                    # initialize env

    # ================= RESET =================
    def reset(self) -> Observation:
        self.step_count = 0  # reset step counter

        if self.task_type == "easy":
            return self._reset_easy()

        elif self.task_type == "medium":
            return self._reset_medium()

        else:
            return self._reset_hard()

    # -------- EASY --------
    def _reset_easy(self):
        return None

    # -------- MEDIUM --------
    def _reset_medium(self):
        return None

    # -------- HARD --------
    def _reset_hard(self):
        return None

    # ================= STEP =================
    def step(self, action: Action):
        self.step_count += 1  # increment step count

        

        if self.task_type == "easy":
            # LLM is called HERE
            llm_action, confidence, llm_target = self.llm.parse(self.observation)
            obs, reward, done, info = self._step_easy(action, llm_action, confidence)

        elif self.task_type == "medium":
            # LLM is called HERE
            llm_action, confidence, llm_target = self.llm.parse(self.observation)
            obs, reward, done, info = self._step_medium(action, llm_action, confidence)

        else:
            # LLM is called HERE
            llm_action, confidence, llm_target = self.llm.parse(self.observation)
            obs, reward, done, info = self._step_hard(action, llm_action, confidence, llm_target)

        obs.step_count = self.step_count  # update step count in observation

        return obs, reward, done, info

    # ================= EASY STEP =================
    def _step_easy(self, action, llm_action, confidence):
        return None

    # ================= MEDIUM STEP =================
    def _step_medium(self, action, llm_action, confidence):
        return None

    # ================= HARD STEP =================
    def _step_hard(self, action, llm_action, confidence, llm_target):
        return None

    # ================= STATE =================
    def state(self):
        return {
            "task_type": self.task_type,
            "state": self.state_data,
            "step_count": self.step_count
        }