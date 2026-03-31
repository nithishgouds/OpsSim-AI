"""
LLMParser

Handles LLM-based reasoning for all tasks:
- EASY: map natural language → config fix action
- MEDIUM: infer hidden bug from complaints
- HARD: policy-based decision making

Key Methods:
- parse(observation) → returns (action, confidence, target)
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMParser:
    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv("API_BASE_URL"),
            api_key=os.getenv("HF_TOKEN") or os.getenv("API_KEY")
        )

        self.model = os.getenv("MODEL_NAME")
        self.cache = {}  # deterministic cache

    # ================= MAIN ENTRY =================
    def parse(self, observation):
        key = self._build_cache_key(observation)  # unique key per observation

        if key in self.cache:
            return self.cache[key]

        # route based on task
        if observation.task_type == "easy":
            prompt = self._build_easy_prompt(observation)

        elif observation.task_type == "medium":
            prompt = self._build_medium_prompt(observation)

        else:
            prompt = self._build_hard_prompt(observation)

        response_text = self.call_llm(prompt)
        action, confidence, target = self._parse_response(response_text)

        self.cache[key] = (action, confidence, target)
        return action, confidence, target

    # ================= PROMPTS =================

    def _build_easy_prompt(self, obs):
        # Add your prompt for easy task here
        return ""

    def _build_medium_prompt(self, obs):
        # Add your prompt for med task here
        return ""

    def _build_hard_prompt(self, obs):
        # Add your prompt for hard task here
        return ""

    # ================= LLM CALL =================
    def call_llm(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=120
        )

        return response.choices[0].message.content or ""

    # ================= PARSING =================
    def _parse_response(self, text: str):
        try:
            data = json.loads(text)

            action = data.get("action", "do_nothing")
            confidence = float(data.get("confidence", 0.3))
            target = data.get("target", None)

            return action, min(confidence, 1.0), target

        except:
            # fallback (important for robustness)
            return "do_nothing", 0.3, None

    # ================= CACHE KEY =================
    def _build_cache_key(self, obs):
        """
        Build deterministic cache key based on observation
        """
        return json.dumps({
            "task": obs.task_type,
            "logs": obs.logs,
            "user_message": getattr(obs, "user_message", None),
            "user_messages": getattr(obs, "user_messages", None),
            "system_state": getattr(obs, "system_state", None),
            "alerts": getattr(obs, "alerts", None)
        }, sort_keys=True)