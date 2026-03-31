"""
LLMParser (Skeleton)

Structure:
- parse() → routes to task-specific handlers
- _parse_easy / _parse_medium / _parse_hard → full pipelines
- Each pipeline:
    1. build prompt
    2. call LLM
    3. parse response
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
        self.cache = {}

    # ================= MAIN ENTRY =================
    def parse(self, observation):
        key = self._build_cache_key(observation)

        if key in self.cache:
            return self.cache[key]

        if observation.task_type == "easy":
            result = self._parse_easy(observation)

        elif observation.task_type == "medium":
            result = self._parse_medium(observation)

        else:
            result = self._parse_hard(observation)

        self.cache[key] = result
        return result

    # NOTE : Each response should return (action, confidence, target : Optional[str])
    # ================= EASY =================
    def _parse_easy(self, obs):
        prompt = self._build_easy_prompt(obs)
        response = self._call_llm(prompt)
        return self._parse_easy_response(response, obs)

    def _build_easy_prompt(self, obs):
        # TODO: implement prompt for EASY task
        return ""

    def _parse_easy_response(self, text, obs):
        # TODO: parse LLM response for EASY task
        return "do_nothing", 0.0, None

    # ================= MEDIUM =================
    def _parse_medium(self, obs):
        prompt = self._build_medium_prompt(obs)
        response = self._call_llm(prompt)
        return self._parse_medium_response(response, obs)

    def _build_medium_prompt(self, obs):
    # Build prompt focusing on complaint-based reasoning
        return f"""
DO NOT output anything except valid JSON.

You are diagnosing a subtle production issue.

IMPORTANT:
- Logs and system metrics appear NORMAL
- The issue must be inferred from USER COMPLAINTS

User complaints:
{obs.user_messages}

Logs:
{obs.logs}

System metrics:
{obs.system_metrics}

Available actions:
{obs.available_actions}

Goal:
Choose the NEXT BEST ACTION to move toward resolving the issue.

Guidelines:
- If pattern unclear → "analyze_complaints"
- If pattern identified → "identify_service"
- If root cause clear → "fix_weekend_bug"

Return ONLY JSON:
{{
    "action": "<one of available_actions>",
    "confidence": 0.0-1.0
}}
"""

    def _parse_medium_response(self, text, obs):
        try:
            data = json.loads(text)  # parse JSON

            action = data.get("action", "do_nothing")
            confidence = float(data.get("confidence", 0.3))

            # validate action
            if action not in obs.available_actions:
                action = "do_nothing"

            # clamp confidence
            confidence = max(0.0, min(1.0, confidence))

            return action, confidence, None  # no target in medium

        except Exception:
            # fallback for bad JSON / malformed output
            return "do_nothing", 0.3, None

    # ================= HARD =================
    def _parse_hard(self, obs):
        prompt = self._build_hard_prompt(obs)
        response = self._call_llm(prompt)
        return self._parse_hard_response(response, obs)

    def _build_hard_prompt(self, obs):
        # TODO: implement prompt for HARD task
        return ""

    def _parse_hard_response(self, text, obs):
        # TODO: parse LLM response for HARD task
        return "do_nothing", 0.0, None

    # ================= LLM CALL =================
    def _call_llm(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,        # deterministic
            max_tokens=120
        )

        return response.choices[0].message.content or ""

    # ================= CACHE KEY =================
    def _build_cache_key(self, obs):
        return json.dumps({
            "task": obs.task_type,
            "messages": getattr(obs, "user_messages", None),
            "logs": obs.logs,
            "metrics": getattr(obs, "system_metrics", None)
        }, sort_keys=True)