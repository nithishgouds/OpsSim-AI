"""
Functions Used


_is_correct_action() - Checks if picked action in step is true action

_llm_parse() - IMPORTANT. Need to add LLM inference here. For now, a basic simulation is added

_degrade_system() - If wrong action picked, penalise system

_generate_logs() - IMPORTANT. Generate example logs.

_generate_messages() - IMPORTANT. Generate messages from user.


"""

import random
from models import Observation, Action, Reward

class DevOpsEnv:
    def __init__(self, seed = 42):
        self.rng = random.Random(seed) # deterministic. For reproducibilty
        self.max_steps = 5 # Max steps in episode
        self.reset() # reset environment

    def reset(self) -> Observation:
        self.true_error = self.rng.choice([
            "db_timeout",
            "cpu_overload",
            "bad_deploy"
        ])  # select a random error from available errors. This is actual error

        self.step_count = 0
        self.resolved = False  # reset back to start
        self.priority_assigned = None

        self.logs = self._generate_logs(self.true_error)
        self.user_messages = self._generate_messages(self.true_error)
        hint, confidence = self._llm_parse(self.logs, self.user_messages)

        self.observation = Observation(
            logs = self.logs,
            user_messages = self.user_messages,

            error_type_hint = hint, 
            confidence = confidence,

            latency_level=self.rng.uniform(0.5, 0.9),  # high latency
            error_rate=self.rng.uniform(0.4, 0.8),     # moderate errors
            cpu_usage=self.rng.uniform(0.3, 0.9),      # varied CPU

            recent_deploy=(self.true_error == "bad_deploy"),  # only true for deploy issue

            user_impact=self.rng.uniform(0.4, 0.9),  # user complaints

            step_count=0
        )

        return self.observation
    
    def step(self, action:Action):
        reward = 0.0
        done = False
        
        self.step_count += 1

        # Based on Action, do this
        if action.action_type == "assign_high_priority":
            self.priority_assigned = "high"

        elif action.action_type == "assign_low_priority":
            self.priority_assigned = "low"

        elif action.action_type in ["restart_db", "scale_up_service", "rollback_deploy"]:
            
            #Correct action
            if self._is_correct_action(action.action_type):
                reward += 2
                self.resolved = True
                done = True
            #Incorrect Action
            else:
                reward -= 1
                self._degrade_system()

        #For now, investigate just increases confidence in LLM's result
        elif action.action_type == "investigate":
            reward -= 0.5
            hint, confidence = self._llm_parse(
                self.logs, self.user_messages, improve = True
            )
            self.observation.error_type_hint = hint
            self.observation.confidence = confidence
        
        reward -= 0.5 # Step penalty

        if self.step_count >= self.max_steps:
            done = True
            if not self.resolved:
                reward -= 3 # no solve = high negative reward
        
        self.observation.step_count = self.step_count

        return self.observation, Reward(value = reward), done, {}

    def state(self): # Keeps track of current state
        return {
            "true_error": self.true_error, # INTERNAL, true error
            "resolved": self.resolved, # is resolved??
            "step_count": self.step_count # How many steps
        }

    def _is_correct_action(self, action):
        mapping = {
            "db_timeout" : "restart_db",
            "cpu_overload": "scale_up_service",
            "bad_deploy": "rollback_deploy"
        }

        return mapping[self.true_error] == action
    
    def _llm_parse(self, logs, messages, improve = False):
        # !!!!!!! IMPORTANT !!!!!!!
        # Should add LLM Inference here. For now, just a basic simulation
        accuracy = 0.6
        if improve:
            accurace = 0.85
        if self.rng.random() < accuracy:
            hint = self.true_error
        else:
            hint = self.rng.choice([
                "db_timeout",
                "cpu_overload",
                "bad_deploy",
                "unknown"
            ])
        
        confidence = self.rng.uniform(0.3,0.8)
        if improve:
            confidence += 0.2

        return hint,min(confidence,1.0)

    def _degrade_system(self):
        self.observation.latency_level = min(1.0, self.observation.latency_level + 0.1)
        self.observation.user_impact = min(1.0, self.observation.user_impact + 0.1)
    
    def _generate_logs(self, error):
        if error == "db_timeout":
            return "ERROR: connection timeout at db-service"

        if error == "cpu_overload":
            return "WARNING: CPU usage exceeded 95 percent on service"

        if error == "bad_deploy":
            return "ERROR: service crash after deployment"

        return "No logs"

    def _generate_messages(self, error):
        if error == "db_timeout":
            return "Users report payment failures and timeouts"

        if error == "cpu_overload":
            return "App is very slow"

        if error == "bad_deploy":
            return "App broke after update"

        return "No complaints"
