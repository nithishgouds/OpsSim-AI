from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from env import DevOpsEnv
from models import Action

app = FastAPI(title="OpsSim AI Environment API")
env_instance = DevOpsEnv()

# In app.py
class ResetRequest(BaseModel):
    task: str = "easy"



@app.get("/")
def root():
    return {"status": "OpsSim AI running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset_env(req: ResetRequest | None = None):
    try:
        task = req.task if req is not None else "easy"
        obs = env_instance.reset(task=task)
        return {"observation": obs.dict()}
    except Exception as e:
        print(f"INTERNAL ERROR: {str(e)}") 
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step_env(action: Action):
    try:
        obs, reward, done, info = env_instance.step(action)
        return {
            "observation": obs.dict(),
            "reward": reward.value,
            "done": done,
            "info": info,
            "last_action_error": env_instance.last_action_error
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def get_state():
    return {"state": env_instance.state()}

def main():
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
