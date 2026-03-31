from env import DevOpsEnv
from models import Action

env = DevOpsEnv()

obs = env.reset()
print("Initial observation:")
print(obs)

for step in range(5):
    action = Action(action_type="investigate")  # try fixed action
    obs, reward, done, _ = env.step(action)

    print(f"\nStep {step+1}")
    print("Observation:", obs)
    print("Reward:", reward)
    print("Done:", done)

    if done:
        break