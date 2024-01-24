import gymnasium as gym
import matplotlib.pyplot as plt
from IPython import display


class RandomAgent:
    def __init__(self, env):
        self.env = env

    def get_action(self, state):
        return self.env.action_space.sample()
    
env = gym.make("Taxi-v3", render_mode="rgb_array")
agent = RandomAgent(env)
env.reset()
env.s = 123
reward = 0
penality = 0
frames = []
done = False
epoches = 0

while not done:
    state = env.s
    action = agent.get_action(state)
    take_action = env.step(action)
    state = take_action[0]
    reward = take_action[1]
    done = take_action[2]

    if(reward == -10):
        penality += 1

    frames.append({
        'state': state,
        'action': action,
        'reward': reward 
    })
    epoches += 1
print(f"Steps: {epoches}")
print(f"reward: {reward}")
print(f"penality: {penality}")

renderIm = env.render()
plt.figure("Randome Agent")
plt.title(label="Random Agent")
plt.imshow(renderIm)
plt.show()

display.display(plt.gcf())
display.clear_output(True)
