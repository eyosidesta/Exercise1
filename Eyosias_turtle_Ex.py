import gym
import turtle

class RandomAgent:
    def __init__(self, env):
        self.env = env

    def get_action(self, state):
        return self.env.action_space.sample()

def draw_state(rgb_array):
    turtle.clear()
    turtle.hideturtle()

    # Assuming the RGB array has the shape (height, width, 3)
    height, width, _ = rgb_array.shape

    # Calculate the size of each cell based on the dimensions of the environment
    cell_size = min(500 // height, 500 // width)

    for i in range(height):
        for j in range(width):
            color = rgb_array[i, j, :]
            turtle.penup()
            turtle.goto((j - width / 2) * cell_size, (height / 2 - i) * cell_size)
            turtle.pendown()
            turtle.color(color / 255.0)  # Normalize color values to range [0, 1]
            turtle.begin_fill()
            for _ in range(4):
                turtle.forward(30)
                turtle.right(90)
            turtle.end_fill()

# Create the Taxi environment with "rgb_array" rendering mode
env = gym.make("Taxi-v3", render_mode="rgb_array")
agent = RandomAgent(env)
env.reset()
env.s = 123
reward = 0
penalty = 0
frames = []
done = False
epochs = 0

# Initialize Turtle for rendering
turtle.speed(1)
turtle.title("Taxi Environment")

while not done:
    state = env.s
    action = agent.get_action(state)
    state_resu = env.step(action)
    state = state_resu[0]
    reward = state_resu[1]
    done = state_resu[2]

    if reward == -10:
        penalty += 1

    frames.append({
        'state': state,
        'action': action,
        'reward': reward
    })
    epochs += 1

    # Draw the current state using Turtle
    draw_state(env.render())

turtle.bye()

print("the epochs is: ", epochs)
print("the penalty is: ", penalty)
print("the reward is: ", reward)
