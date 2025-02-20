import gymnasium as gym

class MuJoCoEnvWrapper:
    def __init__(self, env_name, seed=42):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.env.seed(seed)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()
