import gymnasium as gym
import ptan
import typing as tt
import torch.nn as nn

class ToyEnv(gym.Env):
    """
    observation 0..4 이고 actions 0..2

    """
    def __init__(self):
        super(ToyEnv, self).__init__()
        self.observation_space = gym.spaces.Discrete(5)
        self.action_space = gym.spaces.Discrete(3)
        self.step_index = 0

    def reset(self):
        self.step_index = 0
        return self.step_index, {}

    def step(self, action: int):
        is_done = self.step_index == 10
        if is_done:
            return self.step_index % self.observation_space.n, 0.0, is_done, False, {}
        self.step_index += 1
        return self.step_index % self.observation_space.n, float(action), self.step_index == 10, False, {}

class DQNNet(nn.Module):
    def __init__(self):
        super(DQNNet, self).__init__()
        self.ff = nn.Linear(5, 3)

    def forward(self, x):
        return self.ff(x)

class DullAgent(ptan.agent.BaseAgent):
    def __init__(self, n_actions: int):
        self.action = n_actions

    def __call__(self, observation: tt.List[int], state: tt.Optional[list] = None) -> tt.Tuple[tt.List[int], tt.Optional[list]]:
        return [self.action for _ in observation], state