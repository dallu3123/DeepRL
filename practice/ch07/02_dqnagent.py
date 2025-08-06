import ptan
import torch
from torch import nn
import numpy as np

class DQNNet(nn.Module):
    def __init__(self, n_actions: int):
        super(DQNNet, self).__init__()
        self.actions = n_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.eye(x.size()[0], self.actions)

if __name__ == "__main__":
    net = DQNNet(10)
    net_out = net(torch.zeros(2, 10))

    print("dqn output:")
    print(net_out)
    """
    dqn output:
    tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]])
    """

    selector = ptan.actions.ArgmaxActionSelector()
    agent = ptan.agent.DQNAgent(model=net, action_selector=selector)
    ag_out = agent(np.zeros(shape=(2, 5)))
    print("ArgmaxActionSelector output:",  ag_out)

    """
    ArgmaxActionSelector output: (array([0, 1]), [None, None])
    """

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1.0)
    agent = ptan.agent.DQNAgent(model=net, action_selector=selector)
    ag_out = agent(torch.zeros(10, 5))[0]
    print("EpsilonGreedyActionSelector output:", ag_out)

    """
    EpsilonGreedyActionSelector output: [0 3 3 0 0 4 5 3 3 6]
    """

    selector.epsilon = 0.5
    ag_out = agent(torch.zeros(10, 5))[0]
    print("EpsilonGreedyActionSelector output:", ag_out)

    """
    EpsilonGreedyActionSelector output: [0 1 5 4 7 2 6 1 8 3]
    """