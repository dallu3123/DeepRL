import ptan
import torch
from torch import nn
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, n_actions: int):
        super(PolicyNet, self).__init__()
        self.actions = n_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = (x.size()[0], self.actions)
        res = torch.zeros(shape, dtype=torch.float32)
        res[:, 0] = 1.0
        res[:, 1] = 1.0
        return res

if __name__ == "__main__":
    net = PolicyNet(n_actions=5)
    net_out = net(torch.zeros(6, 10))
    print("policy output:")
    print(net_out)

    """
    policy output:
    tensor([[1., 1., 0., 0., 0.],
            [1., 1., 0., 0., 0.],
            [1., 1., 0., 0., 0.],
            [1., 1., 0., 0., 0.],
            [1., 1., 0., 0., 0.],
            [1., 1., 0., 0., 0.]])
    """
    selector = ptan.actions.ProbabilityActionSelector()
    agent = ptan.agent.PolicyAgent(model=net, action_selector=selector, apply_softmax=True)
    ag_out = agent(torch.zeros(6, 5))[0]
    print("ProbabilityActionSelector output:", ag_out)

    """
    ProbabilityActionSelector output: [0 1 2 0 0 0]
    """
