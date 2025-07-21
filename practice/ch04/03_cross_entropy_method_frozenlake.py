import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

from .module.cem import CEM, filter_batch, iterate_batches  # noqa: F403

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70

class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        shape = (env.observation_space.n, )
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=shape, dtype=np.float32)

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", render_mode="rgb_array")
    env = DiscreteOneHotWrapper(env)

    obs_size = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    CEM_net = CEM(obs_size, HIDDEN_SIZE, n_actions)

    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=CEM_net.parameters(), lr=1e-2)

    writer = SummaryWriter(comment="-frozenlake-naive")

    for iter_no, batch in enumerate(iterate_batches(env, CEM_net, BATCH_SIZE)):
        train_obs_v, train_act_v, reward_bound, reward_mean = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = CEM_net(train_obs_v)
        loss_v = objective(action_scores_v, train_act_v)
        loss_v.backward()
        optimizer.step()

        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            iter_no, loss_v.item(), reward_mean, reward_bound
        ))

        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_mean", reward_mean, iter_no)
        writer.add_scalar("reward_bound", reward_bound, iter_no)

        if reward_mean > 0.80:
            print("Solved!")
            break

    writer.close()

'''
학습결과 학습이 되지 않아서 종료됨
'''