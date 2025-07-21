import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
import typing as tt

from .module.cem import CEM, iterate_batches, Episode  # noqa: F403

HIDDEN_SIZE = 128
BATCH_SIZE = 100
PERCENTILE = 30
GAMMA = 0.9

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

def filter_batch(batch: tt.List[Episode], percentile: float) -> \
        tt.Tuple[tt.List[Episode], tt.List[np.ndarray], tt.List[int], float]:
    reward_fun = lambda s: s.reward * (GAMMA ** len(s.steps))
    disc_rewards = list(map(reward_fun, batch))
    reward_bound = np.percentile(disc_rewards, percentile)

    train_obs: tt.List[np.ndarray] = []
    train_act: tt.List[int] = []
    elite_batch: tt.List[Episode] = []

    for example, discounted_reward in zip(batch, disc_rewards):
        if discounted_reward > reward_bound:
            train_obs.extend(map(lambda step: step.observation, example.steps))
            train_act.extend(map(lambda step: step.action, example.steps))
            elite_batch.append(example)

    return elite_batch, train_obs, train_act, reward_bound

if __name__ == "__main__":
    random.seed(12345)

    env = DiscreteOneHotWrapper(gym.make("FrozenLake-v1", render_mode="rgb_array"))
    obs_size = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    CEM_net = CEM(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=CEM_net.parameters(), lr=1e-3)

    writer = SummaryWriter(comment="-frozenlake-tweaked")

    full_batch = []
    for iter_no, batch in enumerate(iterate_batches(env, CEM_net, BATCH_SIZE)):
        reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))
        full_batch, obs, act, reward_bound = filter_batch(full_batch + batch, PERCENTILE)

        if not full_batch:
            continue

        obs_v = torch.FloatTensor(np.vstack(obs))
        act_v = torch.LongTensor(act)

        full_batch = full_batch[-500:]
        optimizer.zero_grad()
        action_scores_v = CEM_net(obs_v)

        loss_v = objective(action_scores_v, act_v)
        loss_v.backward()
        optimizer.step()

        full_batch.extend(batch)

        print("%d: loss=%.3f, reward_mean=%.3f, rw_bound=%.3f, batch=%d" % (
            iter_no, loss_v.item(), reward_mean, reward_bound, len(full_batch)
        ))

        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_mean", reward_mean, iter_no)
        writer.add_scalar("reward_bound", reward_bound, iter_no)
        writer.add_scalar("batch_size", len(full_batch), iter_no)

        if reward_mean > 0.80:
            print("Solved!")
            break

    writer.close()

