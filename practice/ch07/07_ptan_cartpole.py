import gymnasium as gym
from ptan.experience import ExperienceSourceFirstLast, ExperienceFirstLast

import ptan
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import typing as tt


HIDDEN_SIZE = 128
BATCH_SIZE = 16
TGT_NET_SYNC = 10
GAMMA = 0.9
REPLAY_SIZE = 1000
LR = 1e-3
EPS_DECAY = 0.99

class Net(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, n_actions: int):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

@torch.no_grad()
def unpack_batch(batch: tt.List[ExperienceFirstLast], net: nn.Module, gamma: float):
    """
    """
    states = []
    actions = []
    rewards = []
    dones = []
    last_states = []
    # batch에서 각 경험을 추출하여 리스트에 추가
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)

    states_v = torch.as_tensor(np.stack(states))
    actions_v = torch.tensor(actions)
    rewards_v = torch.tensor(rewards)
    last_states_v = torch.as_tensor(np.stack(last_states))
    last_state_q_v = net(last_states_v)
    best_last_q_v = torch.max(last_state_q_v, dim=1)[0]
    best_last_q_v[dones] = 0.0
    return states_v, actions_v, best_last_q_v * gamma + rewards_v

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    tgt_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.ArgmaxActionSelector()
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1.0, selector=selector)
    agent = ptan.agent.DQNAgent(net, selector)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)

    optimizer = optim.Adam(net.parameters(), LR)

    step = 0
    episode = 0.0
    solved = False

    while True:
        step += 1
        buffer.populate(1)

        for reward, steps in exp_source.pop_rewards_steps():
            episode += 1
            print(f"{step}: episode {episode}, reward {reward:.3f}, eps {selector.epsilon:.3f}")
            solved = reward > 150
        if solved:
            print("Solved!")
            break
        if len(buffer) < REPLAY_SIZE:
            continue

        batch = buffer.sample(BATCH_SIZE)
        states_v, actions_v, taget_q_v = unpack_batch(batch, tgt_net.target_model, GAMMA)
        optimizer.zero_grad()
        q_v = net(states_v)
        q_v = q_v.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        loss_v = F.mse_loss(q_v, taget_q_v)
        loss_v.backward()
        optimizer.step()
        selector.epsilon = max(0.01, EPS_DECAY * selector.epsilon)
        if step % TGT_NET_SYNC == 0:
            tgt_net.sync()

    print("Training done")
    torch.save(net.state_dict(), "cartpole_dqn.pt")
    print("Model saved")

"""
25: episode 1.0, reward 24.000, eps 1.000
42: episode 2.0, reward 17.000, eps 1.000
63: episode 3.0, reward 21.000, eps 1.000
77: episode 4.0, reward 14.000, eps 1.000
88: episode 5.0, reward 11.000, eps 1.000
100: episode 6.0, reward 12.000, eps 1.000
115: episode 7.0, reward 15.000, eps 1.000
139: episode 8.0, reward 24.000, eps 1.000
169: episode 9.0, reward 30.000, eps 1.000
191: episode 10.0, reward 22.000, eps 1.000
206: episode 11.0, reward 15.000, eps 1.000
290: episode 12.0, reward 84.000, eps 1.000
312: episode 13.0, reward 22.000, eps 1.000
335: episode 14.0, reward 23.000, eps 1.000
358: episode 15.0, reward 23.000, eps 1.000
371: episode 16.0, reward 13.000, eps 1.000
389: episode 17.0, reward 18.000, eps 1.000
431: episode 18.0, reward 42.000, eps 1.000
484: episode 19.0, reward 53.000, eps 1.000
507: episode 20.0, reward 23.000, eps 1.000
520: episode 21.0, reward 13.000, eps 1.000
531: episode 22.0, reward 11.000, eps 1.000
548: episode 23.0, reward 17.000, eps 1.000
568: episode 24.0, reward 20.000, eps 1.000
587: episode 25.0, reward 19.000, eps 1.000
604: episode 26.0, reward 17.000, eps 1.000
635: episode 27.0, reward 31.000, eps 1.000
657: episode 28.0, reward 22.000, eps 1.000
699: episode 29.0, reward 42.000, eps 1.000
732: episode 30.0, reward 33.000, eps 1.000
759: episode 31.0, reward 27.000, eps 1.000
799: episode 32.0, reward 40.000, eps 1.000
845: episode 33.0, reward 46.000, eps 1.000
860: episode 34.0, reward 15.000, eps 1.000
886: episode 35.0, reward 26.000, eps 1.000
897: episode 36.0, reward 11.000, eps 1.000
911: episode 37.0, reward 14.000, eps 1.000
927: episode 38.0, reward 16.000, eps 1.000
987: episode 39.0, reward 60.000, eps 1.000
1008: episode 40.0, reward 21.000, eps 0.923
1032: episode 41.0, reward 24.000, eps 0.725
1057: episode 42.0, reward 25.000, eps 0.564
1066: episode 43.0, reward 9.000, eps 0.515
1076: episode 44.0, reward 10.000, eps 0.466
1092: episode 45.0, reward 16.000, eps 0.397
1106: episode 46.0, reward 14.000, eps 0.345
1116: episode 47.0, reward 10.000, eps 0.312
1126: episode 48.0, reward 10.000, eps 0.282
1136: episode 49.0, reward 10.000, eps 0.255
1148: episode 50.0, reward 12.000, eps 0.226
1163: episode 51.0, reward 15.000, eps 0.194
1173: episode 52.0, reward 10.000, eps 0.176
1186: episode 53.0, reward 13.000, eps 0.154
1195: episode 54.0, reward 9.000, eps 0.141
1205: episode 55.0, reward 10.000, eps 0.127
1214: episode 56.0, reward 9.000, eps 0.116
1224: episode 57.0, reward 10.000, eps 0.105
1234: episode 58.0, reward 10.000, eps 0.095
1244: episode 59.0, reward 10.000, eps 0.086
1253: episode 60.0, reward 9.000, eps 0.079
1263: episode 61.0, reward 10.000, eps 0.071
1273: episode 62.0, reward 10.000, eps 0.064
1282: episode 63.0, reward 9.000, eps 0.059
1293: episode 64.0, reward 11.000, eps 0.053
1301: episode 65.0, reward 8.000, eps 0.049
1310: episode 66.0, reward 9.000, eps 0.044
1320: episode 67.0, reward 10.000, eps 0.040
1331: episode 68.0, reward 11.000, eps 0.036
1341: episode 69.0, reward 10.000, eps 0.032
1350: episode 70.0, reward 9.000, eps 0.030
1361: episode 71.0, reward 11.000, eps 0.027
1371: episode 72.0, reward 10.000, eps 0.024
1382: episode 73.0, reward 11.000, eps 0.022
1392: episode 74.0, reward 10.000, eps 0.019
1404: episode 75.0, reward 12.000, eps 0.017
1418: episode 76.0, reward 14.000, eps 0.015
1442: episode 77.0, reward 24.000, eps 0.012
1451: episode 78.0, reward 9.000, eps 0.011
1461: episode 79.0, reward 10.000, eps 0.010
1471: episode 80.0, reward 10.000, eps 0.010
1479: episode 81.0, reward 8.000, eps 0.010
1551: episode 82.0, reward 72.000, eps 0.010
1581: episode 83.0, reward 30.000, eps 0.010
1661: episode 84.0, reward 80.000, eps 0.010
1679: episode 85.0, reward 18.000, eps 0.010
1701: episode 86.0, reward 22.000, eps 0.010
1765: episode 87.0, reward 64.000, eps 0.010
1782: episode 88.0, reward 17.000, eps 0.010
1933: episode 89.0, reward 151.000, eps 0.010
Solved!
Training done
Model saved
"""