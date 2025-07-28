import gymnasium as gym
from lib import dqn_model
from lib import wrappers

from dataclasses import dataclass
import argparse
import time
import numpy as np
import collections
import typing as tt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard.writer import SummaryWriter


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19


# 하이퍼파라미터
GAMMA = 0.99 # 할인 계수
BATCH_SIZE = 32 # 미니배치 크기
REPLAY_SIZE = 10000 # 리플레이 버퍼 크기
LEARNING_RATE = 1e-4 # 학습률
SYNC_TARGET_FRAMES = 1000 # 목표 네트워크 업데이트 주기
REPLAY_START_SIZE = 10000 # 리플레이 시작 크기

EPSILON_DECAY_LAST_FRAME = 150000 # 탐험 감소 최종 프레임
EPSILON_START = 1.0 # 탐험 감소 시작 프레임
EPSILON_FINAL = 0.01 # 탐험 감소 최종 프레임

# 상태, 액션, 배치 텐서 타입
State = np.ndarray
Action = int
BatchTensors = tt.Tuple[
    torch.ByteTensor,           # current state
    torch.LongTensor,           # actions
    torch.Tensor,               # rewards
    torch.BoolTensor,           # done || trunc
    torch.ByteTensor            # next state
]

# 경험 클래스
@dataclass
class Experience:
    state: State
    action: Action
    reward: float
    done_or_trunc: bool
    next_state: State

# 리플레이 버퍼 클래스
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        self.buffer.append(experience)

    # 미니배치 샘플링
    def sample(self, batch_size: int) -> tt.List[Experience]:
        indices = np.random.choice(len(self), size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]


class Agent:
    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer):
        self.env = env
        self.replay_buffer = replay_buffer
        self.state: tt.Optional[np.ndarray] = None
        self._reset()

    # 환경 및 reward 초기화
    def _reset(self):
        self.state, _ = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(
        self,
        net: dqn_model.DQN,
        device: torch.device,
        epsilon: float = 0.0,
        ) -> tt.Optional[float]:
        '''
        환경에서 한 스텝을 실행하고 보상을 반환한다.
        '''

        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        # 탐험 감소 시작 프레임 이후에는 탐험 감소 최종 프레임까지 탐험 감소
        else:
            # 현재 상태를 텐서로 변환한다.
            state_v = torch.as_tensor(self.state).to(device)
            state_v = state_v.unsqueeze(0)

            # 현재 상태를 네트워크를 통과 시켜서 value를 측정하고 최대 value를 가지는 액션을 선택한다.
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # 선택한 액션을 환경에서 실행한다.
        new_state, reward, done, truncated, _ = self.env.step(action)
        self.total_reward += reward

        # 경험을 리플레이 버퍼에 추가한다.
        exp = Experience(
            state=self.state,
            action=action,
            reward=float(reward),
            done_or_trunc=done or truncated,
            next_state=new_state,
        )
        self.replay_buffer.append(exp)
        self.state = new_state

        # 환경이 종료되었으면 상태를 초기화한다.
        if done or truncated:
            done_reward = self.total_reward
            self._reset()

        # 종료되었으면 보상을 반환한다.
        return done_reward

def batch_to_tensors(batch: tt.List[Experience], device: torch.device) -> BatchTensors:
    '''
    경험을 텐서로 변환한다.
    '''
    # 초기화
    states, actions, rewards, dones, next_states = [], [], [], [], []

    # 배열에 추가
    for e in batch:
        states.append(e.state)
        actions.append(e.action)
        rewards.append(e.reward)
        dones.append(e.done_or_trunc)
        next_states.append(e.next_state)

    # 텐서로 변환
    states_t = torch.as_tensor(np.asarray(states))
    actions_t = torch.LongTensor(actions)
    rewards_t = torch.FloatTensor(rewards)
    dones_t = torch.BoolTensor(dones)
    next_states_t = torch.as_tensor(np.asarray(next_states))
    return states_t.to(device), actions_t.to(device), rewards_t.to(device), \
        dones_t.to(device),  next_states_t.to(device)

def calc_loss(
        batch: tt.List[Experience],
        net: dqn_model.DQN,
        tgt_net: dqn_model.DQN,
        device: torch.device,
        ) -> torch.Tensor:
    '''
    손실을 계산한다.
    '''
    # 텐서로 변환
    states_t, actions_t, rewards_t, dones_t, new_states_t = batch_to_tensors(batch, device)

    # 현재 상태에서 최적의 액션을 선택한다.
    state_action_values = net(states_t).gather(
        1, actions_t.unsqueeze(-1)
    ).squeeze(-1)

    # 다음 상태에서 최적의 액션을 선택한다.
    with torch.no_grad():
        next_state_values = tgt_net(new_states_t).max(1)[0]
        next_state_values[dones_t] = 0.0
        next_state_values = next_state_values.detach()

    # 기대 상태-액션 값을 계산한다.
    expected_state_action_values = next_state_values * GAMMA + rewards_t

    # 손실을 계산한다.
    return nn.MSELoss()(state_action_values, expected_state_action_values)

'''
Q(s,a)와 타깃 Q̂(s,a)의 가중치를 무작위로 초기화
초기 탐험 확률 𝜖 ← 1.0
리플레이 버퍼 초기화

확률 𝜖으로 무작위 행동 a 선택,
아니라면 a = argmaxₐ Q(s,a)

a를 실행하고 보상 r과 다음 상태 s′ 관측

(s, a, r, s′) 튜플을 리플레이 버퍼에 저장

리플레이 버퍼에서 무작위 미니배치 샘플링

각 샘플에 대해 타깃 값 y 계산
(예: y = r + γ·maxₐ′ Q̂(s′, a′))

손실 계산:
ℒ = (Q(s,a) − y)²

손실을 최소화하도록 Q 네트워크 업데이트
→ SGD 알고리즘 사용

매 N스텝마다, Q의 가중치를 Q̂로 복사

수렴할 때까지 2번부터 반복
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu", help="Device name, default=cpu")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    args = parser.parse_args()
    device = torch.device(args.dev)

    # 환경 생성
    env = wrappers.make_env(args.env)

    # 네트워크 생성
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="-" + args.env)
    print("Network:", net)

    # 리플레이 버퍼 생성
    buffer = ReplayBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    # 옵티마이저 생성
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    # 학습 시작
    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, device, epsilon)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            print(f"{frame_idx}: done {len(total_rewards)} games, reward {m_reward:.3f}, "
                  f"eps {epsilon:.2f}, speed {speed:.2f} f/s")
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(net.state_dict(), args.env + "-best_%.0f.dat" % m_reward)
                if best_m_reward is not None:
                    print(f"Best reward updated {best_m_reward:.3f} -> {m_reward:.3f}")
                best_m_reward = m_reward
            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break
        if len(buffer) < REPLAY_START_SIZE:
            continue
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device)
        loss_t.backward()
        optimizer.step()
    writer.close()