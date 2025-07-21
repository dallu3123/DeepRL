import os
import typing as tt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

import gymnasium as gym
from dataclasses import dataclass

class CEM(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(CEM, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.layers(x)

# agent가 에피소드에서 수행한 단일 스텝
@dataclass
class EpisodeStep:
    observation: np.ndarray
    action: int
    # 환경에서의 관측값과 행동을 저장

# 하나의 에피소드 전체
@dataclass
class Episode:
    reward: float
    steps: tt.List[EpisodeStep]
    # 에피소드의 총 보상과 여러개의 에피소드 스텝을 저장

# 에피소드 배치를 생성하는 함수
def iterate_batches(env: gym.Env, net: nn.Module, batch_size: int):
    # 배치 초기화
    batch = []

    # 총 보상을 저장할 보상 카운터
    episode_reward = 0.0

    # 에피소드 스텝 리스트(EpisodeStep 객체들의 리스트)
    episode_steps = []

    # 환경 초기화 및 첫 관측값 얻기
    obs, _ = env.reset()

    # 신경망의 출력을 행동 확률 분포로 변환하기 위한 Softmax
    sm = nn.Softmax(dim=1)

    while True:
        # obs값을 텐서로 변환
        obs_v = torch.tensor(obs, dtype=torch.float32)
        '''
        obs_v.unsqueeze(0)
        - 차원 추가
        - 1차원 텐서를 2차원 텐서로 변환
        Pytorch의 모든 nn.Module은 데이터 배치(batch)를 처리할 수 있도록 설계되어 있음
        - 모든 모듈은 입력 텐서의 첫 번째 차원을 배치 차원으로 취급
        - 따라서 모듈에 전달되는 입력 텐서는 항상 2차원 형태여야 함
        - 모듈이 1차원 텐서를 처리해야 하는 경우, unsqueeze(0)을 사용하여 차원을 추가해야 함
        - 이렇게 하면 입력 텐서가 (1, N) 형태로 변환되어 모듈에 전달됨
        '''
        act_probs_v = sm(net(obs_v.unsqueeze(0)))

        # .data필드를 통해서 gradient 추적없이 텐서 값만 추출한 후 numpy 배열로 변환
        act_probs = act_probs_v.data.numpy()[0]

        # 행동 확률 분포에서 샘플링
        action = np.random.choice(len(act_probs), p=act_probs)

        # 뽑아낸 행동을 환경에 전달하고 다음 관측값, 보상, 종료 여부 등을 얻음
        next_obs, reward, done, truncated, _ = env.step(action)

        # 보상 카운터에 보상 값을 더함
        episode_reward += float(reward)

        # 현재 관측값과 선택한 행동을 저장
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)

        # 에피소드가 종료되거나 트레닝 중단 조건을 만족하면 에피소드 정보를 배치에 추가
        if done or truncated:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset() # 환경 초기화 및 다음 관측값 얻기

            # yield: 현재까지 모은 배치를 반환하고, 함수의 실행 상태를 유지한 채 다음 호출 때 이어서 실행됨
            # 배치 크기에 도달하면 현재 배치를 반환하고 초기화
            if len(batch) == batch_size:
                yield batch
                batch = []

        # 다음 관측값을 현재 관측값으로 업데이트
        obs = next_obs

def filter_batch(batch: tt.List[Episode], percentile: float)-> \
    tt.Tuple[torch.FloatTensor, torch.LongTensor, float, float]:

    # 에피소드의 보상 값을 추출
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = float(np.percentile(rewards, percentile)) # 상위 percentile 보상 값
    reward_mean = float(np.mean(rewards)) # 평균 보상 값(모니터링용)

    train_obs : tt.List[np.ndarray] = []
    train_act : tt.List[int] = []

    for episode in batch:

        # 보상 값이 임계값보다 미만이면 훈련 데이터에 추가 X
        if episode.reward < reward_bound:
            continue
        # 보상 임계값 보다 높은 에피소드의 관측값과 행동을 훈련 데이터에 추가
        train_obs.extend(map(lambda s: s.observation, episode.steps))
        train_act.extend(map(lambda s: s.action, episode.steps))

    train_obs_v = torch.FloatTensor(np.vstack(train_obs))
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean
