import os
import typing as tt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

import gymnasium as gym
from dataclasses import dataclass

HIDDEN_SIZE = 128   # 은닉층의 뉴런 개수
BATCH_SIZE = 16     # 배치 크기
PERCENTILE = 70     # 상위 70% 이상의 성공 확률을 가진 행동을 선택하는 데 사용되는 임계값


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

# 에피소드 배치를 생성하는 함수 - 일단은 batch_size만큼 에피소드 저장
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

# Elite set를 만들어주는 필터링하는 함수
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

if __name__ == "__main__":
    '''
    필요한 객체들 
    - 환경
    - 신경망
    - 손실 함수
    - 옵티마이저
    - 텐서보드 작성자(모니터링)
    '''

    # 환경 생성
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    video_path = os.path.join(project_root, "videos/cartpole-cem")
    env = gym.wrappers.RecordVideo(env, video_folder=video_path, name_prefix="cartpole-cem")

    # 관측 공간 및  행동 공간 확인
    assert env.observation_space.shape is not None
    obs_size = env.observation_space.shape[0]
    assert isinstance(env.action_space, gym.spaces.Discrete)
    n_actions = env.action_space.n

    # 신경망 생성
    CEM_net = CEM(obs_size, HIDDEN_SIZE, n_actions)
    print(CEM_net)

    # 손실 함수(목적 함수) CrossEntropyLoss 사용
    objective = nn.CrossEntropyLoss()

    # 옵티마이저 생성
    optimizer = optim.Adam(params=CEM_net.parameters(), lr=1e-2)

    # 텐서보드 작성자 생성
    writer = SummaryWriter(comment="-cartpole-cem")

    for iter_no, batch in enumerate(iterate_batches(env, CEM_net, BATCH_SIZE)):
        # 배치 필터링
        train_obs_v, train_act_v, reward_bound, reward_mean = filter_batch(batch, PERCENTILE)

        # 신경망 훈련
        optimizer.zero_grad()
        action_scores_v = CEM_net(train_obs_v)
        loss_v = objective(action_scores_v, train_act_v)
        loss_v.backward()
        optimizer.step()

        # 훈련 결과 출력
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            iter_no, loss_v.item(), reward_mean, reward_bound
        ))

        # 텐서보드에 기록
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_mean", reward_mean, iter_no)
        writer.add_scalar("reward_bound", reward_bound, iter_no)

        # 훈련 성공 조건
        if reward_mean > 475:
            print("Solved!")
            break

    writer.close()

'''결과
CEM(
  (layers): Sequential(
    (0): Linear(in_features=4, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=2, bias=True)
  )
)
0: loss=0.678, reward_mean=17.6, rw_bound=19.5
1: loss=0.690, reward_mean=19.1, rw_bound=22.0
2: loss=0.666, reward_mean=21.2, rw_bound=23.5
3: loss=0.689, reward_mean=24.8, rw_bound=27.5
4: loss=0.669, reward_mean=26.2, rw_bound=27.0
5: loss=0.678, reward_mean=28.3, rw_bound=35.0
6: loss=0.665, reward_mean=36.2, rw_bound=39.5
7: loss=0.641, reward_mean=23.8, rw_bound=25.0
8: loss=0.650, reward_mean=32.8, rw_bound=38.0
9: loss=0.648, reward_mean=44.0, rw_bound=52.5
10: loss=0.652, reward_mean=39.1, rw_bound=45.5
11: loss=0.637, reward_mean=42.5, rw_bound=55.0
12: loss=0.625, reward_mean=38.1, rw_bound=48.5
13: loss=0.625, reward_mean=50.4, rw_bound=45.0
14: loss=0.615, reward_mean=60.6, rw_bound=78.0
15: loss=0.613, reward_mean=51.2, rw_bound=61.0
16: loss=0.621, reward_mean=81.4, rw_bound=91.0
17: loss=0.613, reward_mean=58.2, rw_bound=57.0
18: loss=0.609, reward_mean=69.6, rw_bound=79.0
19: loss=0.623, reward_mean=68.1, rw_bound=67.5
20: loss=0.607, reward_mean=77.8, rw_bound=78.5
21: loss=0.599, reward_mean=66.3, rw_bound=73.0
22: loss=0.595, reward_mean=103.2, rw_bound=129.0
23: loss=0.601, reward_mean=123.4, rw_bound=136.5
24: loss=0.605, reward_mean=155.9, rw_bound=167.5
25: loss=0.589, reward_mean=134.3, rw_bound=158.0
26: loss=0.588, reward_mean=144.5, rw_bound=160.5
27: loss=0.575, reward_mean=136.7, rw_bound=165.0
28: loss=0.587, reward_mean=186.4, rw_bound=228.0
29: loss=0.588, reward_mean=200.4, rw_bound=198.5
30: loss=0.575, reward_mean=219.4, rw_bound=255.0
31: loss=0.575, reward_mean=197.1, rw_bound=242.5
32: loss=0.585, reward_mean=230.9, rw_bound=259.0
33: loss=0.573, reward_mean=197.6, rw_bound=260.5
34: loss=0.572, reward_mean=233.0, rw_bound=261.0
35: loss=0.569, reward_mean=254.5, rw_bound=339.5
36: loss=0.566, reward_mean=231.0, rw_bound=286.0
37: loss=0.558, reward_mean=274.1, rw_bound=294.0
38: loss=0.563, reward_mean=316.2, rw_bound=377.0
39: loss=0.560, reward_mean=370.2, rw_bound=500.0
40: loss=0.562, reward_mean=309.0, rw_bound=408.5
41: loss=0.559, reward_mean=315.5, rw_bound=427.0
42: loss=0.556, reward_mean=314.1, rw_bound=486.0
43: loss=0.566, reward_mean=315.0, rw_bound=405.0
44: loss=0.552, reward_mean=311.0, rw_bound=480.0
45: loss=0.559, reward_mean=385.8, rw_bound=500.0
46: loss=0.566, reward_mean=388.4, rw_bound=483.0
47: loss=0.553, reward_mean=395.9, rw_bound=473.5
48: loss=0.557, reward_mean=403.4, rw_bound=500.0
49: loss=0.557, reward_mean=411.8, rw_bound=500.0
50: loss=0.555, reward_mean=452.8, rw_bound=500.0
51: loss=0.559, reward_mean=467.9, rw_bound=500.0
52: loss=0.559, reward_mean=497.3, rw_bound=500.0
Solved!
'''