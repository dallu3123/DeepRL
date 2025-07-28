import typing as tt
import gymnasium as gym
from gymnasium import spaces
import collections
import numpy as np
from stable_baselines3.common import atari_wrappers # type: ignore

'''
주요 변환 및 관련 Wrapper 설명:
1. 라이프를 개별 에피소드로 나누기 (EpisodicLifeEnv)
- 하나의 전체 에피소드를 라이프 단위로 쪼갬
- 에피소드가 짧아져 수렴 속도 증가
- Pong 등 일부 게임에서만 사용 가능

2. 랜덤 No-op 수행 (NoopResetEnv)
- 게임 시작 시 무작위로 최대 30번까지 아무 행동도 안 함
- 시작 화면 스킵 목적

3. K step 마다 액션 결정 (MaxAndSkipEnv)
- K(보통 4) 프레임 동안 같은 액션 반복
- 두 프레임 간 픽셀 최대값 사용 (플리커 현상 방지 포함)

4. 게임 시작 시 FIRE 누르기 (FireResetEnv)
- Pong, Breakout처럼 시작 시 FIRE가 필요할 때 사용
- 안 누르면 환경이 POMDP로 인식될 수 있음

5. 프레임 크기 축소 및 흑백 처리 (WarpFrame)
- 210×160 RGB → 84×84 grayscale로 변환
- DeepMind 논문에 기반

6. 최근 k개의 프레임 스택 (BufferWrapper)
- k개 프레임을 스택해 한 상태로 제공
- 방향, 속도 등 동적 정보 반영
- 관찰값은 복사본으로 반환해 replay buffer 오염 방지
- 가장 마지막에 적용되어야 함

7. 보상 클리핑 (ClipRewardEnv)
- -1, 0, 1로 보상 값을 제한
- 다양한 게임에서 스케일 차이 완화

8. PyTorch 형식으로 텐서 차원 재배열 (ImageToPyTorch)
- HWC → CHW로 전환
- PyTorch의 Conv layer가 요구하는 입력 형식으로 맞춤
'''

# 이미지를 PyTorch 텐서로 변환
class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        obs = self.observation_space
        assert isinstance(obs, gym.spaces.Box)
        assert len(obs.shape) == 3
        new_shape = (obs.shape[-1], obs.shape[0], obs.shape[1])
        self.observation_space = gym.spaces.Box(
            low=obs.low.min(), high=obs.high.max(),
            shape=new_shape, dtype=obs.dtype)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

# 버퍼 랩퍼
class BufferWrapper(gym.ObservationWrapper):
    # 버퍼 랩퍼 초기화
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        obs = env.observation_space
        assert isinstance(obs, spaces.Box)
        new_obs = gym.spaces.Box(
            obs.low.repeat(n_steps, axis=0), obs.high.repeat(n_steps, axis=0),
            dtype=obs.dtype)
        self.observation_space = new_obs
        self.buffer = collections.deque(maxlen=n_steps)

    # 리셋
    def reset(self, *, seed: tt.Optional[int] = None, options: tt.Optional[dict[str, tt.Any]] = None):
        for _ in range(self.buffer.maxlen-1):
            self.buffer.append(self.env.observation_space.low)
        obs, extra = self.env.reset()
        return self.observation(obs), extra

    # 관측 변환
    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.buffer.append(observation)
        return np.concatenate(self.buffer)


# 환경 생성
def make_env(env_name: str, **kwargs):
    env = gym.make(env_name, **kwargs)
    env = atari_wrappers.AtariWrapper(env, clip_reward=False, noop_max=0)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, n_steps=4)
    return env
