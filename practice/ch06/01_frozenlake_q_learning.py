import typing as tt
import gymnasium as gym
from collections import defaultdict
from torch.utils.tensorboard.writer import SummaryWriter

ENV_NAME = "FrozenLake-v1"
#ENV_NAME = "FrozenLake8x8-v1"      # uncomment for larger version
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20

State = int
Action = int

# Q 테이블의 키 타입
ValuesKey = tt.Tuple[State, Action]

'''
Q-learning 알고리즘의 기본 구조:
1. 빈 Q 테이블을 생성한다.
2. (s, a, r, s′) 튜플을 환경에서 얻는다.
3. 아래의 Bellman 업데이트를 수행한다:
Q(s, a) ← Q(s, a) + α[r + γmax_a′ Q(s′, a′) − Q(s, a)]
4. 테스트 에피소드를 실행하여 성능을 평가한다.
5. 성능이 향상될 때까지 1~4를 반복한다.
'''

class Agent:
    def __init__(self):
        # 환경 생성 + 초기 상태 설정 + Q 테이블 초기화
        self.env = gym.make(ENV_NAME)
        self.state, _ = self.env.reset()
        self.values: tt.Dict[ValuesKey] = defaultdict(float)

    def sample_env(self) -> tt.Tuple[State, Action, float, State]:
        '''
        환경에서 랜덤 액션을 실행하고 결과를 반환한다.
        (s, a, r, s′) 튜플을 환경에서 얻는다.
        '''
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, done, truncated, _ = self.env.step(action)

        # 환경 초기화
        if done or truncated:
            self.state, _ = self.env.reset()
        else:
            self.state = new_state
        return old_state, action, float(reward), new_state

    def best_value_and_action(self, state: State) -> tt.Tuple[float, Action]:
        '''
        특정 상태에서 최적의 액션을 선택한다.
        '''
        # 모든 액션에 대해 기대 보상을 계산하고 최적의 액션을 선택한다.
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, state: State, action: Action, reward: float, next_state: State):
        '''
        특정 상태에서 특정 액션을 실행했을 때 기대 보상을 계산한다.
        '''
        # 다음 상태에서 최적의 액션을 선택한다.
        best_val, _ = self.best_value_and_action(next_state)

        # 기대 보상을 계산한다.
        new_val = reward + GAMMA * best_val

        # 기존 값을 업데이트한다.
        old_val = self.values[(state, action)]
        key = (state, action)

        # Q 테이블을 업데이트한다.
        # Q(s, a) ← Q(s, a) + α[r + γmax_a′ Q(s′, a′) − Q(s, a)]
        self.values[key] = old_val * (1-ALPHA) + new_val * ALPHA

    def play_episode(self, env: gym.Env) -> float:
        '''
        환경에서 에피소드를 실행하고 총 보상을 반환한다.
        '''
        total_reward = 0.0
        state, _ = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            if done or truncated:
                break
            state = new_state
        return total_reward

if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        state, action, reward, next_state = agent.sample_env()
        agent.value_update(state, action, reward, next_state)

        test_reward = 0.0
        for _ in range(TEST_EPISODES):
            test_reward += agent.play_episode(test_env)
        test_reward /= TEST_EPISODES
        writer.add_scalar("reward", test_reward, iter_no)
        if test_reward > best_reward:
            print("%d: Best test reward updated %.3f -> %.3f" % (iter_no, best_reward, test_reward))
            best_reward = test_reward
        if test_reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()

'''
result
588: Best test reward updated 0.000 -> 0.050
605: Best test reward updated 0.050 -> 0.200
934: Best test reward updated 0.200 -> 0.250
943: Best test reward updated 0.250 -> 0.300
1440: Best test reward updated 0.300 -> 0.400
1445: Best test reward updated 0.400 -> 0.450
4726: Best test reward updated 0.450 -> 0.600
5090: Best test reward updated 0.600 -> 0.650
5149: Best test reward updated 0.650 -> 0.700
9289: Best test reward updated 0.700 -> 0.750
9338: Best test reward updated 0.750 -> 0.800
9340: Best test reward updated 0.800 -> 0.850
Solved in 9340 iterations!
'''

'''
Q-iteration보다 느린 이유
이전 Chapter 5의 q_iteration.py는 테스트 중에도 Q 테이블을 업데이트함.

이번 예제는 테스트 에피소드에서 Q값 수정하지 않음.

그 결과로 수렴 속도가 느려졌지만, 전체 샘플 수는 거의 동일.
'''