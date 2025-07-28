import typing as tt
import gymnasium as gym
from collections import defaultdict, Counter
from torch.utils.tensorboard.writer import SummaryWriter


ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
TEST_EPISODES = 20

State = int
Action = int

'''
보상 테이블
복합 키 "출발 상태" + "액션" + "도착 상태"를 갖는 딕셔너리.
값은 해당 전이에서 받은 즉시 보상이다.
'''
RewardKey = tt.Tuple[State, Action, State] # 보상 키


'''
전이 테이블
복합 키 "출발 상태" + "액션"을 갖는 딕셔너리.
값은 도착 상태와 즉시 보상을 포함하는 튜플이다.

예를 들어, 상태 0에서 액션 1을 10번 실행했더니
3번은 상태 4로, 7번은 상태 5로 이동했다고 하자.
그러면 (0, 1) 키에 대한 값은 {4: 3, 5: 7}이 된다.
'''
TransitKey = tt.Tuple[State, Action] # 전이 테이블을 위한 키

'''
코드의 전체 흐름은 다음과 같이 간단하다:

환경에서 100번의 랜덤 스텝을 실행해
보상 테이블과 전이 테이블을 채운다.

그 후, 모든 상태에 대해 Value Iteration을 수행해
가치 테이블을 갱신한다.

그런 다음, 여러 번의 전체 에피소드를 실행하여
개선된 정책으로 얻은 평균 보상을 확인한다.

이때 평균 보상이 0.8 이상이면 학습을 종료한다.

테스트 에피소드에서도 보상/전이 테이블은 계속 갱신하여
환경으로부터 더 많은 데이터를 활용한다.
'''

class Agent:
    def __init__(self):
        # 환경 만들기
        self.env = gym.make(ENV_NAME)

        # 환경 초기화
        self.state, _ = self.env.reset()

        # 보상 테이블, 전이 테이블 초기화
        self.rewards: tt.Dict[RewardKey, float] = defaultdict(float)
        self.transits: tt.Dict[TransitKey, Counter] = defaultdict(Counter)

        '''
        가치 테이블
        각 상태를 키로 하고 해당 상태의 추정 가치(Value)를 값으로 하는 딕셔너리.
        '''
        self.values: tt.Dict[State, float] = defaultdict(float)

    def play_n_random_steps(self, n: int):
        '''
        랜덤 액션을 n번 실행하고 보상 테이블과 전이 테이블을 업데이트한다.
        '''
        for _ in range(n):
            # 랜덤 액션 실행
            action = self.env.action_space.sample()
            new_state, reward, is_done, is_trunc, _ = self.env.step(action)

            # 보상 테이블 업데이트
            rw_key = (self.state, action, new_state)
            self.rewards[rw_key] = float(reward)

            # 전이 테이블 업데이트
            tr_key = (self.state, action)
            self.transits[tr_key][new_state] += 1

            # 환경 초기화
            if is_done or is_trunc:
                self.state, _ = self.env.reset()
            else:
                self.state = new_state

    def calc_action_value(self, state: State, action: Action) -> float:
        '''
        특정 상태에서 특정 액션을 실행했을 때 기대 보상을 계산한다.
        '''
        # 전이 테이블에서 특정 상태와 액션에 대한 전이 횟수를 가져온다.
        target_counts = self.transits[(state, action)]

        # 전이 횟수의 총합을 계산한다.
        total = sum(target_counts.values())

        # 각 도착 상태에 대한 기대 보상을 계산한다.
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            # 보상 테이블에서 특정 상태와 액션에 대한 즉시 보상을 가져온다.
            rw_key = (state, action, tgt_state)
            reward = self.rewards[rw_key]
            val = reward + GAMMA * self.values[tgt_state]
            action_value += (count / total) * val
        return action_value

    def select_action(self, state: State) -> Action:
        '''
        특정 상태에서 최적의 액션을 선택한다.
        '''
        best_action, best_value = None, None

        # 모든 액션에 대해 기대 보상을 계산하고 최적의 액션을 선택한다.
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)

            # 최적의 액션을 선택한다.
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env: gym.Env) -> float:
        '''
        환경에서 에피소드를 실행하고 총 보상을 반환한다.
        '''
        total_reward = 0.0
        state, _ = env.reset()

        # 에피소드 실행
        while True:
            # 최적의 액션을 선택한다.
            action = self.select_action(state)

            # 환경에서 액션을 실행하고 보상 테이블과 전이 테이블을 업데이트한다.
            new_state, reward, is_done, is_trunc, _ = env.step(action)
            rw_key = (state, action, new_state)
            self.rewards[rw_key] = float(reward)
            tr_key = (state, action)
            self.transits[tr_key][new_state] += 1
            total_reward += reward

            # 에피소드 종료 조건
            if is_done or is_trunc:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        '''
        가치 반복을 실행한다.
        '''
        # 모든 상태에 대해 가치 반복을 실행한다.
        for state in range(self.env.observation_space.n):
            # 모든 액션에 대해 기대 보상을 계산한다.
            state_values = [
                self.calc_action_value(state, action)
                for action in range(self.env.action_space.n)
            ]
            # 가치 테이블을 업데이트한다.
            self.values[state] = max(state_values)

if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-v-iteration")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = 0.0
        # 테스트 에피소드 실행
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)

        # 최적 보상 갱신 시 출력
        if reward > best_reward:
            print(f"{iter_no}: Best reward updated {best_reward:.3} -> {reward:.3}")
            best_reward = reward
        # 최적 보상 달성 시 종료
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()


'''
result
3: Best reward updated 0.0 -> 0.05
4: Best reward updated 0.05 -> 0.1
6: Best reward updated 0.1 -> 0.3
17: Best reward updated 0.3 -> 0.45
19: Best reward updated 0.45 -> 0.5
20: Best reward updated 0.5 -> 0.6
21: Best reward updated 0.6 -> 0.65
40: Best reward updated 0.65 -> 0.75
86: Best reward updated 0.75 -> 0.8
336: Best reward updated 0.8 -> 0.9
Solved in 336 iterations!
'''