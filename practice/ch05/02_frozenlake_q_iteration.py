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
v iteration과 달리 가장 명확한 변화는 value table입니다.
이전 예제에서는 상태의 값만 저장했기 때문에 딕셔너리의 키는 단순히 상태(state)였습니다.
이제는 Q-function의 값을 저장해야 하므로, 딕셔너리의 키는 (State, Action)의 쌍으로 구성된 복합 키가 됩니다.

두 번째 차이점은 calc_action_value() 함수입니다.
이제는 모든 행동 값이 value table에 직접 저장되므로, 이 함수가 더 이상 필요하지 않습니다.

마지막이자 가장 중요한 변화는 에이전트의 value_iteration() 메서드입니다.
이전에는 이 메서드가 단지 calc_action_value()를 호출하여 Bellman 근사를 수행하는 래퍼 역할만 했습니다.
하지만 이제 이 함수가 사라지고 value table이 직접 사용되므로, Bellman 근사를 value_iteration() 메서드 안에서 직접 수행해야 합니다.
'''

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state, _ = self.env.reset()
        self.rewards: tt.Dict[RewardKey, float] = defaultdict(float)
        self.transits: tt.Dict[TransitKey, Counter] = defaultdict(Counter)
        self.values: tt.Dict[TransitKey, float] = defaultdict(float)

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

    def select_action(self, state: State) -> Action:
        '''
        특정 상태에서 최적의 액션을 선택한다.
        '''
        best_action = None
        best_value = None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value: # 최적의 액션을 선택한다.
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
            for action in range(self.env.action_space.n):
                # 모든 액션에 대해 기대 보상을 계산한다.
                action_value = 0.0
                target_counts = self.transits[(state, action)]

                # 전이 횟수의 총합을 계산한다.
                total = sum(target_counts.values())

                # 각 도착 상태에 대한 기대 보상을 계산한다.
                for tgt_state, count in target_counts.items():
                    rw_key = (state, action, tgt_state)
                    reward = self.rewards[rw_key]
                    best_action = self.select_action(tgt_state)
                    val = reward + GAMMA * self.values[(tgt_state, best_action)]
                    action_value += (count / total) * val
                self.values[(state, action)] = action_value

'''
이제는 모든 행동 값이 value table에 직접 저장되므로, 이 함수가 더 이상 필요하지 않습니다.
'''
    # def calc_action_value(self, state: State, action: Action) -> float:
    #     '''
    #     특정 상태에서 특정 액션을 실행했을 때 기대 보상을 계산한다.
    #     '''
    #     # 전이 테이블에서 특정 상태와 액션에 대한 전이 횟수를 가져온다.
    #     target_counts = self.transits[(state, action)]

    #     # 전이 횟수의 총합을 계산한다.
    #     total = sum(target_counts.values())

    #     # 각 도착 상태에 대한 기대 보상을 계산한다.
    #     action_value = 0.0
    #     for tgt_state, count in target_counts.items():
    #         # 보상 테이블에서 특정 상태와 액션에 대한 즉시 보상을 가져온다.
    #         rw_key = (state, action, tgt_state)
    #         reward = self.rewards[rw_key]
    #         val = reward + GAMMA * self.values[tgt_state]
    #         action_value += (count / total) * val
    #     return action_value

if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-iteration")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print(f"{iter_no}: Best reward updated "
                  f"{best_reward:.3} -> {reward:.3}")
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()

'''
Q iteration result
4: Best reward updated 0.0 -> 0.75
18: Best reward updated 0.75 -> 0.8
30: Best reward updated 0.8 -> 0.85
Solved in 30 iterations!
'''

'''
v iteration result
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


'''
calc_action_value 함수는 보상 정보뿐만 아니라 전이 확률 정보도 사용합니다.
이건 value iteration 방법에서는 큰 문제가 아니며, 학습 중 이런 정보에 의존할 수 있기 때문입니다.
하지만 다음 챕터에서는 확률 근사를 하지 않고 환경으로부터 샘플을 직접 받아 사용하는 value iteration의 확장 기법에 대해 배우게 됩니다.
이런 방법들에서는 확률에 대한 의존이 에이전트에게 추가적인 부담으로 작용합니다.
반면 Q-learning에서는, 에이전트가 결정을 내릴 때 필요한 건 단지 Q-값뿐입니다.
'''