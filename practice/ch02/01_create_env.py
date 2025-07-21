import random
from typing import List



class Environment: # 환경 클래스
    def __init__(self):
        self.steps_left = 10


    # 환경 클래스에서 필요한 함수들
    def get_observation(self) -> List[float]: # 환경의 상태 observation 반환
        return [0.0, 0.0, 0.0]

    def get_actions(self) -> List[float]: # 환경에서 가능한 행동들 반환
        return [0, 1]

    def is_done(self) -> bool: # 환경이 종료되었는지 확인
        return self.steps_left == 0

    def action(self, action: int) -> float: # 환경에 행동을 적용하고 보상을 반환
        if self.is_done():
            raise Exception("Game is over")
        self.steps_left -= 1
        return random.random()

class Agent: # 환경에 대한 정보를 받아서 최적의 행동을 결정하는 객체
    def __init__(self):
        self.total_reward = 0.0

    def step(self, env: Environment):
        current_obs = env.get_observation()
        actions = env.get_actions()
        reward = env.action(random.choice(actions))
        self.total_reward += reward

if __name__ == "__main__":
    env = Environment()
    agent = Agent()

    while not env.is_done():
        agent.step(env)

    print("Total reward got: %.4f" % agent.total_reward)
