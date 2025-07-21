import gymnasium as gym
import random

class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env : gym.Env, epsilon : float = 0.1): # 기존에 action에서 epsilon 확률로 랜덤 행동을 반환하는 래퍼 클래스
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    # 래퍼 클래스의 action 함수를 오버라이딩
    def action(self, action : gym.core.WrapperActType) -> gym.core.WrapperActType:
        if random.random() < self.epsilon: # 랜덤 행동 반환
            action = self.env.action_space.sample()
            print(f"Random action {action}")
            return action
        return action

if __name__ == "__main__":
    env = RandomActionWrapper(gym.make("CartPole-v1")) # 랜덤 action     obs, info = env.reset()
    total_reward = 0.0
    env.reset()

    while True:
        obs, reward, is_done, is_trunc, _ = env.step(0)
        total_reward += reward
        if is_done:
            break

    print(f"Reward got: {total_reward:.2f}")