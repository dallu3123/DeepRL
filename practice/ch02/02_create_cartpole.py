import gymnasium as gym

e = gym.make("CartPole-v1")

obs, info = e.reset() # reset 함수는 환경을 초기화하고 초기 관측값을 반환 (info 값은 무시)

print(obs) # 카트폴의 초기 관측값
print(info) # 카트폴의 초기 정보 별로 중요하지 않음

print("카트폴의 action space: ", e.action_space)
print("카트폴의 observation space: ", e.observation_space)

# step 함수는 환경에 행동을 반환
print("카트폴 step 함수 호출 결과: ", e.step(0)) # 0은 카트폴의 왼쪽으로 이동하는 행동

e.close()

