import gymnasium as gym

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", render_mode="ansi")

    # 관측 공간 확인
    print("FrozenLake-v1 관측 공간 크기:", env.observation_space)

    # 행동 공간 확인
    print("FrozenLake-v1 행동 공간 크기:", env.action_space)

    obs, info = env.reset()
    print(obs)
    print(info)

    print(env.render())

    obs, reward, terminated, truncated, info = env.step(1)

'''실행 결과
FrozenLake-v1 관측 공간 크기: Discrete(16) -> 격자 안에 위치를 의미함
FrozenLake-v1 행동 공간 크기: Discrete(4) -> 격자 안에서 움직이는 방향을 의미함

관측값: 0
정보: {'prob': 1}

환경 렌더링:
SFFF
FHFH
FFFH
HFFG
'''