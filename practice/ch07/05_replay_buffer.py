from lib import ToyEnv, DullAgent, ptan

if __name__ == "__main__":
    env = ToyEnv()
    agent = DullAgent(n_actions=1)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=1.0, steps_count=1)


    # 경험을 저장하는 버터 ptan.experience.ExperienceReplayBuffer
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=100)

    for step in range(6):
        # buffer에 경험을 추가하는 함수 populate()
        buffer.populate(1)

        if len(buffer) < 5:
            continue

        # buffer에서 경험을 랜덤샘플링하는 함수 sample()
        batch = buffer.sample(4)
        print("Train time, %d batch samples" % len(batch))
        for s in batch:
            print(s)

    """
    Train time, 4 batch samples
    ExperienceFirstLast(state=2, action=1, reward=1.0, last_state=3)
    ExperienceFirstLast(state=1, action=1, reward=1.0, last_state=2)
    ExperienceFirstLast(state=4, action=1, reward=1.0, last_state=0)
    ExperienceFirstLast(state=3, action=1, reward=1.0, last_state=4)
    Train time, 4 batch samples
    ExperienceFirstLast(state=1, action=1, reward=1.0, last_state=2)
    ExperienceFirstLast(state=1, action=1, reward=1.0, last_state=2)
    ExperienceFirstLast(state=3, action=1, reward=1.0, last_state=4)
    ExperienceFirstLast(state=0, action=1, reward=1.0, last_state=1)
    """