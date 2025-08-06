import ptan
from lib import DullAgent, ToyEnv, DQNNet

if __name__ == "__main__":
    env = ToyEnv()
    s, _ = env.reset()
    print(f"env.reset() -> {s}")
    s = env.step(1)
    print(f"env.step(1) -> {s}")
    s = env.step(2)
    print(f"env.step(2) -> {s}")
    """
    env.reset() -> 0
    env.step(1) -> (1, 1.0, False, False, {})
    env.step(2) -> (2, 2.0, False, False, {})
    """

    for _ in range(10):
        r = env.step(0)
        print(r)
    """
    (3, 0.0, False, False, {})
    (4, 0.0, False, False, {})
    (0, 0.0, False, False, {})
    (1, 0.0, False, False, {})
    (2, 0.0, False, False, {})
    (3, 0.0, False, False, {})
    (4, 0.0, False, False, {})
    (0, 0.0, True, False, {})
    (0, 0.0, True, False, {})
    (0, 0.0, True, False, {})
    """

    agent = DullAgent(n_actions=1)
    print("agent: ", agent([1,2])[0])
    """
    agent: [1 1]
    """

    env = ToyEnv()
    agent = DullAgent(n_actions=1)
    exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=2)
    for idx, exp in enumerate(exp_source):
        if idx > 15:
            break
        print(exp)

    """
    steps_count=2 이므로 2개의 경험을 생성하면서 15번 반복한다.
    (Experience(state=0, action=1, reward=1.0, done_trunc=False), Experience(state=1, action=1, reward=1.0, done_trunc=False))
    (Experience(state=1, action=1, reward=1.0, done_trunc=False), Experience(state=2, action=1, reward=1.0, done_trunc=False))
    (Experience(state=2, action=1, reward=1.0, done_trunc=False), Experience(state=3, action=1, reward=1.0, done_trunc=False))
    (Experience(state=3, action=1, reward=1.0, done_trunc=False), Experience(state=4, action=1, reward=1.0, done_trunc=False))
    (Experience(state=4, action=1, reward=1.0, done_trunc=False), Experience(state=0, action=1, reward=1.0, done_trunc=False))
    (Experience(state=0, action=1, reward=1.0, done_trunc=False), Experience(state=1, action=1, reward=1.0, done_trunc=False))
    (Experience(state=1, action=1, reward=1.0, done_trunc=False), Experience(state=2, action=1, reward=1.0, done_trunc=False))
    (Experience(state=2, action=1, reward=1.0, done_trunc=False), Experience(state=3, action=1, reward=1.0, done_trunc=False))
    (Experience(state=3, action=1, reward=1.0, done_trunc=False), Experience(state=4, action=1, reward=1.0, done_trunc=True))
    (Experience(state=4, action=1, reward=1.0, done_trunc=True),)
    (Experience(state=0, action=1, reward=1.0, done_trunc=False), Experience(state=1, action=1, reward=1.0, done_trunc=False))
    (Experience(state=1, action=1, reward=1.0, done_trunc=False), Experience(state=2, action=1, reward=1.0, done_trunc=False))
    (Experience(state=2, action=1, reward=1.0, done_trunc=False), Experience(state=3, action=1, reward=1.0, done_trunc=False))
    (Experience(state=3, action=1, reward=1.0, done_trunc=False), Experience(state=4, action=1, reward=1.0, done_trunc=False))
    (Experience(state=4, action=1, reward=1.0, done_trunc=False), Experience(state=0, action=1, reward=1.0, done_trunc=False))
    (Experience(state=0, action=1, reward=1.0, done_trunc=False), Experience(state=1, action=1, reward=1.0, done_trunc=False))
    """

    exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=4)
    print(next(iter(exp_source)))
    """
    steps_count=4 이므로 4개의 경험을 생성한다.
    (Experience(state=0, action=1, reward=1.0, done_trunc=False), Experience(state=1, action=1, reward=1.0, done_trunc=False), Experience(state=2, action=1, reward=1.0, done_trunc=False), Experience(state=3, action=1, reward=1.0, done_trunc=False))
    """

    exp_source = ptan.experience.ExperienceSource(env=[ToyEnv(), ToyEnv()], agent=agent, steps_count=2)
    for idx, exp in enumerate(exp_source):
        if idx > 4:
            break
        print(exp)

    """
    ToyEnv()이 2개가 있어서 2개의 환경에서 번갈아가며 실행된다.
    (Experience(state=0, action=1, reward=1.0, done_trunc=False), Experience(state=1, action=1, reward=1.0, done_trunc=False))
    (Experience(state=0, action=1, reward=1.0, done_trunc=False), Experience(state=1, action=1, reward=1.0, done_trunc=False))
    (Experience(state=1, action=1, reward=1.0, done_trunc=False), Experience(state=2, action=1, reward=1.0, done_trunc=False))
    (Experience(state=1, action=1, reward=1.0, done_trunc=False), Experience(state=2, action=1, reward=1.0, done_trunc=False))
    (Experience(state=2, action=1, reward=1.0, done_trunc=False), Experience(state=3, action=1, reward=1.0, done_trunc=False))
    """

