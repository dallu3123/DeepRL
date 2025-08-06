import numpy as np
import ptan

q_vals = np.array([[1, 2, 3], [1, -1, 0]])

print(q_vals)

selector = ptan.actions.ArgmaxActionSelector()
print(selector(q_vals)) # 최대 행동 값이 있는 인덱스를 반환

'''
출력결과
[[ 1  2  3]
 [ 1 -1  0]]

첫번째 행에서 최대 행동 값이 있는 인덱스는 2
두번째 행에서 최대 행동 값이 있는 인덱스는 0

[2 0]
'''

selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=0.5)
print(selector(q_vals))

'''
출력결과

EpsilonGreedyActionSelector는 행동 값이 최대인 인덱스를 반환하는 것이 아니라 확률에 따라 행동을 선택
[1 0]
'''

selector = ptan.actions.ProbabilityActionSelector()
for _ in range(10):
    acts = selector(np.array([
        [0.1, 0.8, 0.1],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
    ]))
    print(acts)

'''
출력결과
[1 2 0]
[1 2 1]
[0 2 0]
[1 2 1]
[1 2 1]
[1 2 0]
[1 2 1]
[0 2 0]
[1 2 1]
[1 2 0]
'''

