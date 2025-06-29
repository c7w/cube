import numpy as np
from environment.rewards.reward import SparseReward, DenseReward, HybridReward

def make_solved_state():
    state = np.zeros((6, 9), dtype=int)
    for i in range(6):
        state[i, :] = i
    return state

def make_unsolved_state():
    state = make_solved_state()
    state[0, 0] = 1  # 打乱一个块
    return state

def test_sparse_reward():
    reward_fn = SparseReward()
    s1 = make_unsolved_state()
    s2 = make_solved_state()
    r = reward_fn.calculate_reward(s1, 'F', s2, True, 10, 100)
    assert r > 0
    r2 = reward_fn.calculate_reward(s1, 'F', s1, False, 10, 100)
    assert r2 <= 0

def test_dense_reward():
    reward_fn = DenseReward()
    s1 = make_unsolved_state()
    s2 = make_solved_state()
    r = reward_fn.calculate_reward(s1, 'F', s2, True, 10, 100)
    assert r > 0
    r2 = reward_fn.calculate_reward(s1, 'F', s1, False, 10, 100)
    assert r2 <= 0

def test_hybrid_reward():
    reward_fn = HybridReward()
    s1 = make_unsolved_state()
    s2 = make_solved_state()
    r = reward_fn.calculate_reward(s1, 'F', s2, True, 10, 100)
    assert r > 0
    r2 = reward_fn.calculate_reward(s1, 'F', s1, False, 10, 100)
    assert r2 <= 0 