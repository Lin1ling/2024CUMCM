import numpy as np 
from scipy.stats import beta
from bellman_3 import * 


def get_ps(n, p, size):
    k = int(p * n)  # 观测到的次品数
    alpha_prior = 1
    beta_prior = 1

    alpha_post = alpha_prior + k  # 后验分布的alpha参数
    beta_post = beta_prior + n - k  # 后验分布的beta参数

    posterior_samples = beta.rvs(alpha_post, beta_post, size=size)
    # print(posterior_samples)
    return posterior_samples

def value_iterations(num_states, num_actions, P, R, gamma, theta=1e-5):
    V = np.zeros(num_states)  # 初始化价值函数 V(s)
    
    while True:
        delta = 0
        for s in range(num_states):
            v = V[s]
            Q_s = np.zeros(num_actions)
            for a in range(num_actions):
                for i in range(0, P.shape[0]):
                    Q_s[a] += sum([P[i].transpose((1, 0, 2))[s, a, s_prime] * (R[i][s, a] + gamma * V[s_prime]) for s_prime in range(num_states)])
                Q_s[a] = Q_s[a] / P.shape[0]
            V[s] = max(Q_s)
            delta = max(delta, abs(v - V[s]))
            
        if delta < theta:
            break

    # 策略提取
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        Q_s = np.zeros(num_actions)
        for a in range(num_actions):
            for i in range(0, P.shape[0]):
                Q_s[a] += sum([P[i].transpose((1, 0, 2))[s, a, s_prime] * (R[i][s, a] + gamma * V[s_prime]) for s_prime in range(num_states)])
                Q_s[a] = Q_s[a] / P.shape[0]
        policy[s] = np.argmax(Q_s)
    
    return V, policy

def value_iterations_1(num_states, num_actions, P, R, gamma, theta=1e-5):
    V = np.zeros(num_states)  # 初始化价值函数 V(s)
    # 先验情况下可以进行检1的状态
    exam_before = [[x for x in range(16, 32)], [4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 39, 52, 53, 54, 55], [x for x in range(1, 64, 4)]]
    exam_after = [[x for x in range(48, 64)], [12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 60, 61, 62, 63], [x for x in range(3, 64, 4)]]

    while True:
        delta = 0
        for s in range(num_states):
            v = V[s]
            Q_s = np.zeros(num_actions)
            for a in range(num_actions):
                if a in [0,1,2]:
                    if s in exam_before[a]:
                        continue
                for i in range(0, P.shape[0]):
                    Q_s[a] += sum([P[i].transpose((1, 0, 2))[s, a, s_prime] * (R[i][s, a] + gamma * V[s_prime]) for s_prime in range(num_states)])
                Q_s[a] = Q_s[a] / P.shape[0]
            V[s] = max(Q_s)
            delta = max(delta, abs(v - V[s]))
            
        if delta < theta:
            break

    # 策略提取
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        Q_s = np.zeros(num_actions)
        for a in range(num_actions):
            for i in range(0, P.shape[0]):
                Q_s[a] += sum([P[i].transpose((1, 0, 2))[s, a, s_prime] * (R[i][s, a] + gamma * V[s_prime]) for s_prime in range(num_states)])
                Q_s[a] = Q_s[a] / P.shape[0]
        policy[s] = np.argmax(Q_s)
    
    return V, policy

def fill_P(matrix):
    for num in range(0, matrix.shape[0]):
        P = matrix[num]
        for index in range(0, P.shape[0]):
            row = P[index]
            if np.all(row == 0):
                matrix[num][index][index] = 1
    return matrix


def get_PS(num_states, num_actions, p1, p2, p3, pc):
    n = len(p1)
    Ps = np.zeros((n, num_actions, num_states, num_states))
    for i in range(0, n):
        p1_h = p1[i] / (1 - (1-p1[i]) * (1-p2[i]) * (1-p3[i]) * (1-pc[i]))
        p2_h = p2[i] / (1 - (1-p1[i]) * (1-p2[i]) * (1-p3[i]) * (1-pc[i]))
        p3_h = p3[i] / (1 - (1-p1[i]) * (1-p2[i]) * (1-p3[i]) * (1-pc[i]))
        P = get_P(num_states, num_actions, p1[i], p2[i], p3[i], pc[i], p1_h, p2_h, p3_h)
        Ps[i] = P
    return Ps

def get_RS(num_states, num_actions, p1, w1, d1, p2, w2, d2, p3, w3, d3, pc, wc, dc, s, r, t):
    n = len(p1)
    Rs = np.zeros((n, num_states, num_actions))
    for i in range(0, n):
        p1_h = p1[i] / (1 - (1-p1[i]) * (1-p2[i]) * (1-p3[i]) * (1-pc[i]))
        p2_h = p2[i] / (1 - (1-p1[i]) * (1-p2[i]) * (1-p3[i]) * (1-pc[i]))
        p3_h = p3[i] / (1 - (1-p1[i]) * (1-p2[i]) * (1-p3[i]) * (1-pc[i]))
        R = get_R(num_states, num_actions, p1[i], w1, d1, p2[i], w2, d2, p3[i], w3, d3, pc[i], wc, dc, s, r, t, p1_h, p2_h, p3_h)
        Rs[i] = R
    return Rs


def bellmans(num_states, num_actions, params:list, n, size):
    p1, w1, d1, p2, w2, d2, p3, w3, d3, pc, wc, dc, s, r, t = params
    p1 = get_ps(n, p1, size)
    p2 = get_ps(n, p2, size)
    p3 = get_ps(n, p3, size)
    pc = get_ps(n, pc, size)

    P = get_PS(num_states, num_actions, p1, p2, p3, pc)
    R = get_RS(num_states, num_actions, p1, w1, d1, p2, w2, d2, p3, w3, d3, pc, wc, dc, s, r, t)

    V, best_policy = value_iterations(num_states, num_actions, P, -R, gamma=0.99)
    V, best_policy = value_iterations_1(num_states, num_actions, P, -R, gamma=0.99)

    print("状态价值函数：", V)
    print("最优策略：", best_policy)
    return best_policy

if __name__ == '__main__':
    params = [[0.1,4,2,0.1,18,3,0.1,6,3,56,6,5],   
         [0.2,4,2,0.2,18,3,0.2,6,3,56,6,5],
         [0.1,4,2,0.1,18,3,0.1,6,3,56,30,5],
         [0.2,4,1,0.2,18,1,0.2,6,2,56,30,5],
         [0.1,4,8,0.2,18,1,0.1,6,2,56,10,5],
        [0.05,4,2,0.05,18,3,0.05,6,3,56,10,40]]
    # for param in params:
    #     # best_policy = bellmans(19, 8, param, 1475, 10)
    #     best_policy = bellmans(19, 8, param, 2436, 10)
    #     bellman(19, 8, param)
    param1 = [0.1, 2, 1, 0.1, 8, 1, 0.1, 12, 2, 0.1, 8, 4, 46, 500, 6]
    param2 = [0, 46, 4, 0, 46, 4, 0, 42, 4, 0.1, 8, 6, 200, 40, 10]
    best_policy = bellmans(67, 10, param2, 3458, 100)
    # bellman(67, 10, param1)
    solve_c(param2, best_policy)


