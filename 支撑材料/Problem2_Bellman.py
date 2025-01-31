import numpy as np
def decimal_to_base4(n):
    if n == 0:
        return "[0,0,0]"
    base4_digits = []
    while n > 0:
        remainder = n % 4
        base4_digits.append(str(remainder))
        n = n // 4
    # 逆序排列余数
    base4_digits.reverse()
    # 将列表转换为字符串
    return ''.join(base4_digits)

def value_iteration(num_states, num_actions, P, R, gamma, theta=1e-5):
    V = np.zeros(num_states)  # 初始化价值函数 V(s)
    
    while True:
        delta = 0
        for s in range(num_states):
            v = V[s]
            Q_s = np.zeros(num_actions)
            for a in range(num_actions):
                Q_s[a] = sum([P[s, a, s_prime] * (R[s, a] + gamma * V[s_prime]) for s_prime in range(num_states)])
            V[s] = max(Q_s)
            delta = max(delta, abs(v - V[s]))
            
        if delta < theta:
            break
    
    # 策略提取
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        Q_s = np.zeros(num_actions)
        for a in range(num_actions):
            Q_s[a] = sum([P[s, a, s_prime] * (R[s, a] + gamma * V[s_prime]) for s_prime in range(num_states)])
        policy[s] = np.argmax(Q_s)
    
    return V, policy

def value_iteration_1(num_states, num_actions, x:list, P, R, gamma, theta=1e-5):
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
                # Q_s[a] = sum([P[s, a, s_prime] * (R[s, a] + gamma * V[s_prime]) for s_prime in range(num_states)])
                for s_prime in range(num_states):
                    delt = P[s, a, s_prime] * (R[s, a] + gamma * V[s_prime])
                    Q_s[a] += delt
                    # if s==5 and delt!=0:    # 011
                        # print(f'V[s_prime]:{V[s_prime]},s_prime:{decimal_to_base4(s_prime)},p:{P[s, a, s_prime]},r:{R[s, a]},delt:{delt},a:{a},Q_s:{Q_s}')

            V[s] = max(Q_s)
            


            # if x[0]==1 and P[s][3][s]==0: # 半成品1为100%合格品，在这一过程中要作为零件默认为已经过检测  且 无自环，可以进行action3
            #     V[s] = Q_s[3]
            # if x[1]==1 and P[s][4][s]==0: # 半成品2为100%合格品，在这一过程中要作为零件默认为已经过检测
            #     V[s] = Q_s[4]
            # if x[2]==1 and P[s][5][s]==0: # 半成品3为100%合格品，在这一过程中要作为零件默认为已经过检测
            #     V[s] = Q_s[5]

            delta = max(delta, abs(v - V[s]))
            
        if delta < theta:
            break
    
    # 策略提取
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        Q_s = np.zeros(num_actions)
        for a in range(num_actions):
            Q_s[a] = sum([P[s, a, s_prime] * (R[s, a] + gamma * V[s_prime]) for s_prime in range(num_states)])
        policy[s] = np.argmax(Q_s)
        # if x[0]==1 and P[s][3][s]==0: # 半成品1为100%合格品，在这一过程中要作为零件默认为已经过检测  且 无自环，可以进行action3
        #     policy[s] = 3
        # if x[1]==1 and P[s][4][s]==0: # 半成品2为100%合格品，在这一过程中要作为零件默认为已经过检测
        #     policy[s] = 4
        # if x[2]==1 and P[s][5][s]==0: # 半成品3为100%合格品，在这一过程中要作为零件默认为已经过检测
        #     policy[s] = 5   
    
    return V, policy

# def policy_evaluation(num_states, num_actions, P, R, gamma, policy, theta=1e-5):
#     V = np.zeros(num_states)  # 初始化状态值函数 V(s)
    
#     while True:
#         delta = 0
#         for s in range(num_states):
#             v = V[s]
#             a = policy[s]  # 根据策略获取动作
#             V[s] = sum([P[s, a, s_prime] * (R[s, a] + gamma * V[s_prime]) for s_prime in range(0, num_states)])
#             delta = max(delta, abs(v - V[s]))
        
#         if delta < theta:
#             break
    
#     return V
def policy_evaluation(num_states, num_actions, P, R, gamma, policy, theta=1e-5):
    V = np.zeros(num_states)  # 初始化状态值函数 V(s)
    
    while True:
        delta = 0
        for s in range(num_states):
            v = V[s]
            a = policy[s]  # 根据策略获取动作
            
            if a < 0 or a >= num_actions:
                raise ValueError(f"Invalid action {a} in policy for state {s}")
                
            V[s] = sum([P[s, a, s_prime] * (R[s, a] + gamma * V[s_prime]) for s_prime in range(num_states)])
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
    
    return V

# V, best_policy = value_iteration(num_states, num_actions, P, R, gamma)

# print("状态价值函数：", V)
# print("最优策略：", best_policy)