import numpy as np
# import mdptoolbox
from Problem2_Bellman import value_iteration, policy_evaluation

def fill_P(matrix):
    for num in range(0, matrix.shape[0]):
        P = matrix[num]
        for index in range(0, P.shape[0]):
            row = P[index]
            if np.all(row == 0):
                matrix[num][index][index] = 1
    return matrix

def get_P(num_states, num_actions, p1, p2, p3, p1_h, p2_h):
    P = np.zeros((num_actions, num_states, num_states))

    '''买1'''
    P[0][0][1] = P[0][2][4] = P[0][5][8] = P[0][13][15] = 1

    '''买2'''
    P[1][0][2] = P[1][1][4] = P[1][3][6] = P[1][12][14] = 1

    '''检1'''
    P[2][1][0] = P[2][4][2] = P[2][8][5] = P[2][15][13] = p1
    P[2][1][3] = P[2][4][6] = P[2][8][7] = P[2][15][9] = 1-p1
    P[2][10][13] = P[2][11][5] = P[2][12][0] = P[2][14][2] = p1_h
    P[2][10][9]  = P[2][11][7] = P[2][12][3] = P[2][14][6]  = 1-p1_h

    '''检2'''
    P[3][2][0] = P[3][4][1] = P[3][6][3] = P[3][14][12] = p2
    P[3][2][5] = P[3][4][8] = P[3][6][7] = P[3][14][11] = 1 - p2
    P[3][9][3] = P[3][10][12] = P[3][13][0] = P[3][15][1] = p2_h 
    P[3][9][7] = P[3][10][11] = P[3][13][5] = P[3][15][8] = 1-p2_h

    q1 = 1 - p1
    q2 = 1 - p2
    q3 = 1 - p3
    q1_h = 1 - p1_h
    q2_h = 1 - p2_h 

    '''成检'''
    P[4][4][16] = q1*q2*q3
    P[4][4][17] = 1 - q1*q2*q3
    P[4][6][16] = q2 * q3
    P[4][6][17] = 1 - q2*q3
    P[4][7][16] = q3
    P[4][7][17] = 1-q3
    P[4][8][16] = q1*q3
    P[4][8][17] = 1 - q1*q3
    P[4][9][16] = q2_h*q3
    P[4][9][17] = 1 - q2_h*q3
    P[4][10][16] = q1_h*q2_h*q3
    P[4][10][17] = 1 - q1_h*q2_h*q3
    P[4][11][16] = q1_h*q3
    P[4][11][17] = 1 - q1_h*q3
    P[4][14][16] = q1_h*q2*q3
    P[4][14][17] = 1 - q1_h*q2*q3
    P[4][15][16] = q1*q2_h*q3
    P[4][15][17] = 1 - q1*q2_h*q3

    '''拆'''
    P[5][17][10] = 1

    '''不拆'''
    P[6][17][0] = 1

    '''合格'''
    P[7][16][18] = 1

    P = fill_P(P)
    return P

def get_c(d3, w3, r, p):
    return min(d3, r * p) + w3

def get_R(num_states, num_actions, p1, w1, d1, p2, w2, d2, p3, w3, d3, s, r, t, p1_h, p2_h):
    R = np.zeros((num_states, num_actions))

    p1 = 1-p1
    p2 = 1-p2
    p3 = 1-p3
    p1_h = 1-p1_h
    p2_h = 1-p2_h

    R[0][0] = w1
    R[0][1] = w2 

    R[1][1] = w2 
    R[1][2] = d1

    R[2][0] = w1
    R[2][3] = d2 

    R[3][1] = w2 

    R[4][2] = d1
    R[4][3] = d2
    R[4][4] = get_c(d3, w3, r, 1-p1*p2*p3)
    R[5][0] = w1 
    
    R[6][3] = d2
    R[6][4] = get_c(d3, w3, r, 1-p2*p3)

    R[7][4] = get_c(d3, w3, r, 1-p3)

    R[8][2] = d1 
    R[8][4] = get_c(d3, w3, r, 1-p1*p3)

    R[9][3] = d2 
    R[9][4] = get_c(d3, w3, r, 1-p2_h*p3)

    R[10][2] = d1 
    R[10][3] = d2 
    R[10][4] = get_c(d3, w3, r, 1-p1_h*p2_h*p3)

    R[11][2] = d1
    R[11][4] = get_c(d3, w3, r, 1-p1_h*p3)

    R[12][1] = w2 
    R[12][2] = d1

    R[13][0] = w1
    R[13][3] = d2 

    R[14][2] = d1 
    R[14][3] = d2 
    R[14][4] = get_c(d3, w3, r, 1 - p1_h*p2*p3)

    R[15][2] = d1 
    R[15][3] = d2 
    R[15][4] = get_c(d3, w3, r, 1 - p1*p2_h*p3)

    R[16][7] = -s

    R[17][5] = t
    

    return R

def check_P(matrix):
    for num in range(0, matrix.shape[0]):
        P = matrix[num]
        for index in range(0, P.shape[0]):
            row = P[index]
            if sum(row) != 1:
                print(num, index)

def bellman(num_states, num_actions, params:list):
    p1, w1, d1, p2, w2, d2, p3, w3, d3, s, r, t = params


    p1_h = p1 / (1 - (1-p1) * (1-p2) * (1-p3))
    p2_h = p2 / (1 - (1-p1) * (1-p2) * (1-p3))
    

    P = get_P(num_states, num_actions, p1, p2, p3, p1_h, p2_h)
    R = get_R(num_states, num_actions, p1, w1, d1, p2, w2, d2, p3, w3, d3, s, r, t, p1_h, p2_h)
    # print(P.shape, R.shape)

    # print(policy_evaluation(num_states, num_actions, P.transpose((1, 0, 2)), -R, 0.99, [0, 1, 0, 1, 4, 0, 4, 4, 4, 4, 4, 4, 1, 0, 4, 4, 7, 5, 0]))

      # check_P(P)
    # vi = mdptoolbox.mdp.ValueIteration(P, -R, 1)
    # vi.run()

    # optimal_policy = vi.policy
    # print("最优策略:", optimal_policy)

    # # 获取每个状态的预期收益（值函数）
    # state_values = vi.V
    # print("每个状态的预期收益:", state_values)

    V, best_policy = value_iteration(num_states, num_actions, P.transpose((1, 0, 2)), -R, gamma=0.99)

    print("状态价值函数：", V)
    print("最优策略：", best_policy)
    return best_policy

def solve_c(param, best_policy):
    strategy = []
    p1, w1, d1, p2, w2, d2, p3, w3, d3, s, r, t = param
    p1_h = p1 / (1 - (1-p1) * (1-p2) * (1-p3))
    p2_h = p2 / (1 - (1-p1) * (1-p2) * (1-p3))

    # x1 = best_policy[1] == 2
    # x2 = best_policy[2] == 3
    x1 = 2 in best_policy[[1,4,8]]
    x2 = 3 in best_policy[[2,4,6]]
    if x1 and x2:
        p = p3
    elif x1 and not x2:
        p = 1 - (1-p2)*(1-p3)
    elif x2 and not x1:
        p = 1 - (1-p1)*(1-p3)
    else:
        p = 1 - (1-p1)*(1-p2)*(1-p3)

    x3 = r * p > d3
    x4 = best_policy[-2] == 5
    # x5 = best_policy[11] == 2
    # x6 = best_policy[9] == 3
    x5 = 2 in best_policy[[10,11,12,14]]
    x6 = 3 in best_policy[[9,10,13,15]]

    if x5 and x6:
        p = p3
    elif x5 and not x6:
        p = 1 - (1-p2_h)*(1-p3)
    elif x5 and not x6:
        p = 1 - (1-p1_h)*(1-p3)
    else:
        p = 1 - (1-p1_h)*(1-p2_h)*(1-p3)
    
    x7 = r * p > d3
    x8 = x4
    
    if x1:
        # print('先验 检测1')
        strategy.append(1)
    else:
        # print('先验 不检测1')
        strategy.append(0)
    if x2:
        # print('先验 检测2')
        strategy.append(1)
    else:
        # print('先验 不检测2')
        strategy.append(0)
    if x3:
        # print('先验 成品检测')
        strategy.append(1)
    else:
        # print('先验 不成品检测')
        strategy.append(0)
    if x4:
        # print('先验 拆')
        strategy.append(1)
    else:
        # print('先验 不拆')
        strategy.append(0)
    if x5:
        # print('后验 检测1')
        strategy.append(1)
    else:
        # print('后验 不检测1')
        strategy.append(0)
    if x6:
        # print('后验 检测2')
        strategy.append(1)
    else:
        # print('后验 不检测2')
        strategy.append(0)
    if x7:
        # print('后验 成品检测')
        strategy.append(1)
    else:
        # print('后验 不成品检测')
        strategy.append(0)
    if x8:
        # print('后验 拆')
        strategy.append(1)
    else:
        # print('后验 不拆')
        strategy.append(0)
    print(strategy)

if __name__=='__main__':
    params = [[0.1,4,2,0.1,18,3,0.1,6,3,56,6,5],   
         [0.2,4,2,0.2,18,3,0.2,6,3,56,6,5],
         [0.1,4,2,0.1,18,3,0.1,6,3,56,30,5],
         [0.2,4,1,0.2,18,1,0.2,6,2,56,30,5],
         [0.1,4,8,0.2,18,1,0.1,6,2,56,10,5],
        [0.05,4,2,0.05,18,3,0.05,6,3,56,10,40]]
    for param in params:
        best_policy = bellman(19, 8, param)
        solve_c(param, best_policy)

    # param_h = [0.1,8,1,0.1,12,2,0.1,8,4,42,500,6]
    # best_policy = bellman(19, 8, param_h)
    # solve_c(param_h, best_policy)