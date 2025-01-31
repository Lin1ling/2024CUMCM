import numpy as np
# import mdptoolbox
from Problem2_Bellman import *

def fill_P(matrix):
    for num in range(0, matrix.shape[0]):
        P = matrix[num]
        for index in range(0, P.shape[0]):
            row = P[index]
            if np.all(row == 0):
                matrix[num][index][index] = 1
    return matrix

def get_P(num_states, num_actions, p1, p2, p3, pc, p1_h, p2_h, p3_h):
    P = np.zeros((num_actions, num_states, num_states))

    '''买1'''
    for i in range(16):
        P[0][i][i+16] = 1
        

    '''买2'''
    for i in range(4):
        P[1][i][i+4] = 1
    for i in range(16, 20):
        P[1][i][i+4] = 1
    for i in range(32, 36):
        P[1][i][i+4] = 1
    for i in range(48, 52):
        P[1][i][i+4] = 1
    

    '''买3'''
    for i in range(16):
        P[2][i*4][i*4+1] = 1


    '''检1'''
    for i in range(16, 32):
        P[3][i][i-16] = p1
        P[3][i+32][i-16] = p1_h
        P[3][i][i+16] = 1-p1
        P[3][i+32][i+16] = 1-p1_h
    

    '''检2'''
    for i in range(4, 8):
        P[4][i][i-4] = p2
        P[4][i+8][i-4] = p2_h
        P[4][i][i+4] = 1 - p2
        P[4][i+8][i+4] = 1 - p2_h
    for i in range(20, 24):
        P[4][i][i-20] = p2
        P[4][i+8][i-20] = p2_h
        P[4][i][i+4] = 1 - p2
        P[4][i+8][i+4] = 1 - p2_h
    for i in range(36, 40):
        P[4][i][i-36] = p2
        P[4][i+8][i-36] = p2_h
        P[4][i][i+4] = 1 - p2
        P[4][i+8][i+4] = 1 - p2_h
    for i in range(52, 56):
        P[4][i][i-52] = p2
        P[4][i+8][i-52] = p2_h
        P[4][i][i+4] = 1 - p2
        P[4][i+8][i+4] = 1 - p2_h


    '''检3'''
    for i in range(16):
        P[5][i*4+1][i*4] = p3
        P[5][i*4+3][i*4] = p3_h
        P[5][i*4+1][i*4+2] = 1 - p3
        P[5][i*4+3][i*4+2] = 1 - p3_h


    q1 = 1 - p1
    q2 = 1 - p2
    q3 = 1 - p3
    qc = 1 - pc
    q1_h = 1 - p1_h
    q2_h = 1 - p2_h 
    q3_h = 1 - p3_h 

    '''成检'''
    P[6][21][64] = q1*q2*q3*qc
    P[6][21][65] = 1 - q1*q2*q3*qc
    P[6][22][64] = q1*q2*qc
    P[6][22][65] = 1 - q1*q2*qc
    P[6][23][64] = q1*q2*q3_h*qc
    P[6][23][65] = 1 - q1*q2*q3_h*qc

    P[6][25][64] = q1*q3*qc
    P[6][25][65] = 1 - q1*q3*qc
    P[6][26][64] = q1*qc
    P[6][26][65] = 1 - q1*qc
    P[6][27][64] = q1*q3_h*qc
    P[6][27][65] = 1 - q1*q3_h*qc

    P[6][29][64] = q1*q2_h*q3*qc
    P[6][29][65] = 1 - q1*q2_h*q3*qc
    P[6][30][64] = q1*q2_h*qc
    P[6][30][65] = 1 - q1*q2_h*qc
    P[6][31][64] = q1*q2_h*q3_h*qc
    P[6][31][65] = 1 - q1*q2_h*q3_h*qc
    # ------------
    P[6][37][64] = q2*q3*qc
    P[6][37][65] = 1 - q2*q3*qc
    P[6][38][64] = q2*qc
    P[6][38][65] = 1 - q2*qc
    P[6][39][64] = q2*q3_h*qc
    P[6][39][65] = 1 - q2*q3_h*qc

    P[6][41][64] = q3*qc
    P[6][41][65] = 1 - q3*qc
    P[6][42][64] = qc
    P[6][42][65] = 1 - qc
    P[6][43][64] = q3_h*qc
    P[6][43][65] = 1 - q3_h*qc

    P[6][45][64] = q2_h*q3*qc
    P[6][45][65] = 1 - q2_h*q3*qc
    P[6][46][64] = q2_h*qc
    P[6][46][65] = 1 - q2_h*qc
    P[6][47][64] = q2_h*q3_h*qc
    P[6][47][65] = 1 - q2_h*q3_h*qc
    # ------------
    P[6][53][64] = q1_h*q2*q3*qc
    P[6][53][65] = 1 - q1_h*q2*q3*qc
    P[6][54][64] = q1_h*q2*qc
    P[6][54][65] = 1 - q1_h*q2*qc
    P[6][55][64] = q1_h*q2*q3_h*qc
    P[6][55][65] = 1 - q1_h*q2*q3_h*qc

    P[6][57][64] = q1_h*q3*qc
    P[6][57][65] = 1 - q1_h*q3*qc
    P[6][58][64] = q1_h*qc
    P[6][58][65] = 1 - q1_h*qc
    P[6][59][64] = q1_h*q3_h*qc
    P[6][59][65] = 1 - q1_h*q3_h*qc

    P[6][61][64] = q1_h*q2_h*q3*qc
    P[6][61][65] = 1 - q1_h*q2_h*q3*qc
    P[6][62][64] = q1_h*q2_h*qc
    P[6][62][65] = 1 - q1_h*q2_h*qc
    P[6][63][64] = q1_h*q2_h*q3_h*qc
    P[6][63][65] = 1 - q1_h*q2_h*q3_h*qc

    '''拆'''
    P[7][65][63] = 1

    '''不拆'''
    P[8][65][0] = 1

    '''结算'''
    P[9][64][66] = 1

    P = fill_P(P)
    return P

def get_c(dc, wc, r, p):
    return min(dc, r * p) + wc

def get_R(num_states, num_actions, p1, w1, d1, p2, w2, d2, p3, w3, d3, pc, wc, dc, s, r, t, p1_h, p2_h, p3_h):
    R = np.zeros((num_states, num_actions))

    q1 = 1-p1
    q2 = 1-p2
    q3 = 1-p3
    qc = 1-pc
    q1_h = 1-p1_h
    q2_h = 1-p2_h
    q3_h = 1-p3_h

    for i in range(16):
        R[i][0] = w1
    
    for i in range(4):
        R[16*i][1] = w2
        R[16*i+1][1] = w2
        R[16*i+2][1] = w2
        R[16*i+3][1] = w2

    for i in range(16):
        R[4*i][2] = w3

    for i in range(16):
        R[16+i][3] = d1
        R[48+i][3] = d1

    for i in range(8):
        R[8*i+4][4] = d2
        R[8*i+5][4] = d2
        R[8*i+6][4] = d2
        R[8*i+7][4] = d2

    for i in range(32):
        R[2*i+1][5] = d3

    # for i in range(21, 64):
    #     if i % 4 == 0:
    #         continue
    #     R[i][6] = get_c(dc, wc, r, 1-p1*p2*p3*pc)
    R[21][6] = get_c(dc, wc, r, 1 - q1*q2*q3*qc)
    R[22][6] = get_c(dc, wc, r, 1 - q1*q2*qc)
    R[23][6] = get_c(dc, wc, r, 1 - q1*q2*q3_h*qc)
    R[25][6] = get_c(dc, wc, r, 1 - q1*q3*qc)
    R[26][6] = get_c(dc, wc, r, 1 - q1*qc)
    R[27][6] = get_c(dc, wc, r, 1 - q1*q3_h*qc)
    R[29][6] = get_c(dc, wc, r, 1 - q1*q2_h*q3*qc)
    R[30][6] = get_c(dc, wc, r, 1 - q1*q2_h*qc)
    R[31][6] = get_c(dc, wc, r, 1 - q1*q2_h*q3_h*qc)
    R[37][6] = get_c(dc, wc, r, 1 - q2*q3*qc)
    R[38][6] = get_c(dc, wc, r, 1 - q2*qc)
    R[39][6] = get_c(dc, wc, r, 1 - q2*q3_h*qc)
    R[41][6] = get_c(dc, wc, r, 1 - q3*qc)
    R[42][6] = get_c(dc, wc, r, 1 - qc)
    R[43][6] = get_c(dc, wc, r, 1 - q3_h*qc)
    R[45][6] = get_c(dc, wc, r, 1 - q2_h*q3*qc)
    R[46][6] = get_c(dc, wc, r, 1 - q2_h*qc)
    R[47][6] = get_c(dc, wc, r, 1 - q2_h*q3_h*qc)
    R[53][6] = get_c(dc, wc, r, 1 - q1_h*q2*q3*qc)
    R[54][6] = get_c(dc, wc, r, 1 - q1_h*q2*qc)
    R[55][6] = get_c(dc, wc, r, 1 - q1_h*q2*q3_h*qc)
    R[57][6] = get_c(dc, wc, r, 1 - q1_h*q3*qc)
    R[58][6] = get_c(dc, wc, r, 1 - q1_h*qc)
    R[59][6] = get_c(dc, wc, r, 1 - q1_h*q3_h*qc)
    R[61][6] = get_c(dc, wc, r, 1 - q1_h*q2_h*q3*qc)
    R[62][6] = get_c(dc, wc, r, 1 - q1_h*q2_h*qc)
    R[63][6] = get_c(dc, wc, r, 1 - q1_h*q2_h*q3_h*qc)
    

    R[64][9] = -s
    R[65][7] = t
    

    return R

def check_P(matrix):
    for num in range(0, matrix.shape[0]):
        P = matrix[num]
        for index in range(0, P.shape[0]):
            row = P[index]
            if sum(row) != 1:
                print(num, index)

def bellman(num_states, num_actions, params:list):
    p1, w1, d1, p2, w2, d2, p3, w3, d3, pc, wc, dc, s, r, t = params

    p1_h = p1 / (1 - (1-p1) * (1-p2) * (1-p3) * (1-pc))
    p2_h = p2 / (1 - (1-p1) * (1-p2) * (1-p3) * (1-pc))
    p3_h = p3 / (1 - (1-p1) * (1-p2) * (1-p3) * (1-pc))

    P = get_P(num_states, num_actions, p1, p2, p3, pc, p1_h, p2_h, p3_h)
    R = get_R(num_states, num_actions, p1, w1, d1, p2, w2, d2, p3, w3, d3, pc, wc, dc, s, r, t, p1_h, p2_h, p3_h)

    x = [0] * 3
    x[0] = 1 if p1==0 else 0
    x[1] = 1 if p2==0 else 0
    x[2] = 1 if p3==0 else 0
    # print(P.shape, R.shape)

    # print(policy_evaluation(num_states, num_actions, P.transpose((1, 0, 2)), -R, 0.99, [0, 1, 0, 1, 4, 0, 4, 4, 4, 4, 4, 4, 1, 0, 4, 4, 7, 5, 0]))

    # vi = mdptoolbox.mdp.ValueIteration(P, -R, 0.99)
    # vi.run()
    # optimal_policy = vi.policy
    # print("最优策略:", optimal_policy)
    # # 获取每个状态的预期收益（值函数）
    # state_values = vi.V
    # print("每个状态的预期收益:", state_values)
    # return optimal_policy

    # --------
    # V, best_policy = value_iteration(num_states, num_actions, P.transpose((1, 0, 2)), -R, gamma=0.99)
    V, best_policy = value_iteration_1(num_states, num_actions, x, P.transpose((1, 0, 2)), -R, gamma=0.99)
    print("状态价值函数：", V)
    print("最优策略：", best_policy)
    return best_policy

def solve_c(param, best_policy):
    strategy_0 = []
    p1, w1, d1, p2, w2, d2, p3, w3, d3, pc, wc, dc, s, r, t = param
    p1_h = p1 / (1 - (1-p1) * (1-p2) * (1-p3) * (1-pc))
    p2_h = p2 / (1 - (1-p1) * (1-p2) * (1-p3) * (1-pc))
    p3_h = p3 / (1 - (1-p1) * (1-p2) * (1-p3) * (1-pc))
    print(f'次品率：{1 - (1-p1) * (1-p2) * (1-p3) * (1-pc)}')
    x1 = 3 in (best_policy[i]for i in [x for x in range(16,32)])
    x2 = 4 in (best_policy[i]for i in [4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 39, 52, 53, 54, 55])
    x3 = 5 in (best_policy[i]for i in [x for x in range(1, 64, 4)])
    # x4 = True # 是否检测成品

    x6 = 3 in (best_policy[i]for i in [x for x in range(48,64)])
    x7 = 4 in (best_policy[i]for i in [12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 60, 61, 62, 63])
    x8 = 5 in (best_policy[i]for i in [x for x in range(3, 64, 4)])

    p1_r = 1 if x1 else p1
    p2_r = 1 if x2 else p2
    p3_r = 1 if x3 else p3

    
    p6_r = 1 if x6 else p1_h
    p7_r = 1 if x7 else p2_h
    p8_r = 1 if x8 else p3_h


    p = 1 - (1-p1_r)*(1-p2_r)*(1-p3_r)*(1-pc)

    x4 = r * p > dc
    x5 = True   # 拆
    p = 1 - (1-p6_r)*(1-p7_r)*(1-p3_r)*(1-pc)
    
    x9 = r * p > dc
    x10 = x5
    
    strategy_0.append(x1)
    strategy_0.append(x2)
    strategy_0.append(x3)
    strategy_0.append(x4)
    strategy_0.append(x5)
    strategy_0.append(x6)
    strategy_0.append(x7)
    strategy_0.append(x8)
    strategy_0.append(x9)
    strategy_0.append(x10)
    strategy = [1 if x else 0 for x in strategy_0]
    
    print(strategy)

if __name__=='__main__':
    
    # params = [[0.1,4,2,0.1,18,3,0.1,6,3,56,6,5],   
    #      [0.2,4,2,0.2,18,3,0.2,6,3,56,6,5],
    #      [0.1,4,2,0.1,18,3,0.1,6,3,56,30,5],
    #      [0.2,4,1,0.2,18,1,0.2,6,2,56,30,5],
    #      [0.1,4,8,0.2,18,1,0.1,6,2,56,10,5],
    #     [0.05,4,2,0.05,18,3,0.05,6,3,56,10,40]]
    # for param in params:
    #     best_policy = bellman(67, 10, param)
    #     solve_c(param, best_policy)

    # p1, w1, d1, p2, w2, d2, p3, w3, d3, pc, wc, dc, s, r, t = params
    param1 = [0.1, 2, 1, 0.1, 8, 1, 0.1, 12, 2, 0.1, 8, 4, 46, 500, 6]
    param2 = [0, 46, 4, 0, 46, 4, 0, 42, 4, 0.1, 8, 6, 200, 40, 10]
    best_policy = bellman(67, 10, param1)
    solve_c(param2, best_policy)


        