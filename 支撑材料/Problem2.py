import itertools

def pass_rate(p, x):
    return (1-p)**(1-x)

def total_reject_rate(q1, q2, p3):
    return (1-q1*q2)+q1*q2*p3

def strategy_cost(d1, x1, d2, x2, d3, q3, r, x3, x4=0):
    if x4==0:
        return x1*d1+x2*d2
    return x1*d1+x2*d2+x3*d3+x4*r
    # dcost = min(q3*r, d3)
    # if dcost==d3:
    #     x3, x4 = 1, 0
    # elif dcost==q3*r:
    #     x3, x4 = 0, 1
    # return x3, x4, x1*d1+x2*d2+dcost

def if_disassemble(p1, p2, pf, w1, w2, t, x1, x2):
    pp1 = p1 / pf * (1-x1)  # 零配件1的后验次品率
    pp2 = p2 / pf * (1-x2)
    w1_real = w1 / (1-p1)
    w2_real = w2 / (1-p2)
    if t >= (1-pp1)*w1_real + (1-pp2)*w2_real:
        return pp1, pp2, w1_real, w2_real, 0
    else:
        return pp1, pp2, w1_real, w2_real, 1

def cal_benchmark(p1, w1, d1, p2, w2, d2, p3, w3, d3, s, r, t, x1=0, x2=0, x3=0, x4=0):
    # q1 = pass_rate(p1, x1)
    # q2 = pass_rate(p2, x2)
    # pf = total_reject_rate(q1, q2, p3)
    # total_pass_rate = pass_rate(pf, x3)
    # benchmark = total_pass_rate*(s+r)
    # print(total_pass_rate)
    # print(benchmark)
    total_pass_rate = (1-p1)*(1-p2)*(1-p3)
    # print(f'total_pass_rate:{total_pass_rate} s:{s} r:{r} w1:{w1} w2:{w2} d3:{d3}')
    benchmark = total_pass_rate*s-(1-total_pass_rate)*r-w1-w2-d3    # 000的预期收益
    # print(f'total_pass_rate:{total_pass_rate} s:{s} r:{r} w1:{w1} w2:{w2} d3:{d3} benchmark:{benchmark}')

    return benchmark

def round2_x3(situa, strategy_2, benchmark_2, pf_1, strategies):
    pf_21, drate, dprofit_20 = func(*(situa+strategy_2+[0]),benchmark=benchmark_2, scheme=2, pf_1=pf_1, x3_p=strategies[2])    # 收益
    pf_22, drate, dprofit_21 = func(*(situa+strategy_2+[1]),benchmark=benchmark_2, scheme=2, pf_1=pf_1, x3_p=strategies[2])    # 收益
    if dprofit_20 > dprofit_21:
        dprofit_2 = dprofit_20
        pf_2 = pf_21
        strategy_2=0
    else:
        pf_2 = pf_22
        dprofit_2 = dprofit_21
        strategy_2=1
    return pf_2, dprofit_2, strategy_2

def func(p1, w1, d1, p2, w2, d2, p3, w3, d3, s, r, t, x1=0, x2=0, x3=0, x4=0, benchmark=0, scheme=1, pf_1=0, x3_p=0):
    def all_zero(*args):
        return all(arg == 0 for arg in args)
    # print(f'决策：{x1} {x2} {x3} {x4}')
    if scheme==2:
        x1 = 1
        x2 = 1
    q1 = pass_rate(p1, x1)
    q2 = pass_rate(p2, x2)
    pf = total_reject_rate(q1, q2, p3)
    # print(f'q1:{q1} q2:{q2} pf:{1-pf}')
    total_pass_rate = pass_rate(pf, x3)
    bm_pass = (1-p1)*(1-p2)*(1-p3)
    prime_cost = strategy_cost(d1, x1, d2, x2, d3, r, total_pass_rate, x3, x4)
    if all_zero(x1,x2,x3,x4):
        benchmark = total_pass_rate
        # profit = total_pass_rate*s-(1-q)*r-w1-w2-d3    # 000的预期收益
    if scheme==1:
        profit = (1-pf)*s-x3*d3-(1-x3)*pf*r-(w1+w2+w3)-prime_cost
        # profit = (total_pass_rate-bm_pass)*(s+r)-prime_cost
        # print(f'x1 x2 x3 x4={x1} {x2} {x3} {x4}')
        # print(f'total_pass_rate:{round(total_pass_rate, 3)}  bm_pass:{round(bm_pass, 3)}  prime_cost:{prime_cost}  profit:{round(profit, 3)}')
        # print(round(profit, 3))

        return pf, total_pass_rate, profit, prime_cost
    elif scheme==2:
        p = pf_1 + (1-pf_1)*total_pass_rate      # 两轮内合格，pf_1是第一次的通过率
        # profit = (p-benchmark)*s-(1-p)*r-prime_cost
        # profit = (p-benchmark)*(s+r)-prime_cost-x3_p*d3
        # print(x3_p)-x3_p*d3
        profit = (pf_1-benchmark)*(s+r)+x4*(1-pf_1)*p*(s+r-6)-prime_cost-w1-w2
        return p, total_pass_rate, profit

def scheme_1(situa, strategies, benchmark):
    strategy_0 = [0,0,0,0]
    for i in range(len(situa)):
        # if i!=0:
        #     break
        comp = []
        comp_dr = []
        print(f'第{i+1}种情形：',end='')
        # print(f'第{i+1}种情形：')
        # a, bm, profit, b = func(*(situa[i]+strategy_0))
        for j in strategies:
            # pf, drate, dprofit, cost = func(*(situa[i]+j),benchmark=benchmark)
            pf, drate, dprofit, cost = func(*(situa[i]+j))
            # dprofit += benchmark[i]
            comp.append(round(dprofit, 3))
            comp_dr.append(round(drate, 3))
            ## print(f'合格率：{drate:.3f}，相对收益：{dprofit:.3f}')
        max_index = comp.index(max(comp))

        # expect = situa[i][9]-situa[i][8]-situa[i][1]-situa[i][4]
        print(strategies[max_index],end='')
        print(round(max(comp), 3))
        # print(comp_dr[max_index])
    return strategies[max_index]

def scheme_2(situa, strategies, benchmark):
    strategy_01 = [0,0,0,0]
    strategy_02 = [0,0,0,1]
    for i in range(6):
        # if i != 0:
        #     continue
        comp = []
        strategies = [list(map(int, bin(i)[2:].zfill(3))) for i in range(8)]    # 生成所有可能的四位二进制组合
        print(f'第{i+1}种情形：',end='')
        a, benchmark_1, profit_1, c = func(*(situa[i]+strategy_01))
        # benchmark_1 = a
        # pf_benchmark, benchmark_2, profit_2, c = func(*(situa[i]+strategy_02))
        # benchmark_2, a, b = func(*(situa[i]+[0,0,0]), scheme=2, pf_1=pf_benchmark)
        for j in range(len(strategies)):    # 每种决策
            strategy_2 = strategies[j].copy()
            # 得到第一轮的后验次品率drate
            pf_1, drate, dprofit_1, c = func(*(situa[i]+strategies[j]+[0]),benchmark=benchmark_1, scheme=1)
            # 判断是否需要进行拆解
            # p1_2, p2_2后验次品率，w1_2, w2_2真成本
            p1_2, p2_2, w1_2, w2_2, x4 = if_disassemble(situa[i][0], situa[i][3], drate, situa[i][1], situa[i][4], situa[i][11], strategies[j][0], strategies[j][1])
            # 将结果x4加入决策
            strategies[j].append(x4)
            if x4 == 0:
                # max_index = comp.index(max(comp))
                # print(strategies[max_index],end='')
                stra = scheme_1([situa[i]], strategies, benchmark)
                stra.append(x4)
                print(stra, end='')
                print("    不进行拆解")
                break
            strategy_2[0], strategy_2[1] = 1-strategies[j][0], 1-strategies[j][1]
            situa[i][0], situa[i][3] = p1_2, p2_2
            # situa[i][1], situa[i][4] = w1_2, w2_2
            
            pf_2, dprofit_2, strategy_2[2] = round2_x3(situa[i], strategy_2[:1], benchmark_1, drate, strategies[j])
            dprofit_2 += benchmark[i]
            strategies[j] += strategy_2
            comp.append(round(dprofit_2, 3))
            
            # print(strategies[j],end='  ')
            # print(round(dprofit_2, 3))

            # print(strategies[j],end='   ')
            # print(round(dprofit_1, 3),end='   ')
            # print(round(dprofit_2, 3))
            # print(round(dprofit_1 + dprofit_2, 3))

        # print(comp)
        # print(strategies)
        if len(comp) > 0:
            max_index = comp.index(max(comp))
            print(strategies[max_index],end=' ')
            print(max(comp))

        # if i==4:
        #     # 第五种情况[0, 1, 1, 1, 1, 0, 1]
        #     pf_1, benchmark_1, profit_1, c = func(*(situa[4]+strategy_01))
        #     pf_21, drate, dprofit_2 = func(*(situa[4]+[1,0]+[1]),benchmark=benchmark_2, scheme=2, pf_1=pf_1, x3_p=1)    # 收益
        #     print(f'第五种情况：{dprofit_2}')
    
def scheme_3():
    pass


situa = [[0.1,4,2,0.1,18,3,0.1,6,3,56,6,5],   
         [0.2,4,2,0.2,18,3,0.2,6,3,56,6,5],
         [0.1,4,2,0.1,18,3,0.1,6,3,56,30,5],
         [0.2,4,1,0.2,18,1,0.2,6,2,56,30,5],
         [0.1,4,8,0.2,18,1,0.1,6,2,56,10,5],
        [0.05,4,2,0.05,18,3,0.05,6,3,56,10,40]]

strategies = [list(map(int, bin(i)[2:].zfill(3))) for i in range(8)]    # 生成所有可能的四位二进制组合


benchmark = []
for i in situa:
    bm = round(cal_benchmark(*i),3)
    benchmark.append(bm)

print(benchmark)
scheme_2(situa, strategies, benchmark)
scheme_1(situa, strategies, benchmark)