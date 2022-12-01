'''item-based decomposition algorithm'''

from gurobipy import gurobipy as gp
# from gurobipy import GRB
import numpy as np
from numpy import linalg as LA
import time
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
# from multiprocessing.dummy import Pool
# from multiprocessing import Process
# from ray.util.multiprocessing import Pool
import random
# import functools
# import pathos.pools as pp
# from pathos.pp import ParallelPool
# from pathos.multiprocessing import ProcessingPool as Pool
#from pathos.helpers import cpu_count

rho = 10     # initial penalty
# 初始化变量
ite = 0       # ite为迭代次数

# 随机数种子
s = 1
random.seed(s)
np.random.seed(s)
'''
# 随机生成数据
N = 10  # T:time 16; 30
M = 6  # J:plant 6/90; 6
K = 20  # I:item 78/600; 78

# set of transportation arc
cL = []
cLL = np.random.choice(a=[False, True], size=(M, M), p=[0.9, 0.1])
for j1 in range(M):
    for j2 in range(M):
        if j2 != j1 and cLL[j1][j2]:
            cL.append((j1 + 1, j2 + 1))
L = len(cL)

# 生成bom tree
tr_depth = K // 10
tr_index = np.random.choice(a=list(range(tr_depth)), size=(1, K))
tr_index_dic = {}
for d in range(tr_depth):
    tr_index_dic[d] = []
for i in range(K):
    tr_index_dic[tr_index[0][i]].append(i + 1)
cE = []
for d in range(tr_depth):
    for i in tr_index_dic[d]:
        for dd in range(d + 1, tr_depth):
            for ii in tr_index_dic[dd]:
                tem = random.uniform(0, 1)
                if tem > 0.9:
                    cE.append((i, ii))
E = len(cE)

# set of replacement arc
cA = []
cAA = np.random.choice(a=[False, True], size=(K, K), p=[0.9, 0.1])
for i1 in range(1, K):
    for i2 in range(0, i1):
        if cAA[i1][i2]:
            cA.append((i1 + 1, i2 + 1))
            cA.append((i2 + 1, i1 + 1))
A = len(cA)

H = np.random.randint(1, 5, size=(K, M))   # holding cost

P = np.random.randint(200, 500, size=(K, N))   # penalty on unmet demand

D = np.random.randint(10, 50, size=(K, N))     # demand

# epsilonUD=np.ones(N)*0.05
# epsilonUI=0.8

v0 = np.random.randint(5, size=(K, M))    # initial inventory

nu = np.random.randint(1, 3, size=(K, M))    # unit capacity used for production

C = np.random.randint(K * 2 * 10, K * 2 * 30, size=(K, N))    # production capacity

dtUL = np.random.randint(1, 3, size=(K, L))  # delta t^L _il, transportation time

dtUP = np.random.randint(1, 3, size=(K, M))  # delta t^P_ij, production time

Q = np.zeros((K, M, N))     # purchase delivery

q = np.random.randint(1, 5, size=E)    # production consumption relationship on bom tree

# gbv.m = 5 * np.ones(gbv.K)
# gbv.wLB = 1 * np.ones((gbv.K, gbv.M))
# gbv.wUB = 20 * np.ones((gbv.K, gbv.M))
'''

# 小规模数据
N = 2  # T:time
M = 3  # J:plant
K = 3  # I:item
L = 2  # number of transportation arcs
E = 1   # number of production arcs in the bom tree
A = 1  # number of replacement arcs

H = np.array([[2, 1, 3]])
H = np.tile(H, (K, 1))     # holding cost

P = 200 * np.ones((K, N))        # penalty on unmet demand

D = np.array([[5, 7], [10, 8], [6, 4]])     # demand

# gbv.epsilonUD = np.ones(gbv.N) * 0.05
# gbv.epsilonUI = 0.8

v0 = np.array([[1, 1, 1], [2, 2, 2], [1, 1, 1]])   # initial inventory

nu = np.array([[2, 2, 2], [1, 1, 1], [2, 2, 2]])   # unit capacity used for production

C = np.array([[50], [50], [50]])     # production capacity
C = np.tile(C, (1, N))

cL = [(1, 2), (2, 3)]     # set of transportation arcs
cE = [(2, 1)]             # set of production relationship arcs
cA = [(3, 2)]             # set of replacement arcs

dtUL = np.ones((K, L))  # delta t^L _il, transportation time

dtUP = np.zeros((K, M))  # delta t^P_ij, production time

Q = np.zeros((K, M, N))     # purchase delivery

q = 2 * np.ones(E)     # production consumption relationship on bom tree

# 每个item作为集合A中边的出端(i,i')或者入端(i',i)时，相应边在集合A中对应的序号
out_place = []
for i in range(K):
    tem = []
    for a in range(A):
        if cA[a][0] == i + 1:
            tem.append(a)
    out_place.append(tem)

in_place = []
for i in range(K):
    tem = []
    for a in range(A):
        if cA[a][1] == i + 1:
            tem.append(a)
    in_place.append(tem)

# 为每个item首先建立一个model存起来
item_model = [0] * K  # 存储item的model
item_var = [0] * K  # 存储item的变量

# variables
X = np.zeros((K, M, N))   # production
S = np.zeros((K, L, N))   # transportation
Z = np.zeros((K, M, N))    # supply
R = np.zeros((A, M, N))    # replacement
V = np.zeros((K, M, N))    # inventory
U = np.zeros((K, N))       # unmet demand
YUI = np.zeros((K, M, N))     # inbound quantity
YUO = np.zeros((K, M, N))     # outbound quantity

# 矩阵x的扩张，x_{i(i')jt}=x_{ijt} for i'
xx = np.zeros((K, K, M, N))

# Copying variables
XC = np.zeros((K, K, M, N))  # x_{i(i')jt}
RC1 = np.zeros((A, M, N))  # r_{a(i)jt}
RC2 = np.zeros((A, M, N))  # r_{a(i')jt}

# Dual variables
Mu = np.zeros((K, K, M, N))
Ksi = np.zeros((A, M, N))
Eta = np.zeros((A, M, N))

# 迭代过程中primal residual, dual residual
pri_re = []
d_re = []

MaxIte = 2000    # 最大迭代次数
# parameters for computing primal/dual tolerance
ep_abs = 1e-2
ep_rel = 1e-4

# 初始化primal/dual residual及primal/dual tolerance
pr = 10     # primal residual
dr = 10     # dual residual
p_tol = 1e-5     # primal tolerance
d_tol = 1e-5      # dual tolerance

# p,n for the computation of primal/dual tolerance
sqrt_dim_p = math.sqrt(M * N * (K ** 2 + A * 2))
sqrt_dim_n = sqrt_dim_p

class item_dim():
    def __int__(self):
        self.key = "setup"

    def model_var_setup(self, i):  # 建立item i的model和variable
        prob = gp.Model("item " + str(i))

        # variable
        ui = prob.addMVar(N)  # u_{it} for t
        si = prob.addMVar((L, N))  # s_{ilt} for l,t
        zi = prob.addMVar((M, N))  # z_{ijt} for j,t
        vi = prob.addMVar((M, N))  # v_{ijt} for j,t
        yUIi = prob.addMVar((M, N))  # y^{I}_{ijt} for j,t
        yUOi = prob.addMVar((M, N))  # y^{O}_{ijt} for j,t
        xCi = prob.addMVar((K, M, N))  # x_{i'j(i)t} for i',j,t
        if len(out_place[i]) > 0:
            rC1i = prob.addMVar((len(out_place[i]), M, N))  # r_{a(i)jt} for a=(i,i')
        if len(in_place[i]) > 0:
            rC2i = prob.addMVar((len(in_place[i]), M, N))  # r_{a(i)jt} for a=(i',i)
        # xUbi = prob.addMVar((gbv.M, gbv.N), vtype=GRB.BINARY)   # x^{b}_{ijt} for j,t
        # wi = prob.addMVar((gbv.M, gbv.N), vtype=GRB.INTEGER)   # w_{ijt} for j,t

        # Constraint
        # unmet demand transition
        prob.addConstr(ui[0] + gp.quicksum(zi[j][0] for j in range(M)) == D[i][0], name='3b+3c1')
        prob.addConstrs(
            (ui[t] - ui[t - 1] + gp.quicksum(zi[j][t] for j in range(M)) == D[i][t] for t in
             range(1, N)), name='3c2')

        # inventory transition
        prob.addConstrs((vi[j][0] - yUIi[j][0] + yUOi[j][0] == v0[i][j] for j in range(M)),
                        name='3e1')
        prob.addConstrs(
            (vi[j][t] - vi[j][t - 1] - yUIi[j][t] + yUOi[j][t] == 0 for j in range(M) for t in
             range(1, N)), name='3e2')

        # inbound total
        prob.addConstrs(
            (yUIi[j][t] - gp.quicksum(
                si[ll][t - int(dtUL[i][ll])] for ll in range(L) if
                cL[ll][1] == j + 1 and t - dtUL[i][ll] >= 0) -
             xCi[i][j][
                 t - int(dtUP[i][j])] == Q[i][j][t] for j in range(M) for t in range(N) if
             t >= dtUP[i][j]),
            name='3g1')
        prob.addConstrs(
            (yUIi[j][t] - gp.quicksum(
                si[ll][t - dtUL[i][ll]] for ll in range(L) if
                cL[ll][1] == j + 1 and t - dtUL[i][ll] >= 0) ==
             Q[i][j][t] for j in
             range(M) for t in range(N) if t < dtUP[i][j]),
            name='3g2')

        # outbound total
        if len(out_place[i]) > 0:
            if len(in_place[i]) > 0:
                prob.addConstrs(
                    (yUOi[j][t] - gp.quicksum(
                        si[ll][t] for ll in range(L) if cL[ll][0] == j + 1) - gp.quicksum(
                        q[e] * xCi[cE[e][1] - 1][j][t] for e in range(E) if cE[e][0] == i + 1) -
                     zi[j][
                         t] - gp.quicksum(
                        rC1i[a][j][t] for a in range(len(out_place[i]))) + gp.quicksum(
                        rC2i[a][j][t] for a in range(len(in_place[i]))) == 0 for j in range(M) for t in
                     range(N)),
                    name='3h')
            elif len(in_place[i]) == 0:
                prob.addConstrs(
                    (yUOi[j][t] - gp.quicksum(
                        si[ll][t] for ll in range(L) if cL[ll][0] == j + 1) - gp.quicksum(
                        q[e] * xCi[cE[e][1] - 1][j][t] for e in range(E) if cE[e][0] == i + 1) -
                     zi[j][
                         t] - gp.quicksum(
                        rC1i[a][j][t] for a in range(len(out_place[i]))) == 0 for j in range(M) for t in
                     range(N)),
                    name='3h')
        elif len(out_place[i]) == 0:
            if len(in_place[i]) > 0:
                prob.addConstrs(
                    (yUOi[j][t] - gp.quicksum(
                        si[ll][t] for ll in range(L) if cL[ll][0] == j + 1) - gp.quicksum(
                        q[e] * xCi[cE[e][1] - 1][j][t] for e in range(E) if cE[e][0] == i + 1) -
                     zi[j][
                         t] + gp.quicksum(
                        rC2i[a][j][t] for a in range(len(in_place[i]))) == 0 for j in range(M) for t in
                     range(N)),
                    name='3h')
            elif len(in_place[i]) == 0:
                prob.addConstrs(
                    (yUOi[j][t] - gp.quicksum(
                        si[ll][t] for ll in range(L) if cL[ll][0] == j + 1) - gp.quicksum(
                        q[e] * xCi[cE[e][1] - 1][j][t] for e in range(E) if cE[e][0] == i + 1) -
                     zi[j][
                         t] == 0 for j in range(M) for t in
                     range(N)),
                    name='3h')

        # production capacity
        prob.addConstrs(
            (gp.quicksum(nu[ii][j] * xCi[ii][j][t] for ii in range(K)) <= C[j][t] for j in range(M) for
             t in
             range(N)),
            name='3i')

        # replacement bounds
        if len(out_place[i]) > 0:
            prob.addConstrs(
                (rC1i[a][j][t] >= 0 for j in range(M) for t in range(N) for a in
                 range(len(out_place[i]))),
                name='3j1')
            prob.addConstrs(
                (vi[j][t] - rC1i[a][j][t] >= 0 for j in range(M) for t in range(N) for a in
                 range(len(out_place[i]))),
                name='3j3')
        if len(in_place[i]) > 0:
            prob.addConstrs(
                (rC2i[a][j][t] >= 0 for j in range(M) for t in range(N) for a in
                 range(len(in_place[i]))),
                name='3j2')

        # outbound quantity bounds
        prob.addConstrs((yUOi[j][t] >= 0 for j in range(M) for t in range(N)),
                        name='3k1')
        prob.addConstrs(
            (vi[j][t] - yUOi[j][t] >= 0 for j in range(M) for t in range(N)),
            name='3k2')

        # non-negativity
        prob.addConstrs((yUIi[j][t] >= 0 for j in range(M) for t in range(N)), name='3l')
        prob.addConstrs((vi[j][t] >= 0 for j in range(M) for t in range(N)), name='3l2')
        prob.addConstrs((zi[j][t] >= 0 for j in range(M) for t in range(N)), name='3l3')
        prob.addConstrs((xCi[ii][j][t] >= 0 for ii in range(K) for j in range(M) for t in range(N)),
                        name='3m')
        prob.addConstrs((ui[t] >= 0 for t in range(N)), name='3n')
        prob.addConstrs((si[ll][t] >= 0 for ll in range(L) for t in range(N)), name='3o')

        # 返回item i的gurobi model及变量
        if len(out_place[i]) > 0:
            if len(in_place[i]) > 0:
                return [prob, ui, si, zi, vi, yUIi, yUOi, xCi, rC1i, rC2i]
            elif len(in_place[i]) == 0:
                return [prob, ui, si, zi, vi, yUIi, yUOi, xCi, rC1i]
        elif len(out_place[i]) == 0:
            if len(in_place[i]) > 0:
                return [prob, ui, si, zi, vi, yUIi, yUOi, xCi, rC2i]
            elif len(in_place[i]) == 0:
                return [prob, ui, si, zi, vi, yUIi, yUOi, xCi]

    def model_item_setObj(self, i):    # solve 1st block of ADMM, which can be solved in parallel
        # 求解item i的子问题

        # item i的model和variables
        prob = item_model[i]
        # [ui, si, zi, vi, yUIi, yUOi, xCi, rC1i, rC2i] = self.item_var[i]
        ui = item_var[i][0]
        si = item_var[i][1]
        zi = item_var[i][2]
        vi = item_var[i][3]
        yUIi = item_var[i][4]
        yUOi = item_var[i][5]
        xCi = item_var[i][6]
        if len(out_place[i]) > 0:
            if len(in_place[i]) > 0:
                rC1i = item_var[i][7]
                rC2i = item_var[i][8]
            elif len(in_place[i]) == 0:
                rC1i = item_var[i][7]
        elif len(out_place[i]) == 0:
            if len(in_place[i]) > 0:
                rC2i = item_var[i][7]

        # set objective
        if ite == 0:
            if len(out_place[i]) > 0:
                if len(in_place[i]) > 0:
                    prob.setObjective(gp.quicksum(H[i][j] * vi[j][t] for j in range(M) for t in range(N))
                                      + gp.quicksum(P[i][t] * ui[t] for t in range(N))
                                      - gp.quicksum(
                        Mu[ii][i][j][t] * xCi[ii][j][t] for ii in range(K) for j in
                        range(M) for t in
                        range(N))
                                      - gp.quicksum(
                        Ksi[out_place[i][a]][j][t] * rC1i[a][j][t] for a in range(len(out_place[i])) for
                        j in
                        range(M) for t in range(N))
                                      - gp.quicksum(
                        Eta[in_place[i][a]][j][t] * rC2i[a][j][t] for a in range(len(in_place[i])) for j
                        in
                        range(M) for t in range(N))
                                      )
                elif len(in_place[i]) == 0:
                    prob.setObjective(gp.quicksum(H[i][j] * vi[j][t] for j in range(M) for t in range(N))
                                      + gp.quicksum(P[i][t] * ui[t] for t in range(N))
                                      - gp.quicksum(
                        Mu[ii][i][j][t] * xCi[ii][j][t] for ii in range(K) for j in
                        range(M) for t in
                        range(N))
                                      - gp.quicksum(
                        Ksi[out_place[i][a]][j][t] * rC1i[a][j][t] for a in range(len(out_place[i])) for
                        j in
                        range(M) for t in range(N))
                                      )
            elif len(out_place[i]) == 0:
                if len(in_place[i]) > 0:
                    prob.setObjective(gp.quicksum(H[i][j] * vi[j][t] for j in range(M) for t in range(N))
                                      + gp.quicksum(P[i][t] * ui[t] for t in range(N))
                                      - gp.quicksum(
                        Mu[ii][i][j][t] * xCi[ii][j][t] for ii in range(K) for j in
                        range(M) for t in
                        range(N))
                                      - gp.quicksum(
                        Eta[in_place[i][a]][j][t] * rC2i[a][j][t] for a in range(len(in_place[i])) for j
                        in
                        range(M) for t in range(N))
                                      )
                elif len(in_place[i]) == 0:
                    prob.setObjective(gp.quicksum(H[i][j] * vi[j][t] for j in range(M) for t in range(N))
                                      + gp.quicksum(P[i][t] * ui[t] for t in range(N))
                                      - gp.quicksum(
                        Mu[ii][i][j][t] * xCi[ii][j][t] for ii in range(K) for j in
                        range(M) for t in
                        range(N))
                                      )
        else:
            if len(out_place[i]) > 0:
                if len(in_place[i]) > 0:
                    prob.setObjective(gp.quicksum(H[i][j] * vi[j][t] for j in range(M) for t in range(N))
                                      + gp.quicksum(P[i][t] * ui[t] for t in range(N))
                                      - gp.quicksum(
                        Mu[ii][i][j][t] * xCi[ii][j][t] for ii in range(K) for j in
                        range(M) for t in
                        range(N))
                                      - gp.quicksum(
                        Ksi[out_place[i][a]][j][t] * rC1i[a][j][t] for a in range(len(out_place[i])) for
                        j in
                        range(M) for t in range(N))
                                      - gp.quicksum(
                        Eta[in_place[i][a]][j][t] * rC2i[a][j][t] for a in range(len(in_place[i])) for j
                        in
                        range(M) for t in range(N))
                                      + rho / 2 * gp.quicksum(
                        (X[ii][j][t] - xCi[ii][j][t]) ** 2 for ii in range(K) for j in range(M) for t in
                        range(N))
                                      + rho / 2 * gp.quicksum(
                        (R[out_place[i][a]][j][t] - rC1i[a][j][t]) ** 2 for a in range(len(out_place[i]))
                        for j in
                        range(M) for t in range(N))
                                      + rho / 2 * gp.quicksum(
                        (R[in_place[i][a]][j][t] - rC2i[a][j][t]) ** 2 for a in range(len(in_place[i]))
                        for j in
                        range(M) for t in range(N))
                                      )
                elif len(in_place[i]) == 0:
                    prob.setObjective(gp.quicksum(H[i][j] * vi[j][t] for j in range(M) for t in range(N))
                                      + gp.quicksum(P[i][t] * ui[t] for t in range(N))
                                      - gp.quicksum(
                        Mu[ii][i][j][t] * xCi[ii][j][t] for ii in range(K) for j in
                        range(M) for t in
                        range(N))
                                      - gp.quicksum(
                        Ksi[out_place[i][a]][j][t] * rC1i[a][j][t] for a in range(len(out_place[i])) for
                        j in
                        range(M) for t in range(N))
                                      + rho / 2 * gp.quicksum(
                        (X[ii][j][t] - xCi[ii][j][t]) ** 2 for ii in range(K) for j in range(M) for t in
                        range(N))
                                      + rho / 2 * gp.quicksum(
                        (R[out_place[i][a]][j][t] - rC1i[a][j][t]) ** 2 for a in range(len(out_place[i]))
                        for j in
                        range(M) for t in range(N))
                                      )
            elif len(out_place[i]) == 0:
                if len(in_place[i]) > 0:
                    prob.setObjective(gp.quicksum(H[i][j] * vi[j][t] for j in range(M) for t in range(N))
                                      + gp.quicksum(P[i][t] * ui[t] for t in range(N))
                                      - gp.quicksum(
                        Mu[ii][i][j][t] * xCi[ii][j][t] for ii in range(K) for j in
                        range(M) for t in
                        range(N))
                                      - gp.quicksum(
                        Eta[in_place[i][a]][j][t] * rC2i[a][j][t] for a in range(len(in_place[i])) for j
                        in
                        range(M) for t in range(N))
                                      + rho / 2 * gp.quicksum(
                        (X[ii][j][t] - xCi[ii][j][t]) ** 2 for ii in range(K) for j in range(M) for t in
                        range(N))
                                      + rho / 2 * gp.quicksum(
                        (R[in_place[i][a]][j][t] - rC2i[a][j][t]) ** 2 for a in range(len(in_place[i]))
                        for j in
                        range(M) for t in range(N))
                                      )
                elif len(in_place[i]) == 0:
                    prob.setObjective(gp.quicksum(H[i][j] * vi[j][t] for j in range(M) for t in range(N))
                                      + gp.quicksum(P[i][t] * ui[t] for t in range(N))
                                      - gp.quicksum(
                        Mu[ii][i][j][t] * xCi[ii][j][t] for ii in range(K) for j in
                        range(M) for t in range(N))
                                      + rho / 2 * gp.quicksum(
                        (X[ii][j][t] - xCi[ii][j][t]) ** 2 for ii in range(K) for j in range(M) for t
                        in
                        range(N)))

        prob.optimize()

        # return solutions of item i's local variables
        if len(out_place[i]) > 0:
            if len(in_place[i]) > 0:
                return [ui.X, si.X, zi.X, vi.X, yUIi.X, yUOi.X, xCi.X, rC1i.X, rC2i.X]
            elif len(in_place[i]) == 0:
                return [ui.X, si.X, zi.X, vi.X, yUIi.X, yUOi.X, xCi.X, rC1i.X]
        elif len(out_place[i]) == 0:
            if len(in_place[i]) > 0:
                return [ui.X, si.X, zi.X, vi.X, yUIi.X, yUOi.X, xCi.X, rC2i.X]
            elif len(in_place[i]) == 0:
                return [ui.X, si.X, zi.X, vi.X, yUIi.X, yUOi.X, xCi.X]

    def model_item_global(self):    # solve 2nd block of ADMM, which is to update the global variables
        X_new = np.maximum(0, np.sum(rho * XC - Mu, axis=1)) / (rho * K)

        xx_new = np.zeros((K, K, M, N))

        for i in range(K):
            for j in range(M):
                for t in range(N):
                    xx_new[i, :, j, t] = X_new[i][j][t]

        R_new = np.maximum(0, RC1 + RC2 - (Ksi + Eta) / rho) / 2

        return [X_new, xx_new, R_new]

    def comp_obj(self):    # Compute the objective corresponding to a solution
        VV = np.sum(V, axis=2)
        ob = np.sum(np.multiply(P, U)) + np.sum(np.multiply(H, VV))

        return ob

    def go(self):     # solve items' problems in parallel
        p = mp.Pool(processes=min(mp.cpu_count(), K))
        sc = p.map(self, range(K))
        return sc

    def __call__(self, i):
        if self.key == "setup":
            return self.model_var_setup(i)
        elif self.key == "setObj":
            return self.model_item_setObj(i)

# generate instance of methods of item-based decomposition algorithm
dim = item_dim()

# 为每个item首先建立一个model存起来
lists = list(range(K))
# tem = pool.map(dim.model_var_setup, lists)
#tem = pool.map(dim.model_var_setup, lists)
'''
dim.key = "setup"
tem = dim.go()
for i in lists:
    item_model[i] = tem[i][0]  # tem[i][0]为item i(i=0,...,K-1)的model
    item_var[i] = tem[i][1:]  # tem[i][1:]为item i(i=0,...,K-1)的变量的列表
'''
for i in lists:
    tem = dim.model_var_setup(i)
    item_model[i] = tem[0]  # tem[i][0]为item i(i=0,...,K-1)的model
    item_var[i] = tem[1:]  # tem[i][1:]为item i(i=0,...,K-1)的变量的列表

# 初始化变量
time_start2 = time.time()

dim.key = "setObj"
# x-subproblem
# tem = pool.map(dim.model_item_setObj, lists)
tem = dim.go()
for i in lists:
    U[i] = tem[i][0]
    S[i] = tem[i][1]
    Z[i] = tem[i][2]
    V[i] = tem[i][3]
    YUI[i] = tem[i][4]
    YUO[i] = tem[i][5]
    XC[:, i, :, :] = tem[i][6]
    if len(out_place[i]) > 0:
        if len(in_place[i]) > 0:
            RC1[out_place[i]] = tem[i][7]
            RC2[in_place[i]] = tem[i][8]
        elif len(in_place[i]) == 0:
            RC1[out_place[i]] = tem[i][7]
    elif len(out_place[i]) == 0:
        if len(in_place[i]) > 0:
            RC2[in_place[i]] = tem[i][7]

# z-subproblem
tem = dim.model_item_global()
X = tem[0]
xx = tem[1]
R = tem[2]

# begin ADMM loop
while pr > p_tol or dr > d_tol:    # 当primal/dual residual < tolerance时终止循环
    ite += 1
    if ite > MaxIte:    # 当达到最大迭代次数时终止循环
        break

    # x-subproblem
    # tem = pool.map(dim.model_item_setObj, lists)
    tem = dim.go()
    for i in lists:
        U[i] = tem[i][0]
        S[i] = tem[i][1]
        Z[i] = tem[i][2]
        V[i] = tem[i][3]
        YUI[i] = tem[i][4]
        YUO[i] = tem[i][5]
        XC[:, i, :, :] = tem[i][6]
        if len(out_place[i]) > 0:
            if len(in_place[i]) > 0:
                RC1[out_place[i]] = tem[i][7]
                RC2[in_place[i]] = tem[i][8]
            elif len(in_place[i]) == 0:
                RC1[out_place[i]] = tem[i][7]
        elif len(out_place[i]) == 0:
            if len(in_place[i]) > 0:
                RC2[in_place[i]] = tem[i][7]

    # pool.map(dim.model_item_setObj, lists)

    # 存储z^k
    xp = X
    rp = R

    # z-subproblem
    tem = dim.model_item_global()
    X = tem[0]
    xx = tem[1]
    R = tem[2]

    # Update dual variables
    Mu += 1.6 * rho * (xx - XC)
    Ksi += 1.6 * rho * (R - RC1)
    Eta += 1.6 * rho * (R - RC2)

    xx_p = np.zeros((K, K, M, N))
    for i in range(K):
        for j in range(M):
            for t in range(N):
                xx_p[i, :, j, t] = xp[i][j][t]
    # primal residual
    tem1 = (R - RC1).reshape((-1, 1))
    tem2 = (R - RC2).reshape((-1, 1))
    tem3 = (xx - XC).reshape((-1, 1))
    tem = np.concatenate((tem1, tem2, tem3), axis=0)
    pr = LA.norm(tem)

    # dual residual
    tem1 = (R - rp).reshape((-1, 1))
    tem2 = (xx - xx_p).reshape((-1, 1))
    tem = np.concatenate((tem1, tem1, tem2), axis=0)
    dr = rho * LA.norm(tem)

    # primal tolerance
    A_x = np.concatenate((XC.reshape(-1, 1), RC1.reshape(-1, 1), RC2.reshape(-1, 1)), axis=0)
    B_z = np.concatenate((xx.reshape((-1, 1)), R.reshape((-1, 1)), R.reshape((-1, 1))), axis=0)
    p_tol = sqrt_dim_p * ep_abs + ep_rel * max(LA.norm(A_x), LA.norm(B_z))

    # dual tolerance
    A_y = np.concatenate((Mu.reshape((-1, 1)), Ksi.reshape((-1, 1)), Eta.reshape((-1, 1))), axis=0)
    d_tol = sqrt_dim_n * ep_abs + ep_rel * LA.norm(A_y)

    # adaptively update penalty
    if pr > 10 * dr:
        rho *= 2
    elif dr > 10 * pr:
        rho /= 2

    # 每十次迭代输出一次结果
    if ite % 10 == 0:
        # temo_error = abs(tem_ob - gbv.ob) / gbv.ob * 100
        # gbv.o_err.append(temo_error)

        pri_re.append(pr)
        d_re.append(dr)

        wr = "Iteration : " + str(ite) + '\n' + '\n' + "Primal residual : " + str(
            pr) + '\n' + "Dual residual : " + str(dr) + '\n' + '\n'
        wr_s = open('LS.txt', 'a')
        wr_s.write(wr)
        wr_s.close()

time_end2 = time.time()
time_A = time_end2 - time_start2

# 收敛时计算所得解相应的目标函数
Ob = dim.comp_obj()

if ite % 10 != 0:
    # temo_error = abs(tem_ob - gbv.ob) / gbv.ob * 100
    # gbv.o_err.append(temo_error)

    pri_re.append(pr)
    d_re.append(dr)

    wr = "Iteration : " + str(ite) + '\n' + "Primal residual : " + str(
        pr) + '\n' + "Dual residual : " + str(dr) + '\n' + "Objective : " + str(Ob) + '\n' + "ADMM time: " + str(
        time_A) + '\n' + '\n' + "Finished!"

    wr_s = open('LS.txt', 'a')
    wr_s.write(wr)
    wr_s.close()
else:
    wr = '\n' + "Objective : " + str(Ob) + '\n' + "ADMM time: " + str(
        time_A) + '\n' + '\n' + "Finished!"
    wr_s = open('LS.txt', 'a')
    wr_s.write(wr)
    wr_s.close()

nz = len(pri_re)
pri_re = np.array(pri_re)
d_re = np.array(d_re)

plt.figure(1)
plt.plot(range(nz), pri_re, c='red', marker='*', linestyle='-', label='Primal residual')
plt.plot(range(nz), d_re, c='green', marker='+', linestyle='--', label='Dual residual')
plt.legend()
plt.title("primal/dual residual")  # 设置标题
plt.xlabel("iterations(*10)")  # 设置x轴标注
plt.ylabel("primal/dual residual")  # 设置y轴标注
plt.savefig("LS1.png")

'''
plt.figure(2)
plt.plot(range(nz), .Ob, c='blue', linestyle='-', label='Objective')
plt.title("Objective value")  # 设置标题
plt.xlabel("iterations(*10)")  # 设置x轴标注
plt.ylabel("Objective")  # 设置y轴标注
plt.savefig("LS2.png")

plt.figure(3)
plt.plot(range(nz), gbv.o_err, c='m', linestyle='-', label='Objective')
plt.title("Relative error of objective value")  # 设置标题
plt.xlabel("iterations(*10)")  # 设置x轴标注
plt.ylabel("Relative error (%)")  # 设置y轴标注
plt.savefig("LS3.png")

plt.figure(4)
plt.plot(range(nz), gbv.r_err, c='k', linestyle='-', label='Objective')
plt.title("Relative error of solutions (L2-norm)")  # 设置标题
plt.xlabel("iterations(*10)")  # 设置x轴标注
plt.ylabel("Relative error (%)")  # 设置y轴标注
plt.savefig("LS4.png")
'''




