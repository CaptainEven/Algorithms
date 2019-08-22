#_*_coding:utf-8

'''
1. 极大似然估计取对数操作的好处：
(1). log函数在定义域上是单调函数，便于求极值 
(2). 便于计算机计算，因为过小的值做乘法运算可能会导致溢出，
取对数操作之后，将乘法转换为加法，避免连乘运算溢出。
2. 如果一个变量的期望等于他的理想值，那么就称该变量无偏；否则称为有偏
    加载Irish数据, 4维, 3个类别，故3个高斯模型
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math
import random


param_dict = {}
param_dict['Mu_1'] = np.array([0, 0])
param_dict['Sigma_1'] = np.array([[1, 0], [0, 1]])
param_dict['Mu_2'] = np.array([0, 0])
param_dict['Sigma_2'] = np.array([[1, 0], [0, 1]])
param_dict['Pi_weight'] = 0.5
param_dict['Gamma_list'] = []


def set_param(mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, pi_weight):
    param_dict['Mu_1'] = mu_1
    param_dict['Mu_1'].shape = (4, 1)    # 一个数据4维
    param_dict['Sigma_1'] = sigma_1      # 4行4列的方阵
    param_dict['Mu_2'] = mu_2
    param_dict['Mu_2'].shape = (4, 1)
    param_dict['Sigma_2'] = sigma_2
    param_dict['Mu_3'] = mu_3
    param_dict['Mu_3'].shape = (4, 1)
    param_dict['Sigma_3'] = sigma_3
    param_dict['Pi_weight'] = pi_weight  # 3行一列


def PDF(data, Mu, sigma, n_dim):
    '''
    multivariate Gauss probability density function
    @param data: input n-dim data
    @param Mu: mean value(ndarray)
    @param sigma: covariant matrix(ndarray)
    @retur: probability value of input data
    '''
    sigma_sqrt = math.sqrt(np.linalg.det(sigma))  # 协方差矩阵绝对值的1/2次方
    sigma_inv = np.linalg.inv(sigma)              # 协方差矩阵的逆
    data.shape = (n_dim, 1)
    Mu.shape = (n_dim, 1)
    minus_mu = data - Mu
    minus_mu_trans = np.transpose(minus_mu)
    res = (1.0 / (2.0 * math.pi * sigma_sqrt)) \
        * math.exp((-0.5) * (np.dot(np.dot(minus_mu_trans, sigma_inv), minus_mu)))
    return res


def E_step(Data):
    '''
    E-step: compute responsibilities
    calculate current round's Gamma_list
    @param Data: a series of 4D points
    @return Gamma_list: probability's estimate
    '''

    # priori covariance matrix
    sigma_1 = param_dict['Sigma_1']
    sigma_2 = param_dict['Sigma_2']
    sigma_3 = param_dict['Sigma_3']

    # priori weight
    pw = param_dict['Pi_weight']  # 3 rows and 1 column

    # priori mean value
    mu_1 = param_dict['Mu_1']
    mu_2 = param_dict['Mu_2']
    mu_3 = param_dict['Mu_3']

    # compute expectation(probability)
    param_dict['Gamma_list'] = np.zeros((len(Data), 3))
    for (i, point) in enumerate(Data):
        pb_1 = pw[0] * PDF(point, mu_1, sigma_1, 4)
        pb_2 = pw[1] * PDF(point, mu_2, sigma_2, 4)
        pb_3 = pw[2] * PDF(point, mu_3, sigma_3, 4)

        # mormalize the probability
        pb_sum = pb_1 + pb_2 + pb_3
        param_dict['Gamma_list'][i][0] = pb_1 / pb_sum
        param_dict['Gamma_list'][i][1] = pb_2 / pb_sum
        param_dict['Gamma_list'][i][2] = pb_3 / pb_sum


def M_step(Data):
    '''
    M_step: compute weighted means and variance
    update mean and covariance matrix
    @param Data: input 4D points
    '''
    # compute sum of probability
    N_1 = 0.0
    N_2 = 0.0
    N_3 = 0.0
    for gamma in param_dict['Gamma_list']:
        N_1 += gamma[0]
        N_2 += gamma[1]
        N_3 += gamma[2]

    # update mean value:miu(μ)
    new_mu_1 = np.array([0, 0, 0, 0])  # 4D data
    new_mu_2 = np.array([0, 0, 0, 0])
    new_mu_3 = np.array([0, 0, 0, 0])
    for (i, gamma) in enumerate(param_dict['Gamma_list']):
        new_mu_1 = new_mu_1 + Data[i] * gamma[0] / N_1
        new_mu_2 = new_mu_2 + Data[i] * gamma[1] / N_2
        new_mu_3 = new_mu_3 + Data[i] * gamma[2] / N_3

    # specify shape for calculation
    new_mu_1.shape = (4, 1)
    new_mu_2.shape = (4, 1)
    new_mu_3.shape = (4, 1)

    # update covariance matrix:sigma(Σ) 4×4
    new_sigma_1 = np.zeros((4, 4))
    new_sigma_2 = np.zeros((4, 4))
    new_sigma_3 = np.zeros((4, 4))
    for i in range(len(param_dict['Gamma_list'])):
        gamma = param_dict['Gamma_list'][i]
        X = np.array([[Data[i][0]], [Data[i][1]], [
                     Data[i][2]], [Data[i][3]]])  # 1row 3columns
        new_sigma_1 = new_sigma_1 + \
            np.dot((X - new_mu_1), (X - new_mu_1).transpose()) * gamma[0] / N_1
        new_sigma_2 = new_sigma_2 + \
            np.dot((X - new_mu_2), (X - new_mu_2).transpose()) * gamma[1] / N_2
        new_sigma_3 = new_sigma_3 + \
            np.dot((X - new_mu_3), (X - new_mu_3).transpose()) * gamma[2] / N_3

    # update weight
    new_pi = np.zeros(3)
    N_sum = N_1 + N_2 + N_3
    new_pi[0] = N_1 / N_sum
    new_pi[1] = N_2 / N_sum
    new_pi[2] = N_3 / N_sum

    # write back updated parameters
    param_dict['Mu_1'] = new_mu_1
    param_dict['Mu_2'] = new_mu_2
    param_dict['Mu_3'] = new_mu_3
    param_dict['Sigma_1'] = new_sigma_1
    param_dict['Sigma_2'] = new_sigma_2
    param_dict['Sigma_3'] = new_sigma_3
    param_dict['Pi_weight'] = new_pi


def EM_iterate(iter_time, Data,
               mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3,
               pi_weight, esp=1e-7):
    '''
    @param iter_time: iterative times
    @param Data: input 4D data
    @param esp: stopping criterion for iteration
    '''
    set_param(mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, pi_weight)
    count = 0
    print('--Start...\n')
    if iter_time == None:
        while True:
            count += 1
            print('\n-- Round %d' % count)
            old_mu_1 = param_dict['Mu_1'].copy()
            old_mu_2 = param_dict['Mu_2'].copy()
            old_mu_3 = param_dict['Mu_3'].copy()

            # EM process
            E_step(Data)
            M_step(Data)

            # judge delta
            delta_1 = param_dict['Mu_1'] - old_mu_1  # delta是1行4列
            delta_2 = param_dict['Mu_2'] - old_mu_2
            delta_3 = param_dict['Mu_3'] - old_mu_3

            if math.fabs(delta_1[0]) < esp and math.fabs(delta_1[1]) < esp \
                    and math.fabs(delta_1[2]) < esp and math.fabs(delta_1[3]) < esp \
                    and math.fabs(delta_2[0]) < esp and math.fabs(delta_2[1]) < esp \
                    and math.fabs(delta_2[2]) < esp and math.fabs(delta_2[3]) < esp \
                    and math.fabs(delta_3[0]) < esp and math.fabs(delta_3[1]) < esp \
                    and math.fabs(delta_3[2]) < esp and math.fabs(delta_3[3]) < esp:
                break

            # show iteration result
            print('Mu_1:\n', param_dict['Mu_1'])
            print('Mu_2:\n', param_dict['Mu_2'])
            print('Mu_3:\n', param_dict['Mu_3'])
            print('Sigma_1:\n', param_dict['Sigma_1'])
            print('Sigma_2:\n', param_dict['Sigma_2'])
            print('Sigma_3:\n', param_dict['Sigma_3'])
            print('Pi_weight:\n', param_dict['Pi_weight'])
    else:
        for i in range(iter_time):
            count += 1
            print('\n-- Round %d' % count)
            old_mu_1 = param_dict['Mu_1'].copy()
            old_mu_2 = param_dict['Mu_2'].copy()
            E_step(Data)
            M_step(Data)
            delta_1 = param_dict['Mu_1'] - old_mu_1
            delta_2 = param_dict['Mu_2'] - old_mu_2

            # show iteration result
            print('Mu_1:\n', param_dict['Mu_1'])
            print('Mu_2:\n', param_dict['Mu_2'])
            print('Mu_3:\n', param_dict['Mu_3'])
            print('Sigma_1:\n', param_dict['Sigma_1'])
            print('Sigma_2:\n', param_dict['Sigma_2'])
            print('Sigma_3:\n', param_dict['Sigma_3'])
            print('Pi_weight:\n', param_dict['Pi_weight'])


def task_1(iter_num=None):
    # read data and run algorithm
    Data_list = []
    with open('./iris/iris.data', 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            data = line.strip().split(',')
            if len(data) != 5:
                print(line)
                continue

            point = []
            try:
                point.append(float(data[0]))
                point.append(float(data[1]))
                point.append(float(data[2]))
                point.append(float(data[3]))
            except Exception as e:
                print(e)

            Data_list.append(point)
    Data = np.array(Data_list)  # turn list into numpy array
    # print('Data:\n', Data)

    # initiate parameters
    Mu_1 = np.array([5.8, 2.9, 1.2, 1.5])
    Sigma_1 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    Mu_2 = np.array([4.9, 3.2, 1.1, 0.3])
    Sigma_2 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    MU_3 = np.array([5.0, 3.2, 1.0, 0.2])
    Sigma_3 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    Pi_weight = np.array([0.33333, 0.33333, 0.33333])

    EM_iterate(iter_num, Data, Mu_1, Sigma_1,
               Mu_2, Sigma_2, MU_3, Sigma_3, Pi_weight)


task_1()

# ref:
# http://blog.csdn.net/xiaopangxia/article/details/53542666
# <<EM算法推导与GMM训练应用>>
