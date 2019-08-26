# _*_coding:utf-8_*_

'''
可视马尔科夫模型(VMM)与隐马尔科夫模型(HMM)
1.数学上最漂亮的办法是最大熵(maximum entropy)模型,
就是要保留全部的不确定性，将风险降到最小。
2. 匈牙利著名数学家、信息论最高奖香农奖得主希萨（Csiszar）证明，
对任何一组不自相矛盾的信息，这个最大熵模型不仅存在，而且是唯一的。
而且它们都有同一个非常简单的形式 -- 指数函数。
3. 已知模型的类型(比如指数函数，高斯函数...)，要求模型的参数,这个过程称为模型的训练。
4. HMM包含可见状态联合隐含状态连：
(1). 隐含状态之间存在转换概率(transmition probability),一般HMM中说到的马尔科夫链其实是指隐含状态连。
(2). 可见状态之间没有转换概率。
(3). 尽管可见状态之间没有转换概率，但是隐含状态和可见状态之间有一个概率叫做输出概率(emission probability)。
HMM模型相关的算法主要分为三类，分别解决三种问题： 
      1）知道骰子有几种（隐含状态数量），每种骰子是什么（转换概率），根据掷骰子掷出的结果（可见状态链），
我想知道每次掷出来的都是哪种骰子（隐含状态链）。 
      这个问题呢，在语音识别领域呢，叫做解码问题。这个问题其实有两种解法，会给出两个不同的答案。
每个答案都对，只不过这些答案的意义不一样。第一种解法求最大似然状态路径，说通俗点呢，就是我求一串骰子序列，
这串骰子序列产生观测结果的概率最大。第二种解法呢，就不是求一组骰子序列了，而是求每次掷出的骰子分别是某种
骰子的概率。比如说我看到结果后，我可以求得第一次掷骰子是D4的概率是0.5，D6的概率是0.3，D8的概率是0.2.第
一种解法我会在下面说到，但是第二种解法我就不写在这里了，如果大家有兴趣，我们另开一个问题继续写吧。
2）还是知道骰子有几种（隐含状态数量），每种骰子是什么（转换概率），根据掷骰子掷出的结果（可见状态链），
我想知道掷出这个结果的概率。看似这个问题意义不大，因为你掷出来的结果很多时候都对应了一个比较大的概率。问
这个问题的目的呢，其实是检测观察到的结果和已知的模型是否吻合。如果很多次结果都对应了比较小的概率，那么就
说明我们已知的模型很有可能是错的，有人偷偷把我们的骰子給换了。
3）知道骰子有几种（隐含状态数量），不知道每种骰子是什么（转换概率），观测到很多次掷骰子的结果(可见状态链)
，我想反推出每种骰子是什么(转换概率)。这个问题很重要，因为这是最常见的情况。很多时候我们只有可见结果，
不知道HMM模型里的参数，我们需要从可见结果估计出这些参数，这是建模的一个必要步骤。
5. Dynamic programming algorithm and Viterbi algorithm?
6. Viterbi algorithm 属于动态规划算法的一种?
7. HMM适合缺失一部分信息的情况吗？
8. 最大似然状态路径问题求解(求解隐含状态链)
'''

import sys
import numpy as np

def test_hmm():
    #   隐状态
    hidden_state = ['sunny', 'rainy']

    #   观测序列
    obsevition = ['walk', 'shop', 'clean']

    #   根据观测序列、发射概率、状态转移矩阵、发射概率
    #   返回最佳路径

    def viterbi(obs, states, start_p, trans_p, emit_p):
        #   max_p（3*2）每一列存储第一列不同隐状态的最大概率
        max_probs = np.zeros((len(obs), len(states)))

        #   path（2*3）每一行存储上max_p对应列的路径
        path = np.zeros((len(states), len(obs)))

        #   初始化
        for i in range(len(states)):
            max_probs[0][i] = start_p[i] * emit_p[i][obs[0]]
            path[i][0] = i

        for t in range(1, len(obs)):

            new_path = np.zeros((len(states), len(obs)))

            for state_cur in range(len(states)):

                prob_max = -1

                for state_pre in range(len(states)):
                    prob = max_probs[t-1][state_pre] * \
                        trans_p[state_pre][state_cur] * \
                        emit_p[state_cur][obs[t]]

                    if prob > prob_max:
                        prob_max = prob
                        state_pre = state_pre

                        # record max probs
                        max_probs[t][state_cur] = prob_max

                        # update path
                        for t_pre in range(t):
                            new_path[state_cur][t_pre] = path[state_pre][t_pre]
                        new_path[state_cur][t] = state_cur

            path = new_path

        max_prob = -1
        path_state = 0

        #   返回最大概率的路径
        for state_cur in range(len(states)):
            if max_probs[len(obs)-1][state_cur] > max_prob:
                max_prob = max_probs[len(obs)-1][state_cur]
                path_state = state_cur

        return path[path_state]

    state_s = [0, 1]
    observed = [0, 1, 2, 2, 2, 1, 0, 0]

    #   初始状态，测试集中，0.6概率观测序列以sunny开始
    start_probability = [0.6, 0.4]

    #   转移概率，0.7：sunny下一天sunny的概率
    transititon_probability = np.array([[0.7, 0.3], [0.4, 0.6]])

    #   发射概率，0.3：sunny在0.3概率下为shop
    emission_probability = np.array([[0.6, 0.3, 0.1], [0.1, 0.4, 0.5]])

    result = viterbi(observed, state_s, start_probability,
                     transititon_probability, emission_probability)

    for k in range(len(result)):
        if k != len(result) - 1:
            print(hidden_state[int(result[k])], end='=>')
        else:
            print(hidden_state[int(result[k])])


if __name__ == "__main__":
    test_hmm()

    print('\n=> Test done.')


# ref:
# http://blog.163.com/zhoulili1987619@126/blog/static/353082012013113191924334/
# https://www.cnblogs.com/skyme/p/4651331.html (一文搞懂HMM)
# http://blog.csdn.net/shijing_0214/article/details/51173887
# http://blog.csdn.net/tostq/article/details/70854455
