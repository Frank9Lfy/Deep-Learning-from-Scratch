import numpy as np

"""
SGD的缺点：若函数的形状非均向 例如 f(x,y) = 1/20 * x^2 + y^2  梯度的方向没有指向最小值的方向
搜索路径（个人理解：沿梯度下降的路径）呈锯齿状，学习效率低
"""
class SGD:
    """随机梯度下降法（Stochastic Gradient Descent）"""

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


# print('='*15,"Momentum优化算法",'='*20,'\n')
"""类比动量，v为速度
个人理解：之前的“速度”（虽然有权重α）对当前的“速度”有累积的影响
（同向速度累加，反向抵消，故没有锯齿状，且更快到达距离底部较近的地方），
而SGD只看当前的梯度
"""
class Momentum:
    def __init__(self,lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    def update(self,params,grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


# print('='*15,"AdaGrad优化算法",'='*20,'\n')
"""
学习率衰减： 随着学习的进行，学习率逐渐减小
AdaGrad的伟大之处：为参数的*每个*元素调整学习率 (adaptive)
AdaGrad会记录过去所有梯度的平方和，
学习越深入，更新的幅度就越小(有可能趋于0，这是一个problem，使用RMSProp（逐渐遗忘过去的梯度）来改善)
"""
class AdaGrad:
    def __init__(self,lr=0.01,eps=1e-8):
        self.lr = lr
        self.h = None
        self.eps = eps
    def update(self,params,grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key]/(np.sqrt(self.h[key]) + self.eps)


class RMSprop:
    """RMSprop"""

    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


"Adam: 将Momentum和AdaGrad融合"
# 以下代码摘自"本书配套资源【源代码】深度学习入门：基于Python的理论与实现_20240716\common\optimizer.py"

class Adam:
    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            # self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            # self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

            # unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            # unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            # params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)


