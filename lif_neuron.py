from typing import Callable

import matplotlib
import torch
from spikingjelly.clock_driven import surrogate
from spikingjelly.clock_driven.neuron import BaseNode
import torch
import torch.nn as nn
import numpy as np
from spikingjelly.clock_driven import neuron
from spikingjelly import visualizing
from matplotlib import pyplot as plt


class LIFNodeCustom(BaseNode):
    def __init__(self, tau_curr: float = 2., tau_mem: float = 2., no_active_time: int = 2, resistance: float = 1.,
                 v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):
        """
        * :ref:`API in English <LIFNode.__init__-en>`

        .. _LIFNode.__init__-cn:

        :param tau: 膜电位时间常数
        :type tau: float

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool


        Leaky Integrate-and-Fire 神经元模型，可以看作是带漏电的积分器。其阈下神经动力学方程为：

        .. math::
            V[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset})

        * :ref:`中文API <LIFNode.__init__-cn>`

        .. _LIFNode.__init__-en:

        :param tau: membrane time constant
        :type tau: float

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        The Leaky Integrate-and-Fire neuron, which can be seen as a leaky integrator.
        The subthreshold neural dynamics of it is as followed:

        .. math::
            V[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset})
        """
        assert isinstance(tau_mem, float) and tau_mem > 1.
        assert isinstance(tau_curr, float) and tau_curr > 1.
        assert isinstance(resistance, float)

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.tau_mem = tau_mem
        self.tau_curr = tau_curr
        self.no_active_time = no_active_time
        self.resistance = resistance

        # register current
        self.register_memory('c', 0.)
        self.register_memory('hyper_time', 0)

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'

    def neuronal_charge(self, x: torch.Tensor):
        # update current
        # C(t+1) = (C(t) + w) - (C(t) + w)/ tau_curr * delta T
        self.c = self.c + x - (self.c + x) / self.tau_curr
        # update voltage
        # if self.hyper_time > 0:
        self.v = self.v + (self.v_reset + self.resistance * self.c - self.v) / self.tau_mem

    def neuronal_fire(self):
        """
        * :ref:`API in English <BaseNode.neuronal_fire-en>`

        .. _BaseNode.neuronal_fire-cn:

        根据当前神经元的电压、阈值，计算输出脉冲。

        * :ref:`中文API <BaseNode.neuronal_fire-cn>`

        .. _BaseNode.neuronal_fire-en:


        Calculate out spikes of neurons by their current membrane potential and threshold voltage.
        """

        self.spike = self.surrogate_function(self.v - self.v_threshold)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    lif = LIFNodeCustom(tau_curr=5., tau_mem=30., no_active_time=4, resistance=1., v_threshold=-0.4, v_reset=-0.65,
                        surrogate_function=surrogate.Sigmoid(),
                        detach_reset=False
                        )
    lif.reset()
    x = torch.as_tensor([0.1])
    T = 150
    s_list = []
    v_list = []
    c_list = []
    for t in range(T):
        s_list.append(lif(x))
        v_list.append(lif.v)
        c_list.append(lif.c)

    # plot
    fig = plt.figure(dpi=200)
    ax0 = plt.subplot2grid((6, 1), (0, 0), rowspan=2)
    ax0.set_title('$V_{t}$, $C_{t}$  and $S_{t}$ of the neuron')
    T = np.asarray(s_list).shape[0]
    t = np.arange(0, T)
    ax0.plot(t, np.asarray(v_list))
    ax0.set_xlim(-0.5, T - 0.5)
    ax0.set_ylabel('voltage')
    ax0.axhline(lif.v_threshold, label='$V_{threshold}$', linestyle='-.', c='r')
    if lif.v_reset is not None:
        ax0.axhline(lif.v_reset, label='$V_{reset}$', linestyle='-.', c='g')
    ax0.legend()
    t_spike = np.asarray(s_list) * t
    s = np.asarray(s_list)
    mask = (s == 1)  # eventplot中的数值是时间发生的时刻，因此需要用mask筛选出
    ax3 = plt.subplot2grid((6, 1), (3, 0))
    ax3.plot(t, np.asarray(c_list))
    ax1 = plt.subplot2grid((6, 1), (5, 0))
    ax1.eventplot(t_spike[mask], lineoffsets=0, colors='r')
    ax1.set_xlim(-0.5, T - 0.5)

    ax1.set_xlabel('simulating step')
    ax1.set_ylabel('spike')
    ax1.set_yticks([])

    ax1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    plt.show()
