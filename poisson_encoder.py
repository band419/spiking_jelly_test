import torch
from spikingjelly.clock_driven.encoding import StatelessEncoder


class PoissonEncoder(StatelessEncoder):
    def __init__(self, max_freq):
        """
        * :ref:`API in English <PoissonEncoder.__init__-en>`

        .. _PoissonEncoder.__init__-cn:

        无状态的泊松编码器。输出脉冲的发放概率与输入 ``x`` 相同。

        .. warning::

            必须确保 ``0 <= x <= 1``。

        * :ref:`中文API <PoissonEncoder.__init__-cn>`

        .. _PoissonEncoder.__init__-en:

        The poisson encoder will output spike whose firing probability is ``x``。

        .. admonition:: Warning
            :class: warning

            The user must assert ``0 <= x <= 1``.
        """
        super().__init__()
        self.max_freq = max_freq

    def forward(self, x: torch.Tensor):

        out_spike = torch.rand_like(x).le(x).to(x)
        return out_spike