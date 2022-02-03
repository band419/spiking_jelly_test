import torch



if __name__ == '__main__':
    x = torch.randn(2,2)
    print(x)
    out_spike = torch.rand_like(x).le(x).to(x)
    print(out_spike)