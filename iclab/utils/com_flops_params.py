import torch
from thop import profile


def FLOPs_and_Params(model, size, img_dim, device):
    x = torch.randn(1, img_dim, size, size).to(device)
    model.eval()

    flops, params = profile(model, inputs=(x, ))
    print('=================== FLOPs & Params ===================')
    print('- GFLOPs : ', flops / 1e9 * 2)
    print('- Params : ', params / 1e6, ' M')

    model.train()


if __name__ == "__main__":
    pass
