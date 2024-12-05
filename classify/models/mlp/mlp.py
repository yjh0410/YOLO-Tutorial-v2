import torch
import torch.nn as nn

try:
    from .modules import SLP
except:
    from  modules import SLP


# Multi Layer Perceptron
class MLP(nn.Module):
    def __init__(self,
                 in_dim     :int,
                 inter_dim  :int,
                 out_dim    :int,
                 act_type   :str = "sigmoid",
                 norm_type  :str = "bn") -> None:
        super().__init__()
        self.stem   = SLP(in_dim, inter_dim, act_type, norm_type)
        self.layers = nn.Sequential(
            SLP(inter_dim, inter_dim, act_type, norm_type),
            SLP(inter_dim, inter_dim, act_type, norm_type),
            SLP(inter_dim, inter_dim, act_type, norm_type),
            SLP(inter_dim, inter_dim, act_type, norm_type),            
            )
        self.fc     = nn.Linear(inter_dim, out_dim)

    def forward(self, x):
        """
        Input:
            x : (torch.Tensor) -> [B, C, H, W] or [B, C]
        """
        if len(x.shape) > 2:
            x = x.flatten(1)

        x = self.stem(x)
        x = self.layers(x)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    bs, c = 8, 256
    hidden_dim  = 512
    num_classes = 10
    
    # Make an input data randomly
    x = torch.randn(bs, c)

    # Build a MLP model
    model = MLP(in_dim=c, inter_dim=hidden_dim, out_dim=num_classes, act_type='sigmoid', norm_type='bn')

    # Inference
    output = model(x)
    print(output.shape)
