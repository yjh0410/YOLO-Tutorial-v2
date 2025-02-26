import torch.nn as nn


def get_activation(act_type=None):
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'mish':
        return nn.Mish(inplace=True)
    elif act_type == 'silu':
        return nn.SiLU(inplace=True)
    elif act_type is None:
        return nn.Identity()

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([in_dim] + h, h + [out_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class FFN(nn.Module):
    def __init__(self, d_model=256, ffn_dim=1024, dropout=0., act_type='relu', pre_norm=False):
        super().__init__()
        # ----------- Basic parameters -----------
        self.pre_norm = pre_norm
        self.ffn_dim = ffn_dim
        # ----------- Network parameters -----------
        self.linear1 = nn.Linear(d_model, self.ffn_dim)
        self.activation = get_activation(act_type)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(self.ffn_dim, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
        if self.pre_norm:
            src = self.norm(src)
            src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
            src = src + self.dropout3(src2)
        else:
            src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
            src = src + self.dropout3(src2)
            src = self.norm(src)
        
        return src
