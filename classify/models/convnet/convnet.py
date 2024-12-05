import torch
import torch.nn as nn

try:
    from .modules import ConvModule
except:
    from  modules import ConvModule


# Convolutional Network
class ConvNet(nn.Module):
    def __init__(self,
                 img_size      :int = 224,
                 in_dim        :int = 3,
                 hidden_dim    :int = 16,
                 num_classes   :int = 10,
                 act_type      :str = "relu",
                 norm_type     :str = "bn",
                 depthwise     :bool = False,
                 use_adavgpool :bool = True,
                 ) -> None:
        super().__init__()
        # ---------- Basic parameters ----------
        self.img_size    = img_size
        self.num_classes = num_classes
        self.act_type    = act_type
        self.norm_type   = norm_type
        self.use_adavgpool = use_adavgpool
        self.layer_dims    = [hidden_dim, hidden_dim*2, hidden_dim*4, hidden_dim*4]
        # ---------- Model parameters ----------
        self.layer_1 = nn.Sequential(
            ConvModule(in_dim, hidden_dim,
                       kernel_size=3, padding=1, stride=2,
                       act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            ConvModule(hidden_dim, hidden_dim,
                       kernel_size=3, padding=1, stride=1,
                       act_type=act_type, norm_type=norm_type, depthwise=depthwise)            
        )
        self.layer_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvModule(hidden_dim, hidden_dim * 2,
                       kernel_size=3, padding=1, stride=1,
                       act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            ConvModule(hidden_dim * 2, hidden_dim * 2,
                       kernel_size=3, padding=1, stride=1,
                       act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        self.layer_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvModule(hidden_dim * 2, hidden_dim * 4,
                       kernel_size=3, padding=1, stride=1,
                       act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            ConvModule(hidden_dim * 4, hidden_dim * 4,
                       kernel_size=3, padding=1, stride=1,
                       act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        self.layer_4 = nn.Sequential(
            ConvModule(hidden_dim * 4, hidden_dim * 4,
                       kernel_size=3, padding=1, stride=1,
                       act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            ConvModule(hidden_dim * 4, hidden_dim * 4,
                       kernel_size=3, padding=1, stride=1,
                       act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )

        if use_adavgpool:
            self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.fc      = nn.Linear(hidden_dim * 4, num_classes)
        else:
            self.avgpool = None
            fc_in_dim    = (img_size // 8) ** 2 * (hidden_dim * 4)  # N = Co x Ho x W
            self.fc      = nn.Linear(fc_in_dim , num_classes)

    def forward(self, x):
        """
        Input:
            x : (torch.Tensor) -> [B, C, H, W]
        Output:
            x : (torch.Tensor) -> [B, Nc], Nc is the number of the object categories.
        """
        # [B, C_in, H, W]   -> [B, C1, H/2, W/2]
        x = self.layer_1(x)
        # [B, C1, H/2, W/2] -> [B, C2, H/4, W/4]
        x = self.layer_2(x)
        # [B, C2, H/4, W/4] -> [B, C3, H/8, W/8]
        x = self.layer_3(x)
        # [B, C3, H/8, W/8] -> [B, C3, H/8, W/8]
        x = self.layer_4(x)

        if self.use_adavgpool:
            x = self.avgpool(x)

        # reshape [B, Co, Ho, Wo] to [B, N], N = Co x Ho x Wo
        x = x.flatten(1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    bs, img_dim, img_size = 8, 3, 28
    hidden_dim  = 16
    num_classes = 10
    
    # Make an input data randomly
    x = torch.randn(bs, img_dim, img_size, img_size)

    # Build a MLP model
    model = ConvNet(img_size      = img_size,
                    in_dim        = img_dim,
                    hidden_dim    = hidden_dim,
                    num_classes   = num_classes,
                    act_type      = 'relu',
                    norm_type     = 'bn',
                    depthwise     = False,
                    use_adavgpool = False)

    # Inference
    output = model(x)
    print(output.shape)
