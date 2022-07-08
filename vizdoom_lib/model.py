from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    [120][160][3]
    conv2d(kernel size 3, 16 feature planes, padding=1)
    [120][160][16]
    maxpooling(kernel size 4)
    [30][40][16]
    ReLU()
    conv2d(kernel size 3, 16 feature planes, padding=1)
    [30][40][16]
    maxpooling(kernel size 4)
    [7][10][16]
    ReLU()
    linear(7 * 10 * 16, 3)
    """
    def __init__(self, image_height: int, image_width: int, num_actions: int):
        super().__init__()
        h = image_height
        w = image_width
        self.c1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=4)
        h //= 4
        w //= 4
        self.c2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=4)
        h //= 4
        w //= 4

        self.output = nn.Linear(h * w * 16, num_actions)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.c1(x)
        x = self.pool1(x)
        x = F.relu(x)

        x = self.c2(x)
        x = self.pool2(x)
        x = F.relu(x)
        # [C][H][W]
        # [C * H * W]
        x = x.view(batch_size, -1)
        x = self.output(x)
        return x
