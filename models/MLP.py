import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # nn.Linear(2048, 4096),
            # nn.ReLU(True),
            # nn.Linear(4096, 2048),
            # nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64))
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048)
            # nn.ReLU(True),
            # nn.Linear(2048, 4096),
            # nn.ReLU(True),
            # nn.Linear(4096, 2048)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
