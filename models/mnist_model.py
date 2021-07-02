import torch
import torch.nn as nn
import torch.nn.functional as F

# Object Recognition
dropout_value = 0.05
class Net(nn.Module):
    def __init__(self, norm_type):
        super(Net, self).__init__()

        self.norm_type = norm_type
        assert self.norm_type in ('BatchNorm', 'GroupNorm', 'LayerNorm'), "Incorrect normalization applied"

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.norm_layer(self.norm_type, 8),
            nn.Dropout(dropout_value)                        
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            self.norm_layer(self.norm_type, 16),
            nn.Dropout(dropout_value)                     
        )
        # Maxpooling
        self.pool1 = nn.MaxPool2d(2, 2) 
        # TRANSITION BLOCK 1
        self.transitionblock1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            self.norm_layer(self.norm_type, 12),
            nn.Dropout(dropout_value)                        
        )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            self.norm_layer(self.norm_type, 12),
            nn.Dropout(dropout_value)                        
        )
       
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            self.norm_layer(self.norm_type, 16),
            nn.Dropout(dropout_value)                        
        )
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            self.norm_layer(self.norm_type, 16),
            nn.Dropout(dropout_value)                        
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        )

        self.translinear = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=1, padding=0, bias=False),
            )
    
    def norm_layer(self, norm_type, channels):
        if norm_type == 'BatchNorm':
            return nn.BatchNorm2d(channels)
        elif norm_type == 'GroupNorm':
            return nn.GroupNorm(num_groups=int(channels/2), num_channels=channels)
        elif norm_type == 'LayerNorm':
            return nn.GroupNorm(num_groups=1, num_channels=channels)


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.transitionblock1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.gap(x)
        x = self.translinear(x)
        x = x.view(-1, 10)
        return x 