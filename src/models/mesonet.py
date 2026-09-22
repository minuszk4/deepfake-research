import torch.nn as nn

class Meso4(nn.Module):
    def __init__(self, num_classes=1):
        super(Meso4, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(8, 8, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(8)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.maxpool4 = nn.MaxPool2d(4, 4)
        
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.maxpool1(self.relu(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.relu(self.bn2(self.conv2(x))))
        x = self.maxpool3(self.relu(self.bn3(self.conv3(x))))
        x = self.maxpool4(self.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1) 
        x = self.dropout(self.leakyrelu(self.fc1(x)))
        return self.fc2(x)
