import torch
import torch.nn as nn

class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()

    #Batch Normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(256)

    #Regularization
        self.dropout1 = nn.Dropout2d(p=0.15)
        self.dropout2 = nn.Dropout(p=0.3)

    #Activations
        self.activation = nn.ReLU()

    #Convolution Layers        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 256, 3, padding=1)

    #Skip Connections
        self.skip1 = nn.Identity()
        self.skip2 = nn.Conv2d(64, 128, 1)
        self.skip3 = nn.Conv2d(128, 256, 1)
        self.bn_skip2 = nn.BatchNorm2d(128)
        self.bn_skip3 = nn.BatchNorm2d(256)

    #Pooling Layers
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(1)
        
    #Classifier
        self.linear = nn.Linear(256, 100)


    def forward(self, x):

        #CONVOLUTION 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool1(x)

        #CONVOLUTION 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool2(x)

        #RESIDUAL BLOCK
        identity1 = x

        x = self.conv3(x)
        x = self.bn3(x)

        skip1 = self.skip1(identity1)

        x = x + skip1
        x = self.activation(x)

        #RESIDUAL BLOCK 2
        identity2 = x

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)

        x = self.conv5(x)
        x = self.bn5(x)

        skip2 = self.skip2(identity2)
        skip2 = self.bn_skip2(skip2)

        x = x + skip2
        x = self.activation(x)

        #RESIDUAL BLOCK 3
        identity3 = x

        x = self.conv6(x)
        x = self.bn6(x)

        skip3 = self.skip3(identity3)
        skip3 = self.bn_skip3(skip3)

        x = x + skip3
        x = self.activation(x)

        #CLASSIFIER
        x = self.dropout1(x)
        x = self.pool3(x)
        x = torch.flatten(x, 1)

        x = self.dropout2(x)
        x = self.linear(x)

        return x

        
        
    
