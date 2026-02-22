import torch
from torch import nn


class TinyVGG(nn.Module):

    """
    Replicates the TinyVGG architecture from the CNN explainer website in PyTorch (With a little twist of flavor)
    See the original architecture here: https://poloclub.github.io/cnn-explainer/

    There are no parameters that you can alter here for creating the neural network
    It is defined as 3 consecutive 2D convolutional blocks, each block with (Conv2d, LeakyReLU, Conv2d, LeakyReLU, MaxPool2d)
    In channels = 10, Out channels = 10, kernel_size = 3, stride = 1, padding = 1
    Then a sequential layer with (Flatten, Linear, Softmax)
    """

    def __init__(self):

        super().__init__()

        #The in channels is no longer 1, as now we have 3 channels RGB from our input
        self.conv_b1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_b2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_b3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        #again, we don't know the input feature size, so we just run and see the tensor error, then replace it with the correct number
        self.end_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=2560, out_features=3), 
            nn.Softmax(dim=1)
        )

    #ok... it's not really just a image, when through all those convolution layers there's like 10 "feature" images running parallel
    def forward(self, image):
        return self.end_classifier(self.conv_b3(self.conv_b2(self.conv_b1(image))))