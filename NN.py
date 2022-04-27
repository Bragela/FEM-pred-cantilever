from numpy import dtype
import torch.nn as nn
import torch
import torch.nn.functional as F


class NeuralNet(nn.Module):

    def __init__(self, layer_sizes):
        super(NeuralNet, self).__init__()

        layers = []

        input_features = 3
        for i, output_features in enumerate(layer_sizes):
            layers.append(nn.Linear(input_features, output_features))
            layers.append(nn.ReLU())
            input_features = output_features
        self.layers = nn.Sequential(*layers)


    def forward(self, forces):  # [B,3]

        B = forces.shape[0]
        pred_mises = self.layers(forces)   # [B,P]

        return pred_mises
