from numpy import dtype
import torch.nn as nn
import torch
import torch.nn.functional as F


class NeuralNet(nn.Module):

    def __init__(self, layer_sizes):
        super(NeuralNet, self).__init__()
        layers = []
        input_features = 1
        for i, output_features in enumerate(layer_sizes):
            layers.append(nn.Linear(input_features, output_features))
            #layers.append(nn.BatchNorm1d(output_features))
            layers.append(nn.ReLU())
            input_features = output_features
        self.layers = nn.Sequential(*layers)

        # for param in self.layers.parameters():
        #     if isinstance(param, nn.Linear):
        #         print('here')
        #         torch.nn.init.normal_(param.weight, 0, 0.001)
        #         torch.nn.init.zeros_(param.bias)

    def forward(self, forces):  # [B,1]       
        pred_mises = self.layers(forces)   # [B,P]
        return pred_mises
