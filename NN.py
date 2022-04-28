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

    def forward(self, forces, coords):  # [B,1] [B,P,3]
        B = forces.shape[0]
        P = coords.shape[1]

        forces = forces.view(B,1,1).repeat(1,P,1)
        a = torch.cat((forces,coords), dim=2)           #[B,P,4]
        pred_mises = self.layers(forces)                #[B,P,1]


        return pred_mises
