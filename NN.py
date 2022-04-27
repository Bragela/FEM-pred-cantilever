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
            #layers.append(nn.BatchNorm1d(output_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_features = output_features
        self.layers = nn.Sequential(*layers)


    def forward(self, forces):  # [B,3]
        
        batch_size = forces.shape[0]
        print(forces.shape)
        a = self.layers(forces)         #[B,P x 4]
        a = a.view(batch_size,-1,4)

        pred_disp = a[:,:,:3]      # [B,P,3]
        pred_mises = a[:,:,3]      # [B,P]
        print(pred_disp.shape,pred_mises.shape)

        return pred_disp, pred_mises
