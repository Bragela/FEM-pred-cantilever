from numpy import dtype
import torch.nn as nn
import torch



class NeuralNet(nn.Module):
    def __init__(self, layer_sizes_geometry, layer_sizes_network):
        super(NeuralNet, self).__init__()
        layers_geometry = []
        layers_network = []

        input_features = 27
        for i, output_features in enumerate(layer_sizes_geometry):
            layers_geometry.append(nn.Linear(input_features, output_features))
            layers_geometry.append(nn.BatchNorm1d(output_features))
            layers_geometry.append(nn.ReLU())
            input_features = output_features
        self.layers_geometry = nn.Sequential(*layers_geometry)

        input_features = input_features + 3
        for i, output_features in enumerate(layer_sizes_network):
            layers_network.append(nn.Linear(input_features, output_features))
            #layers_network.append(nn.BatchNorm1d(output_features))
            layers_network.append(nn.ReLU())
            layers_network.append(nn.Dropout(0.1))
            input_features = output_features
        self.layers_network = nn.Sequential(*layers_network)

    def forward(self, center_pts, vectors, forces, coords):  # [B,4,3] [B,4,3] [B,3] [B,3,P]
    
        P = coords.shape[-1]                                                                            # P=2369
        batch_size = center_pts.shape[0]
        forces = forces.view(batch_size, 1, 3)                                                          # [B,1,3] 
        geometry_representation = torch.cat((center_pts, vectors, forces),dim = 1)                      # [B,9,3]
        geometry_representation = geometry_representation.view(batch_size, -1).to(torch.float16)        # [B,27]
        geometry_representation = self.layers_geometry(geometry_representation)                         # [B,32]
        feature_size = geometry_representation.shape[1]

        geometry_representation = geometry_representation.unsqueeze(1).repeat(1, P, 1)                  # [B,P,32]
        coords = coords.permute(0,2,1)                                                                  # [B,P,3]
        a = torch.cat((geometry_representation,coords), dim=2)                                          # [B,P,32+3]
        a = self.layers_network(a)                                                                      # [B,P,4]

        pred_disp = a[:,:,:3]      # [B,P,3]
        pred_mises = a[:,:,3]      # [B,P]

        return pred_disp, pred_mises