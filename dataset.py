from tkinter import Y
from matplotlib.pyplot import axis
from sympy import Float
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import random
import math
import matplotlib.pylab as plt
from sklearn import preprocessing



class GridDataset(Dataset):
    def __init__(self, root_dir="data", split="train", force_scaler=None, coords_scaler=None):
        self.data_path = f"{root_dir}/{split}" 
        self.data = [folder for folder in os.listdir(self.data_path)]
        self.split = split
        self.force_scaler = force_scaler
        self.coords_scaler = coords_scaler

    def __len__(self):
        return len(self.data)
        #return 1

    def __getitem__(self, idx):
        folder_name = self.data[idx]
        full_path = f"{self.data_path}/{folder_name}"

        forces = []
        with open(f'{full_path}/Input.txt','r') as f:
            
            for i, line in enumerate(f):
                F = line.rstrip('\n')
                #forces.append(float(F)/10000)
                forces.append(float(F))

        if self.force_scaler != None:
            forces = self.force_scaler.transform(np.array(forces).reshape(-1,1))
            forces = torch.from_numpy(forces).float().squeeze(0)
        else:
            forces = torch.tensor(forces)

        FEM_mises = []
        coords = []
        with open(f'{full_path}/Stress_Box_1.txt','r') as f:
            for i, line in enumerate(f):
                _, _, _, mises, x, y, z = line.rstrip('\n').split(',')
                FEM_mises.append(float(mises))
                coords.append([float(x), float(y), float(z)])

        if self.coords_scaler != None:
            x = [item[0] for item in coords]
            y = [item[1] for item in coords]
            z = [item[2] for item in coords]

            x = self.coords_scaler.transform(np.array(x).reshape(1,-1))
            y = self.coords_scaler.transform(np.array(y).reshape(1,-1))
            z = self.coords_scaler.transform(np.array(z).reshape(1,-1))

            coords = np.array([x,y,z])
            coords = torch.from_numpy(coords).float().squeeze(0)
        else:
            coords = torch.tensor(coords)

        FEM_mises = torch.tensor(FEM_mises).view(-1,1)
        coords = torch.tensor(coords)

        return forces, coords, FEM_mises

def main():

    dataset = GridDataset(split='validation')
    forces, coords, FEM_mises = dataset[6]

    vec_trans = torch.mean(coords, axis=0)
    coords = coords - vec_trans
    rot_mat = torch.tensor([[math.cos(math.pi/2), -math.sin(math.pi/2),0], [math.sin(math.pi/2), math.cos(math.pi/2),0],[0, 0,1]])
    coords = coords @ rot_mat.T

    coords = coords.cpu().squeeze().detach().numpy()
    max, min = coords.max().item(), coords.min().item()

    x, y, z = coords[:,0],coords[:,1],coords[:,2]

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(x,y,z, s=50, c = FEM_mises, cmap='viridis')

    ax.set_xlim(min,max)
    ax.set_ylim(min,max)
    ax.set_zlim(min,max)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()