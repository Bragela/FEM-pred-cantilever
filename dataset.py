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


class GridDataset(Dataset):
    def __init__(self, root_dir="data", split="train"):
        self.data_path = f"{root_dir}/{split}" 
        self.data = [folder for folder in os.listdir(self.data_path)]
        self.split = split

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
                forces.append(float(F))
        forces = torch.tensor(forces)

        FEM_mises = []
        coords = []

        with open(f'{full_path}/Stress_Box_1.txt','r') as f:
            for i, line in enumerate(f):
                _, _, _, mises, x, y, z = line.rstrip('\n').split(',')
                FEM_mises.append(float(mises))
                coords.append([float(x), float(y), float(z)])

        FEM_mises = torch.tensor(FEM_mises)
        coords = torch.tensor(coords)
        
        return forces, coords, FEM_mises


def main():
    dataset = GridDataset()
    forces, coords, FEM_mises = dataset[0]

    coords = np.array(coords)
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