import wandb
from NN import NeuralNet
import torch
from dataset import GridDataset
from trainer import train
from torch.utils.data import DataLoader
from torch import autograd
import numpy as np
import random 
import wandb


# Fixed seed
seed = 11
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run():
    layer_sizes = [16,32,64,128,256,557]
    num_epochs = 2000
    batch_size = 1
    learning_rate = 0.001

    dict = {
        'learning_rate': learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size
    }

    use_existing_model = False

    # Dataset
    train_dataset = GridDataset()
    test_dataset = GridDataset(split="test")
    validation_dataset = GridDataset(split="validation")

    # Data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True, num_workers=6, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=6, pin_memory=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=6, pin_memory=True)

    model = NeuralNet(layer_sizes).to(device)
    if use_existing_model:
        model.load_state_dict(torch.load("./003model.pth")["state_dict"])
    wandb.init(project="FEM_case1", entity="master-thesis-ntnu", config=dict)

    train(model, num_epochs, batch_size, train_loader, test_loader, validation_loader, learning_rate=learning_rate, device=device)

    # Save model
    config = {
        "state_dict": model.state_dict()
    }

    torch.save(config, "model.pth")

if __name__ == "__main__":
    run()