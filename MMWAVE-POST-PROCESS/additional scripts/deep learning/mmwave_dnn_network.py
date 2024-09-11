## Imports
# Torch Imports
print("Importing PyTorch")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.optim as optim


print("Checking CUDA...", end=" ")
print("CUDA v%s is availble" % torch.cuda_version if torch.cuda.is_available() else "CUDA not found...")
print()

# Other imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import glob

## Load data
# Get file directories
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
data_dir = os.path.join(script_dir,"canine_dnn_data")
simulated_datapaths = glob.glob(os.path.join(data_dir,"**","*simulated*.csv"), recursive=True)
# dog_datapaths = glob.glob(os.path.join(data_dir,"**","*exp*.csv"), recursive=True)

num_simulated_observations = len(simulated_datapaths)
print("Number of simulated observations: ",num_simulated_observations)
simulated_labels = torch.tensor([float(path.split("__")[1]) for path in simulated_datapaths]).float()

def get_file_indices(identifier,filepaths):
    return [i for i, x in enumerate([(identifier in path) for path in filepaths]) if x] 

# num_measured_observations = len(dog_datapaths)
# print("Number of canine observations: ",num_measured_observations)
# dog_labels = torch.tensor([float(path.split("__")[1]) for path in dog_datapaths]).float()

# bullet_idx = get_file_indices("Bullet",dog_datapaths)
# paddy_idx =  get_file_indices("Paddy",dog_datapaths)
# guster_idx = get_file_indices("Guster",dog_datapaths)
# sassy_idx =  get_file_indices("Sassy",dog_datapaths)
# kasey_idx =  get_file_indices("Kasey",dog_datapaths)


# define dataset
class HeartRateDataset(Dataset):
    def __init__(self, label_list, data_dirs, transform=None, target_transform=None):
        self.hr_labels = label_list
        self.data_dirs = data_dirs
        self.transform = transform
        self.target_transform = target_transform
        self.i=0

    def __len__(self):
        return len(self.hr_labels)

    def __getitem__(self, idx):
        self.i += 1
        csv_data = pd.read_csv(self.data_dirs[idx]).values.transpose()*1000 # put into millimetres
        # print(csv_data.shape)
        label = self.hr_labels[idx]
        
        if self.transform:
            tensor = self.transform(csv_data)
            # print(tensor.shape)
            sequence = torch.squeeze(tensor).float()
            # print(sequence.shape)
            # print(self.i)
            
        if self.target_transform:
            label = self.target_transform(label)
            
        return sequence, label

# create datasets

simulated_dataset = HeartRateDataset(simulated_labels,simulated_datapaths,ToTensor())
training_dataset, validation_dataset = random_split(simulated_dataset,[0.8,0.2])
# test_dataset =  HeartRateDataset(dog_labels,dog_datapaths,ToTensor())

# print(simulated_dataset.__sizeof__())
# print(training_dataset.dataset.__sizeof__())

# exit()

# create DataLoaders
batch_size = 64
train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print("Data has been loaded.\n")

## Define Network
class CANINet(nn.Module):

    def __init__(self):
        super(CANINet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv1d(1, 64, 5)
        self.conv2 = nn.Conv1d(64, 64, 5)
        self.conv3 = nn.Conv1d(64, 64, 5)
    
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(15616, 1024)  # 5*5 from image dimension
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 32)
        self.fc5 = nn.Linear(32, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = torch.reshape(x,[batch_size,1,256])
        x = self.conv1(x) # F.max_pool1d(F.relu(), 5)
        x = self.conv2(x) # F.max_pool1d(F.relu(), 5)
        x = self.conv3(x) #F.max_pool1d(F.relu(), 5)
        
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        x = self.fc5(x)# .squeeze()
        x = torch.reshape(x,[batch_size])
        x = torch.clamp(x,40.0,200.0)
        return x


net = CANINet()


criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.005) # , momentum=0.9

total_epochs= 10

CLR = "\x1b[1K"
print("Training...")
for epoch in range(total_epochs):  # loop over the dataset multiple times

    print("Epoch: %d" % (epoch+1))
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        net.train()
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print("                                                             ",end="\r")
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}',end="\r")


    print("\n")
    
    # print("Validating Epoch: %d" % (epoch+1))
    # valid_loss = 0.0
    # net.eval()     # Optional when not using Model Specific layer
    # j= 0
    # for data, labels in validation_dataloader:
    #     j+=1
    #     # if torch.cuda.is_available():
    #     #     data, labels = data.cuda(), labels.cuda()
    #     inputs, labels
        
        
    #     target = net(inputs)
    #     loss = criterion(target,labels)
        
    #     # print("Calculating Validation loss")
    #     valid_loss = loss.item() * inputs.size(0)
    #     print(j,end="\r")
    #     # print("Finished calculating Validation loss")
        
    
    # print(f'Epoch {epoch+1} \t\t Training Loss: {running_loss / len(train_dataloader)} \t\t Validation Loss: {valid_loss / len(validation_dataloader)}')
    # if min_valid_loss > valid_loss:
    #     print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
    #     min_valid_loss = valid_loss
    #     # Saving State Dict
    #     torch.save(net.state_dict(), 'saved_model.pth')

    print()
    
print('Finished Training')


