#!/usr/bin/env python

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
from torch.autograd import Variable
import pickle
import numpy as np
import math
import torchvision

import torchvision.transforms as transforms
from scipy.spatial.distance import cosine

from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
torch.cuda.set_device(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 512
num_epochs = 250
learning_rate = 0.001
random_seed = 42

epoch_interval = 5
save_interval = 50
l1_alpha = 1e-3

address ='./output/RESNET_GTSRB_opt/'
if not os.path.exists(address):
    os.makedirs(address)

f_name = 'ResNet18' + 'lr_' + 'adam'+\
        '_learning_rate:'+ str(learning_rate) +\
        'batch_size:'+str(batch_size)+\
        '_num_epochs:'+ str(num_epochs) +\
        '_l1_alpha:' + str(l1_alpha)

address_1 = os.path.join(address, f_name+'_cos_sim.txt')
address_2 = os.path.join(address, f_name+'_acc.txt')
file1 = open(address_1,'w')
file2 = open(address_2,'w')

address_5 = os.path.join(address, f_name+'sparsify.txt')
file5 = open(address_5,'w')

def seed_torch(seed=random_seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False   
seed_torch()

def cos_similarity_matrix_row(matrix):
    num_rows = matrix.shape[0]
    similarity_matrix = np.zeros((num_rows, num_rows))
    for i in range(num_rows):
        for j in range(i, num_rows):
            similarity_matrix[i, j] = 1 - cosine(matrix[i], matrix[j])
            similarity_matrix[j, i] = similarity_matrix[i, j]
    return abs(similarity_matrix)

def cos_similarity_matrix_column(matrix):
    num_column = matrix.shape[1]
    similarity_matrix = np.zeros((num_column, num_column))
    for i in range(num_column):
        for j in range(i, num_column):
            similarity_matrix[i, j] = 1 - cosine(matrix[:,i], matrix[:,j])
            similarity_matrix[j, i] = similarity_matrix[i, j]
    return abs(similarity_matrix)

def Mean(matrix):
    number = matrix.shape[0]
    matrix_mean = matrix.mean()
    mean_out = abs((matrix_mean - (1/number))*(number/(number-1)))
    return mean_out

def Gram_matrix_row(matrix):
    matrix_transpose = np.transpose(matrix)
    Gram_matrix = np.dot(matrix,matrix_transpose)
    return Gram_matrix

def Gram_matrix_column(matrix):
    matrix_transpose = np.transpose(matrix)
    Gram_matrix = np.dot(matrix_transpose,matrix)
    return Gram_matrix

def cos_similarity_conv(name):
    file1.writelines(f" {name}"+'  ')
    cos_sim_row = cos_sim_column = 0.0
    for i in range(param.size(2)):
        for j in range(param.size(3)):
            param_i_j = param.data[:,:,i,j]
            cos_sim_row = cos_sim_row + cos_similarity_matrix_row(param_i_j .cpu().data)
            cos_sim_column =cos_sim_column + cos_similarity_matrix_column(param_i_j .cpu().data) 
    
    cos_sim_row = Mean(cos_sim_row) /(param.size(2)*param.size(3))
    cos_sim_column = Mean(cos_sim_column) /(param.size(2)*param.size(3))
    mean_cos_sim_row = round(cos_sim_row,6)
    mean_cos_sim_column= round(cos_sim_column,6)
    
    print(mean_cos_sim_row, mean_cos_sim_column)
    file1.writelines(str(mean_cos_sim_row)+'  '+ str(mean_cos_sim_column)+'  ')

def cos_similarity_ffn(name):
    print(f"Parameter name: {name}")
    file1.writelines(f"{name}"+' ')
    print(f"Parameter value: {param.data.size()}")  
    cos_sim_row = cos_similarity_matrix_row(param.cpu().data)
    cos_sim_column = cos_similarity_matrix_column(param.cpu().data) 
    mean_cos_sim_row = round(Mean(cos_sim_row),6)
    mean_cos_sim_column= round(Mean(cos_sim_column),6)
    print(mean_cos_sim_row, mean_cos_sim_column)
    file1.writelines(str(mean_cos_sim_row)+'  '+ str(mean_cos_sim_column)+'  ')
    print('='*50)

train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.GTSRB(root='/home/sda/zhouqin/sparity/data/GTSRB', split='train', download=True, transform=train_transform)

train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

test_dataset = torchvision.datasets.GTSRB(root='/home/sda/zhouqin/sparity/data/GTSRB', split='test', download=True, transform=val_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=43):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion*4, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

if torch.cuda.is_available():
    model = ResNet18().cuda()
else:
    model = ResNet18()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
optimizer.zero_grad()

for name, param in model.named_parameters():
    print(name)

file1.writelines('epoch:'+str(0)+'  ')
for name, param in model.named_parameters():
    if name == 'layer1.0.conv1.weight':
        cos_similarity_conv(name)
    if name == 'layer2.0.conv1.weight':
        cos_similarity_conv(name) 
    if name == 'layer3.0.conv1.weight':
        cos_similarity_conv(name) 
        file1.writelines('\n') 

def compute_accuracy(model, data_loader):
    model.eval()
    running_loss = 0.0
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for i,(features, targets) in enumerate(data_loader):
            features = features.to(device)
            targets = targets.to(device)
            targets = Variable(targets).cuda()
            probas = model(features)

            test_loss = criterion(probas, targets)
            running_loss += test_loss.item()

            _, predicted_labels = torch.max(probas, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()

    return correct_pred.float()/num_examples * 100, running_loss / len(data_loader)

def compute_sparsity(weight_tensor, threshold=1e-4):
    num_insignificant = torch.sum(torch.abs(weight_tensor) < threshold)
    total_elements = weight_tensor.nelement()
    sparsity = 100. * float(num_insignificant) / float(total_elements)
    return sparsity

def calculate_global_sparsity(model, threshold=1e-4):
    insignificant_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            insignificant_params += torch.sum(torch.abs(param) < threshold).item()
            total_params += param.numel()
    return insignificant_params / total_params

file5.writelines(str(0)+'  ')
for name, param in model.named_parameters():
    if 'weight' in name:
        sparsity = compute_sparsity(param)
        file5.writelines(name+".sparsity"+" "+str(sparsity)+"%"+" ")
global_sparsity = 100. * calculate_global_sparsity(model)
print(f"Global sparsity: {global_sparsity:.2f}%")
file5.writelines("global_sparsity"+" "+str(global_sparsity)+"%"+"\n")

for epoch in tqdm(range(num_epochs)):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs= model(images)
        loss = criterion(outputs, labels)
        if l1_alpha:
            l1_loss = 0
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l1_loss += torch.sum(torch.abs(param))
            loss += l1_alpha * l1_loss
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()        

        loss_out=loss.item()
        
    print('Epoch: [%d/%d], Loss:%.4f ' % (epoch+1, num_epochs, loss.item()))
    text_accuracy, avg_testloss = compute_accuracy(model, test_loader)
    text_accuracy = round(text_accuracy.item(),4)
    print(str(text_accuracy)+'  '+str(loss_out)+'  '+str(avg_testloss))
    file2.writelines(str(text_accuracy)+'  '+str(loss_out)+'  '+str(avg_testloss)+'\n')

    if (epoch+1) % epoch_interval == 0:
        file1.writelines('epoch:'+str(epoch)+'  ')
        for name, param in model.named_parameters():   
            if name == 'layer1.0.conv1.weight':
                cos_similarity_conv(name)
            if name == 'layer2.0.conv1.weight':
                cos_similarity_conv(name) 
            if name == 'layer3.0.conv1.weight':
                cos_similarity_conv(name) 
                file1.writelines('\n')

    file5.writelines(str(epoch)+'  ')
    for name, param in model.named_parameters():
        if 'weight' in name:
            sparsity = compute_sparsity(param)
            file5.writelines(name+".sparsity"+" "+str(sparsity)+"%"+" ")
    global_sparsity = 100. * calculate_global_sparsity(model)
    print(f"Global sparsity: {global_sparsity:.2f}%")
    file5.writelines("global_sparsity"+" "+str(global_sparsity)+"%"+"\n")

def plot_weight_histogram(model, save_path,bins=100):
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            abs_weights = torch.abs(param.data).view(-1)
            all_weights.append(abs_weights)

    if not all_weights:
        print("No weights found in model.")
        return

    all_weights = torch.cat(all_weights).cpu().numpy()

    plt.figure(figsize=(8, 5))
    plt.hist(all_weights, bins=bins, log=True, color='steelblue', edgecolor='black')
    plt.title("Histogram of Absolute Weight Values")
    plt.xlabel("Absolute Weight Value")
    plt.ylabel("Frequency (log scale)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

plot_weight_histogram(model,os.path.join(address, f_name+f"final_hist.png"))

save_path = os.path.join(address, f_name+f"final.pth")
torch.save(model.state_dict(), save_path)
print(f"Model saved at final, path: {save_path}")

file1.close() 
file2.close()
file5.close()
