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
torch.cuda.set_device(1)
import torchvision.transforms as transforms
from scipy.spatial.distance import cosine
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from tqdm import tqdm
from thop import profile
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size = 768
num_epochs = 20
learning_rate = 0.0001
random_seed = 42
num_classes = 1000

epoch_interval = 50
save_interval = 50
l1_alpha = 0

finetune_steps = 1

address ='./output/PRUNE_RESNET18_IMAGENET_no/'
if not os.path.exists(address):
    os.makedirs(address)
pre_name = 'PRUNE_ResNet18_IMAGENET' + 'lr_' + 'adam'+'_learning_rate:'+ str(learning_rate) +'batch_size:'+str(batch_size)+'_num_epochs:'+ str(num_epochs) +'_l1_alpha:' + str(l1_alpha)
f_name = 'PRUNE_ResNet18' + 'lr_' + 'adam'+\
        '_learning_rate:'+ str(learning_rate) +\
        'batch_size:'+str(batch_size)+\
        '_num_epochs:'+ str(num_epochs) +\
        '_l1_alpha:' + str(l1_alpha) +\
        '_finetune_steps' + str(finetune_steps)

address_3 = address+f_name+'no_prune_cos_sim.txt'
address_4 = address+f_name+'no_prune_acc.txt'
address_5 = address+f_name+'no_sparsify.txt'
address_6 = address+f_name+'no_flops.txt'

model_1 = './model/PRUNE_RESNET18_IMAGENET/'+pre_name
prune_checkpoint = address+f_name+'no_prune'

file3 = open(address_3,'w')
file4 = open(address_4,'w')
file5 = open(address_5,'w')
file6 = open(address_6,'w')

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
            if torch.all(matrix[i] == 0) or torch.all(matrix[j] == 0):
                similarity_matrix[i, j] = 0
                similarity_matrix[j, i] = 0
            else:
                similarity_matrix[i, j] = 1 - cosine(matrix[i], matrix[j])
                similarity_matrix[j, i] = similarity_matrix[i, j]
    return abs(similarity_matrix)

def cos_similarity_matrix_column(matrix):
    num_column = matrix.shape[1]
    similarity_matrix = np.zeros((num_column, num_column))
    for i in range(num_column):
        for j in range(i, num_column):
            if torch.all(matrix[:,i] == 0) or torch.all(matrix[:,j] == 0):
                similarity_matrix[i, j] = 0
                similarity_matrix[j, i] = 0
            else:
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

data_dir = '/home/sda/luzhixing/datasets/imagenet'
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_dataset = datasets.ImageNet(root=data_dir+'/train', split='train', transform=data_transforms['train'])
val_dataset = datasets.ImageNet(root=data_dir+'/val', split='val', transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

model = models.resnet18()

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for name, param in model.named_parameters():
    print(name,param.data.size())

checkpoint_path = model_1 + "_best.pth"

checkpoint = torch.load(checkpoint_path)

model.load_state_dict(checkpoint['model_state_dict'])

parameters_to_prune = []

for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        parameters_to_prune.append((module, 'weight'))

print(parameters_to_prune)

target_prune_rate = 0.95
save_every = 30

learning_rate = 0.0001

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
optimizer.zero_grad()

prune_rates = np.arange(0.1, 1.0, 0.1)
print(prune_rates)

def cos_similarity_conv_pr(name):
    file3.writelines(f" {name}"+'  ')
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
    file3.writelines(str(mean_cos_sim_row)+'  '+ str(mean_cos_sim_column)+'  ')

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

count = 0
def count_total_nonzero_params(model):
    total_nonzero_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_nonzero_params += torch.count_nonzero(param).item()
    return total_nonzero_params

total_nonzero_params = count_total_nonzero_params(model)
input_data = torch.randn(512, 3, 32, 32).to(device)
flops, params = profile(model, inputs=(input_data,))
print(f"FLOPs: {flops}, Parameters: {total_nonzero_params}")
file6.writelines(f"FLOPs {flops}"+" "+f"Parameters {total_nonzero_params}\n")

def compute_sparsity(weight_tensor):
    num_zero_elements = torch.sum(weight_tensor == 0)
    total_elements = weight_tensor.nelement()
    sparsity = 100. * float(num_zero_elements) / float(total_elements)
    return sparsity

file3.writelines('prune_rate:' + str(0) + '  ')
for name, param in model.named_parameters():
    if name == 'layer1.0.conv1.weight':
        cos_similarity_conv_pr(name)
    if name == 'layer2.0.conv1.weight':
        cos_similarity_conv_pr(name)
    if name == 'layer3.0.conv1.weight':
        cos_similarity_conv_pr(name)
    if name == 'layer4.0.conv1.weight':
        cos_similarity_conv_pr(name)
file3.writelines('\n')

for prune_rate in tqdm(prune_rates):
    
    count += 1
    prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=prune_rate,
    )

    total_zero_elements = 0
    total_elements = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            total_zero_elements += torch.sum(module.weight == 0)
            total_elements += module.weight.nelement()

    global_sparsity = 100. * float(total_zero_elements) / float(total_elements)
    print(f"conv sparsity: {global_sparsity:.2f}%")

    for epoch in range(finetune_steps):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs= model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            loss_out=round(loss.item(),4)

    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name) 

    text_accuracy, avg_testloss = compute_accuracy(model, val_loader)
    text_accuracy = round(text_accuracy.item(),4)
    print(str(text_accuracy)+'  '+str(avg_testloss)+'  '+str(prune_rate))
    file4.writelines(str(text_accuracy)+'  '+str(avg_testloss)+'  '+str(round(prune_rate,4))+'\n')
    
    file3.writelines('prune_rate:'+str(round(prune_rate,4))+'  ')
    for name, param in model.named_parameters():
        if name == 'layer1.0.conv1.weight':
            cos_similarity_conv_pr(name)
        if name == 'layer2.0.conv1.weight':
            cos_similarity_conv_pr(name)
        if name == 'layer3.0.conv1.weight':
            cos_similarity_conv_pr(name)
        if name == 'layer4.0.conv1.weight':
            cos_similarity_conv_pr(name)
    file3.writelines('\n')

    file5.writelines('prune_rate'+' '+str(round(prune_rate,4))+'  ')

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            sparsity = compute_sparsity(module.weight)
            file5.writelines(name+".sparsity"+" "+str(sparsity)+"%"+" ")

    total_zero_elements = 0
    total_elements = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            total_zero_elements += torch.sum(module.weight == 0)
            total_elements += module.weight.nelement()

    flops, params = profile(model, inputs=(input_data,))
    total_nonzero_params = count_total_nonzero_params(model)
    print(f"FLOPs: {flops}, Parameters: {total_nonzero_params}")
    file6.writelines(f"FLOPs {flops}"+" "+f"Parameters {total_nonzero_params}\n")

    global_sparsity = 100. * float(total_zero_elements) / float(total_elements)
    print(f"Global sparsity: {global_sparsity:.2f}%")
    file5.writelines("global_sparsity"+" "+str(global_sparsity)+"%"+"\n")
        
file3.close() 
file4.close()            
file5.close()
