import torch
import os
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
import pickle
import numpy as np
import math
import torchvision
torch.cuda.set_device(1)
import torchvision.transforms as transforms
from scipy.spatial.distance import cosine

from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 28 * 28
batch_size = 512
num_epochs = 150
learning_rate = 0.001
hidden_size = 512
number_H =5
random_seed = 42

epoch_interval = 1
save_interval = 50
l1_alpha = 1e-15

# 使用相对路径存储输出文件
address ='./output/'
if not os.path.exists(address):
    os.makedirs(address)
    
f_name = 'exp_' + 'lr_'+ 'adam'+\
        '_learning_rate:'+ str(learning_rate) +\
        'batch_size:'+str(batch_size)+\
        '_num_epochs:'+ str(num_epochs) +\
        '_hidden_size:' +str(hidden_size) + \
        '_number_H:' + str(number_H) + \
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

def cos_similarity(name):
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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)  

train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

test_dataset = torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class feedforward_neural_network(nn.Module):
    def __init__(self, input_size, hidden, num_classes):
        super(feedforward_neural_network, self).__init__()
        self.linear = nn.Linear(input_size, hidden)
        self.r = nn.ReLU()
        self.hidden = hidden
        self.linearH = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(number_H)])
        self.out = nn.Linear(hidden, num_classes)
    
    def forward(self, x):
        x = x.view(-1, input_size)
        x = self.linear(x)
        x = self.r(x)
        
        for i in  range(number_H):
            x = self.linearH[i](x)
            x = self.r(x) 

        out = self.out(x)
        return out

if torch.cuda.is_available():
    model = feedforward_neural_network(input_size = input_size, hidden = hidden_size, num_classes = 10).cuda()
else:
    model = feedforward_neural_network(input_size = input_size, hidden = hidden_size, num_classes = 10)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
optimizer.zero_grad()

for name, param in model.named_parameters():
    print(name)

file1.writelines('epoch:'+str(0)+'  ')
for name, param in model.named_parameters():
    if name == 'linearH.0.weight':
        cos_similarity(name)
    if name == 'linearH.1.weight':
        cos_similarity(name)
    if name == 'linearH.2.weight':
        cos_similarity(name)
    if name == 'linearH.3.weight':
        cos_similarity(name)        
    if name == 'linearH.4.weight':
        cos_similarity(name)    
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
    total_loss = 0.0
    correct = 0
    total = 0
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

        loss_out=round(loss.item(),4)
        if (i+1) % 40 == 0:
            print('Epoch: [%d/%d],Step:[%d/%d],Loss:%.4f ' % 
                  (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size,loss.item()))
        
    text_accuracy, avg_testloss = compute_accuracy(model, test_loader)
    text_accuracy = round(text_accuracy.item(),4)
    print(str(text_accuracy)+'  '+str(loss_out)+'  '+str(avg_testloss))
    file2.writelines(str(text_accuracy)+'  '+str(loss_out)+'  '+str(avg_testloss)+'\n')
    if (epoch+1) % epoch_interval == 0:
        file1.writelines('epoch:'+str(epoch)+'  ')
        for name, param in model.named_parameters():   
            if name == 'linearH.0.weight':
                cos_similarity(name)
            if name == 'linearH.1.weight':
                cos_similarity(name)
            if name == 'linearH.2.weight':
                cos_similarity(name)
            if name == 'linearH.3.weight':
                cos_similarity(name)        
            if name == 'linearH.4.weight':
                cos_similarity(name)
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

# 使用相对路径保存结果文件
plot_weight_histogram(model, os.path.join(address, f_name+f"final_hist.png"))

save_path = os.path.join(address, f_name+f"final.pth")
torch.save(model.state_dict(), save_path)
print(f"Model saved at final, path: {save_path}")

file1.close() 
file2.close()
file5.close()
