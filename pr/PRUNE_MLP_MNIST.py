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
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from tqdm import tqdm
import json

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
l1_alpha = 0

finetune_steps = 10
prune_iters = 100

address ='./output/PRUNE_MNIST_no/'
if not os.path.exists(address):
    os.makedirs(address)
pre_name = 'PR_MNIST' + 'lr_'+ 'adam'+\
        '_learning_rate:'+ str(learning_rate) +\
        'batch_size:'+str(batch_size)+\
        '_num_epochs:'+ str(num_epochs) +\
        '_hidden_size:' +str(hidden_size) + \
        '_number_H:' + str(number_H) + \
        '_l1_alpha:' + str(l1_alpha)
f_name = 'PR_MNIST' + 'lr_'+ 'adam'+\
        '_learning_rate:'+ str(learning_rate) +\
        'batch_size:'+str(batch_size)+\
        '_num_epochs:'+ str(num_epochs) +\
        '_hidden_size:' +str(hidden_size) + \
        '_number_H:' + str(number_H) + \
        '_l1_alpha:' + str(l1_alpha) +\
        '_finetune_steps' + str(finetune_steps) +\
        'prune_iters' + str(prune_iters)

address_3 = address+f_name+'no_prune_cos_sim.txt'
address_4 = address+f_name+'no_prune_acc.txt'
address_5 = address+f_name+'no_sparsify.txt'

model_1 = './model/PRUNE_MNIST/'+pre_name
prune_checkpoint = address+f_name+'no_prune'

file3 = open(address_3,'w')
file4 = open(address_4,'w')
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
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

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

for name, param in model.named_parameters():
    print(name)

checkpoint_path = model_1 + 'checkpoint.pth'

checkpoint = torch.load(checkpoint_path)

model.load_state_dict(checkpoint['model_state_dict'])

parameters_to_prune = (
    (model.linear, 'weight'),
    (model.linearH[0], 'weight'),
    (model.linearH[1], 'weight'),
    (model.linearH[2], 'weight'),
    (model.linearH[3], 'weight'),
    (model.linearH[4], 'weight'),
    (model.out, 'weight'),
)

target_prune_rate = 0.99
save_every = 2

learning_rate = 0.0001

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
optimizer.zero_grad()

prune_rates = np.arange(0.05, 0.95, 0.05)
print(prune_rates)

def cos_similarity_pr(name):
    file3.writelines(f"{name}"+' ')
    cos_sim_row = cos_similarity_matrix_row(param.cpu().data)
    cos_sim_column = cos_similarity_matrix_column(param.cpu().data) 
    mean_cos_sim_row = round(Mean(cos_sim_row),6)
    mean_cos_sim_column= round(Mean(cos_sim_column),6)
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

text_accuracy, avg_testloss = compute_accuracy(model, test_loader)
text_accuracy = round(text_accuracy.item(),4)
train_accuracy, avg_trainloss = compute_accuracy(model, train_loader)
train_accuracy = round(train_accuracy.item(),4)
generalization_gap_acc = train_accuracy - text_accuracy
generalization_gap_loss = avg_trainloss - avg_testloss
print(str(text_accuracy)+'  '+str(avg_testloss)+'  '+str(0)+' '+str(train_accuracy)+' '+str(generalization_gap_acc)+' '+str(generalization_gap_loss))
file4.writelines(str(text_accuracy)+'  '+str(avg_testloss)+'  '+str(round(0,4))+' '+str(train_accuracy)+' '+str(generalization_gap_acc)+' '+str(generalization_gap_loss)+'\n')

count = 0
for prune_rate in tqdm(prune_rates):
    count += 1
    prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=prune_rate,
    )
    print(
        "Global sparsity: {:.2f}%".format(
            100. * float(
                torch.sum(model.linear.weight == 0)
                + torch.sum(model.linearH[0].weight == 0)
                + torch.sum(model.linearH[1].weight == 0)
                + torch.sum(model.linearH[2].weight == 0)
                + torch.sum(model.linearH[3].weight == 0)
                + torch.sum(model.linearH[4].weight == 0)
                + torch.sum(model.out.weight == 0)
            )
            / float(
                model.linear.weight.nelement()
                + model.linearH[0].weight.nelement()
                + model.linearH[1].weight.nelement()
                + model.linearH[2].weight.nelement()
                + model.linearH[3].weight.nelement()
                + model.linearH[4].weight.nelement()
                + model.out.weight.nelement()
            )
        )
    )

    if count % save_every == 0:
        checkpoint = prune_checkpoint + str(prune_rate)
    else:
        checkpoint = None

    for epoch in range(finetune_steps):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs= model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name) 

    text_accuracy, avg_testloss = compute_accuracy(model, test_loader)
    text_accuracy = round(text_accuracy.item(),4)
    train_accuracy, avg_trainloss = compute_accuracy(model, train_loader)
    train_accuracy = round(train_accuracy.item(),4)
    generalization_gap_acc = train_accuracy - text_accuracy
    generalization_gap_loss = avg_trainloss - avg_testloss
    print(str(text_accuracy)+'  '+str(avg_testloss)+'  '+str(prune_rate)+' '+str(train_accuracy)+' '+str(generalization_gap_acc)+' '+str(generalization_gap_loss))
    file4.writelines(str(text_accuracy)+'  '+str(avg_testloss)+'  '+str(round(prune_rate,4))+' '+str(train_accuracy)+' '+str(generalization_gap_acc)+' '+str(generalization_gap_loss)+'\n')
    
    file3.writelines('prune_rate:'+str(round(prune_rate,4))+'  ')
    for name, param in model.named_parameters():   
        if 'weight' in name:
            cos_similarity_pr(name)
    file3.writelines('\n')

    if checkpoint:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, prune_checkpoint+str(round(prune_rate,4))+'_checkpoint.pth')
    
    file5.writelines('prune_rate'+' '+str(round(prune_rate,4))+'  ')
    def compute_sparsity(weight_tensor):
        num_zero_elements = torch.sum(weight_tensor == 0)
        total_elements = weight_tensor.nelement()
        sparsity = 100. * float(num_zero_elements) / float(total_elements)
        return sparsity

    linear_sparsity = compute_sparsity(model.linear.weight)
    linearH_sparsities = [compute_sparsity(layer.weight) for layer in model.linearH]
    out_sparsity = compute_sparsity(model.out.weight)

    total_zero_elements = torch.sum(model.linear.weight == 0)
    total_elements = model.linear.weight.nelement()

    for layer in model.linearH:
        total_zero_elements += torch.sum(layer.weight == 0)
        total_elements += layer.weight.nelement()

    total_zero_elements += torch.sum(model.out.weight == 0)
    total_elements += model.out.weight.nelement()

    global_sparsity = 100. * float(total_zero_elements) / float(total_elements)
    file5.writelines(
        "linear.weight {:.2f}% "
        "linearH[0].weight {:.2f}% "
        "linearH[1].weight {:.2f}% "
        "linearH[2].weight {:.2f}% "
        "linearH[3].weight {:.2f}% "
        "linearH[4].weight {:.2f}% "
        "out.weight {:.2f}% "
        "global_sparsity {:.2f}%\n".format(
            linear_sparsity,
            *linearH_sparsities,
            out_sparsity,
            global_sparsity
        )
    )
    
    print(
        "Global sparsity: {:.2f}%".format(
            100. * float(
                torch.sum(model.linear.weight == 0)
                + torch.sum(model.linearH[0].weight == 0)
                + torch.sum(model.linearH[1].weight == 0)
                + torch.sum(model.linearH[2].weight == 0)
                + torch.sum(model.linearH[3].weight == 0)
                + torch.sum(model.linearH[4].weight == 0)
                + torch.sum(model.out.weight == 0)
            )
            / float(
                model.linear.weight.nelement()
                + model.linearH[0].weight.nelement()
                + model.linearH[1].weight.nelement()
                + model.linearH[2].weight.nelement()
                + model.linearH[3].weight.nelement()
                + model.linearH[4].weight.nelement()
                + model.out.weight.nelement()
            )
        )
    )
        
file3.close() 
file4.close()            
file5.close()