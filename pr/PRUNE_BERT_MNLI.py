import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch.nn.functional as F
import os
import pickle
import numpy as np
import math
from scipy.spatial.distance import cosine

from sklearn.metrics import accuracy_score
from tqdm import tqdm
from thop import profile
import torch.nn.utils.prune as prune
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


batch_size = 32
num_epochs = 2
learning_rate = 2e-6
random_seed = 42

epoch_interval = 1
save_interval = 100
l1_alpha = 0

save_every = 1

address = './output/PRUNE_BERT_mnli_no/'
if not os.path.exists(address):
    os.makedirs(address)
f_name = 'PR_BERT'+'lr_'+ 'adam'+\
        '_learning_rate:'+ str(learning_rate) +\
        'batch_size:'+str(batch_size)+\
        '_num_epochs:'+ str(num_epochs) +\
        '_l1_alpha:' + str(l1_alpha)

address_3 = os.path.join(address, f_name+'no_prune_cos_sim.txt')
address_4 = os.path.join(address, f_name+'no_prune_acc.txt')
address_5 = os.path.join(address, f_name+'no_sparsify.txt')
address_6 = os.path.join(address, f_name+'no_flops.txt')

model_1 = './model/saved_bert_model_weights.pth'
prune_checkpoint = os.path.join(address, f_name+'no_prune')

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


dataset = load_dataset('./data/mnli',data_files={'train': './data/mnli/0.0.0/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/glue-train.arrow','validation': './data/mnli/0.0.0/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/glue-validation_matched.arrow','test': './data/mnli/0.0.0/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/glue-test_matched.arrow'})
print(dataset)

tokenizer = BertTokenizer.from_pretrained('./model/bert-base-uncased/')

def preprocess_function(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length', max_length=512, return_tensors='pt')

encoded_dataset = dataset.map(preprocess_function, batched=True)
for key in encoded_dataset.keys():
    print(key)
train_dataset = encoded_dataset['train']
val_dataset = encoded_dataset['validation']
test_dataset = encoded_dataset['test']


train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=32)


model = BertForSequenceClassification.from_pretrained('./model/bert-base-uncased/', num_labels=3)

model.to(device)

checkpoint_path = model_1
checkpoint = torch.load(checkpoint_path)

model.load_state_dict(checkpoint)
parameters_to_prune = []

for ii in range(12):
    layer = model.bert.encoder.layer[ii]

    parameters_to_prune.append((layer.attention.self.query, 'weight'))
    parameters_to_prune.append((layer.attention.self.key, 'weight'))
    parameters_to_prune.append((layer.attention.self.value, 'weight'))
    parameters_to_prune.append((layer.attention.output.dense, 'weight'))

    parameters_to_prune.append((layer.intermediate.dense, 'weight'))
    parameters_to_prune.append((layer.output.dense, 'weight'))

print(parameters_to_prune)

prune_rates = np.arange(0.1, 1.0, 0.1)
print(prune_rates)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = torch.stack(batch['input_ids'], dim=1).to(device)
            attn_mask = torch.stack(batch['attention_mask'], dim=1).to(device)
            token_type_ids = torch.stack(batch['token_type_ids'], dim=1).to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attn_mask,
                            labels=labels)

            loss, logits = outputs.loss, outputs.logits

            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=-1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    accuracy = accuracy_score(all_labels, all_predictions)

    return avg_loss, accuracy

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

count = 0
def count_total_nonzero_params(model):
    total_nonzero_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_nonzero_params += torch.count_nonzero(param).item()
    return total_nonzero_params

total_nonzero_params = count_total_nonzero_params(model)
input_data = torch.randint(0, 1000, (512, 512)).to(device)
flops, params = profile(model, inputs=(input_data,))
print(f"FLOPs: {flops}, Parameters: {total_nonzero_params}")
file6.writelines(f"FLOPs {flops}"+" "+f"Parameters {total_nonzero_params}\n")

def compute_sparsity(weight_tensor):
    """计算单个权重矩阵的稀疏度"""
    num_zero_elements = torch.sum(weight_tensor == 0)
    total_elements = weight_tensor.nelement()
    sparsity = 100. * float(num_zero_elements) / float(total_elements)
    return sparsity

def cos_similarity_ffn_pr(name):
    print(f"Parameter name: {name}")
    file3.writelines(f"{name}"+' ')
    print(f"Parameter value: {param.data.size()}")
    cos_sim_row = cos_similarity_matrix_row(param.cpu().data)
    cos_sim_column = cos_similarity_matrix_column(param.cpu().data)
    mean_cos_sim_row = round(Mean(cos_sim_row),6)
    mean_cos_sim_column= round(Mean(cos_sim_column),6)
    print(mean_cos_sim_row, mean_cos_sim_column)
    file3.writelines(str(mean_cos_sim_row)+'  '+ str(mean_cos_sim_column)+'  ')
    print('='*50)

file3.writelines('prune_rate:'+str(0)+'  ')
for name, param in model.named_parameters():
    if name == 'bert.encoder.layer.11.attention.self.query.weight':
        cos_similarity_ffn_pr(name)
    if name == 'bert.encoder.layer.11.attention.self.key.weight':
        cos_similarity_ffn_pr(name)
    if name == 'bert.encoder.layer.11.attention.self.value.weight':
        cos_similarity_ffn_pr(name)
    if name == 'bert.encoder.layer.11.attention.output.dense.weight':
        cos_similarity_ffn_pr(name)
    if name == 'bert.encoder.layer.11.intermediate.dense.weight':
        cos_similarity_ffn_pr(name)
file3.writelines('\n')

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
iter_count = 0


for prune_rate in prune_rates:
    count += 1
    prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=prune_rate,
    )

    if count % save_every == 0:
        checkpoint = prune_checkpoint + str(prune_rate)
    else:
        checkpoint = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader):
            iter_count += 1

            input_ids, attn_mask, token_type_ids, label = torch.stack(batch['input_ids'],dim=1).to(device), torch.stack(batch['attention_mask'],dim=1).to(device), torch.stack(batch['token_type_ids'],dim=1).to(device), batch['label'].to(device)

            loss, prediction = model(input_ids,
                                        token_type_ids=token_type_ids,
                                        attention_mask=attn_mask,
                                        labels=label).values()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter_count % save_interval == 0:
                print(f"Iterations: {iter_count}- Training Loss: {loss.item():.4f}")
        train_loss = loss.item()


    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)

    avg_loss, accuracy = evaluate(model, val_dataloader, device)

    print(f"Prune Rate: {prune_rate:.4f},Test Loss: {avg_loss:.4f},Validation ACC: {accuracy:.4f}")
    file4.writelines(str(avg_loss)+' '+str(accuracy)+'  '+str(round(prune_rate,4))+'\n')

    file3.writelines('prune_rate:'+str(round(prune_rate,4))+'  ')
    for name, param in model.named_parameters():
        if name == 'bert.encoder.layer.11.attention.self.query.weight':
            cos_similarity_ffn_pr(name)
        if name == 'bert.encoder.layer.11.attention.self.key.weight':
            cos_similarity_ffn_pr(name)
        if name == 'bert.encoder.layer.11.attention.self.value.weight':
            cos_similarity_ffn_pr(name)
        if name == 'bert.encoder.layer.11.attention.output.dense.weight':
            cos_similarity_ffn_pr(name)
        if name == 'bert.encoder.layer.11.intermediate.dense.weight':
            cos_similarity_ffn_pr(name)
    file3.writelines('\n')

    file5.writelines('prune_rate'+' '+str(round(prune_rate,4))+'  ')

    for name, module in model.named_modules():
        if 'bert.encoder.layer' in name:
            if isinstance(module, torch.nn.ModuleList):
                for submodule in module:
                    if hasattr(submodule, 'weight') and submodule.weight.dim() == 2:
                        sparsity = compute_sparsity(submodule.weight)
                        file5.writelines(name+".sparsity"+" "+str(sparsity)+"%"+" ")
            elif hasattr(module, 'weight') and module.weight.dim() == 2:
                sparsity = compute_sparsity(module.weight)
                file5.writelines(name+".sparsity"+" "+str(sparsity)+"%"+" ")


    total_zero_elements = 0
    total_elements = 0
    for name, module in model.named_modules():
        if 'bert.encoder.layer' in name:
            if isinstance(module, torch.nn.ModuleList):
                for submodule in module:
                    if hasattr(submodule, 'weight') and submodule.weight.dim() == 2:
                        total_zero_elements += torch.sum(module.weight == 0)
                        total_elements += module.weight.nelement()
            elif hasattr(module, 'weight') and module.weight.dim() == 2:
                total_zero_elements += torch.sum(module.weight == 0)
                total_elements += module.weight.nelement()

    flops, params = profile(model, inputs=(input_data,))
    total_nonzero_params = count_total_nonzero_params(model)
    print(f"FLOPs: {flops}, Parameters: {total_nonzero_params}")
    file6.writelines(f"FLOPs {flops}"+" "+f"Parameters {total_nonzero_params}\n")

    global_sparsity = 100. * float(total_zero_elements) / float(total_elements)
    print(f"Global sparsity: {global_sparsity:.2f}%")
    file5.writelines("global_sparsity"+" "+str(global_sparsity)+"%"+"\n")

    torch.save(model.state_dict(), prune_checkpoint+str(round(prune_rate,4))+'_checkpoint.pth')


file3.close()
file4.close()
file5.close()

