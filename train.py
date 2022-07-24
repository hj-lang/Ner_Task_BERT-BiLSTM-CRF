import logging
import os,sys
from time import strftime,localtime
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

from utiles import read_file,convert_format
from model import Bert_BiLSTM_CRF

import torch
from torch.utils.data import TensorDataset,DataLoader,SequentialSampler,RandomSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

path_file = os.path.dirname(os.path.abspath(__file__))
path_next = os.path.join(path_file,'logs')
if not os.path.exists(path_next):
    os.makedirs(path_next)
time = '{}'.format(strftime("%y%m%d-%H%M%S", localtime()))
log_file_name = '{}.log'.format(time)
log_file = os.path.join(path_next,log_file_name)
logger.addHandler(logging.FileHandler(log_file))
logger.info('log file: {}'.format(log_file))


#验证集查看损失和准确率，F1等指标
def evaluate(epoch,step,model,valid_loader,device):
    model.eval()
    y_true,y_pred = [],[]
    valid_losses = 0
    index = 0
    with torch.no_grad():
        for batch in tqdm(valid_loader,desc='Valid Model'):
            inputs,targets,mask = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            y_pred_temp = model(inputs,targets,mask,is_test=True)
            valid_loss = model(inputs,targets,mask)
            valid_losses += valid_loss.item()
            for j in y_pred_temp:
                y_pred.extend(j)
            mask = (mask==1)
            y_prob = torch.masked_select(targets,mask)
            y_true.append(y_prob.cpu())
            index +=1
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = np.array(y_pred)
    acc = accuracy_score(y_true,y_pred)*100

    # print("Epoch: {}, Step:{},Val Loss:{:.4f}, Val Acc:{:.3f}%".format(epoch,step, valid_losses/index, acc))  #需要其他指标
    logger.info("Epoch: {}, Step:{},Val Loss:{:.4f}, Val Acc:{:.3f}".format(epoch,step, valid_losses/index, acc))
    return model,valid_losses/index,acc

def train(epochs,model,train_loader,valid_loader,optimizer,scheduler,device):
    best_acc = 0
    best_loss = 0
    best_model = None
    model.train()
    for epoch in range(1,epochs+1):
        train_losses = 0
        model.train()
        index = 0
        for step,batch in tqdm(enumerate(train_loader,1),desc='Training Model'):
            inputs,targets,mask = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            train_loss = model(inputs,targets,mask)
            train_losses += train_loss.item()
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            index += 1
            # if step % 100 ==0:
            #     print('Epoch:{},Step:{},Train_Loss:{:.4f}'.format(epoch,step,train_losses/index))
            #     index = 0
        # print('Epoch:{},Step:{},Train_Loss:{:.4f}'.format(epoch,step,train_losses/index))
        logger.info('Epoch:{},Step:{},Train_Loss:{:.4f}'.format(epoch,step,train_losses/index))
        model,valid_loss,valid_acc = evaluate(epoch,step,model,valid_loader,device)
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_loss = valid_loss
            best_model = model
    return best_model
        

def test(model,test_loader,device):
    model.eval()
    y_true,y_pred = [],[]
    test_losses = 0
    index = 0
    with torch.no_grad():
        for batch in tqdm(test_loader,desc="Test Model"):
            inputs,targets,mask = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            y_pred_temp = model(inputs,targets,mask,is_test=True)
            test_loss = model(inputs,targets,mask)
            test_losses += test_loss.item()
            for j in y_pred_temp:
                y_pred.extend(j)
            mask = (mask==1)
            y_prob = torch.masked_select(targets,mask)
            y_true.append(y_prob.cpu())
            index +=1
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = np.array(y_pred)
    acc = accuracy_score(y_true,y_pred)*100
    logger.info("Test Loss:{:.4f}, Test Acc:{:.3f}".format(test_losses/index, acc))
    return test_losses/index,acc
def save(best_model):
    model_dir = os.path.join(path_file,'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir,'model.pt')
    torch.save(best_model,model_path)
    logger.info('Model Path:{}'.format(model_path))

if __name__=='__main__':
    #参数设置
    max_len = 256
    label_list = ['[PAD]','[CLS]', '[SEP]', 'O', 'B-BODY','I-TEST', 'I-EXAMINATIONS',
                'I-TREATMENT', 'B-DRUG', 'B-TREATMENT', 'I-DISEASES', 'B-EXAMINATIONS',
                    'I-BODY', 'B-TEST', 'B-DISEASES', 'I-DRUG']
    label_map = {value:idx for idx,value in enumerate(label_list)}
    label_map_reverse = {idx:value for idx,value in enumerate(label_list)}
    model_name = 'bert-base-chinese'
    train_batch_size = 16
    valid_batch_size = 8
    test_batch_size = 8
    lr  = 3e-5
    num_epoch = 100


    #获取数据
    train_data = read_file('../ner_task/data/chinese_medical/train.txt')
    valid_data = read_file('../ner_task/data/chinese_medical/valid.txt')
    test_data = read_file('../ner_task/data/chinese_medical/test.txt')
    logger.info('Train Data Num: {}'.format(len(train_data)))
    logger.info('Valid Data Num: {}'.format(len(valid_data)))
    logger.info('Test Data Num: {}'.format(len(test_data)))

    train_token,train_label = convert_format(train_data,max_len,label_map,model_name)
    valid_token,valid_label = convert_format(valid_data,max_len,label_map,model_name)
    test_token,test_label = convert_format(test_data,max_len,label_map,model_name)

    train_token_tensor = torch.tensor(train_token)
    train_label_tensor = torch.tensor(train_label)
    train_mask = (train_token_tensor!=0)
    valid_token_tensor = torch.tensor(valid_token)
    valid_label_tensor = torch.tensor(valid_label)
    valid_mask = (valid_token_tensor!=0)
    test_token_tensor = torch.tensor(test_token)
    test_label_tensor = torch.tensor(test_label)
    test_mask = (test_token_tensor!=0)

    train_set = TensorDataset(train_token_tensor,train_label_tensor,train_mask)
    valid_set = TensorDataset(valid_token_tensor,valid_label_tensor,valid_mask)
    test_set = TensorDataset(test_token_tensor,test_label_tensor,test_mask)


    train_sampler = SequentialSampler(train_set)
    train_dataloader = DataLoader(train_set, sampler=train_sampler, batch_size=train_batch_size,)

    valid_sampler = RandomSampler(valid_set)
    valid_dataloader = DataLoader(valid_set,sampler=valid_sampler,batch_size=valid_batch_size)

    test_sampler = RandomSampler(test_set)
    test_dataloader = DataLoader(test_set,sampler=test_sampler,batch_size=test_batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Bert_BiLSTM_CRF(len(label_list))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-6)
    warm_up_ratio = 0.1 
    len_dataset = len(train_data)
    total_steps = (len_dataset // train_batch_size) * num_epoch if len_dataset % train_batch_size == 0 else (len_dataset // train_batch_size + 1) * num_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warm_up_ratio * total_steps, num_training_steps = total_steps)
    best_model = train(num_epoch,model,train_dataloader,valid_dataloader,optimizer,scheduler,device)
    test_loss,test_acc = test(best_model,test_dataloader,device)
    save(best_model)













