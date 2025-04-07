import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
import os
import time
from sksurv.metrics import concordance_index_censored

#grade evalaution
from sklearn.metrics import roc_auc_score,precision_score,f1_score,average_precision_score
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F
import torch.nn as nn

def train_loop_survival_coattn(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16, args=None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    
    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c, grade) in enumerate(loader):
        #continue
        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)

        hazards, S, Y_hat, A, _ = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 100 == 0:
            train_batch_str = 'batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}'.format(
                batch_idx, loss_value, label.item(), float(event_time), float(risk))
            with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
                f.write(train_batch_str+'\n')
            f.close()
            print(train_batch_str)
        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()
    #return
    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    c_index_train = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    
    train_epoch_str = 'Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(
        epoch, train_loss_surv, train_loss, c_index_train)
    print(train_epoch_str)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(train_epoch_str+'\n')
    f.close()

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index_train, epoch)


def validate_survival_coattn(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, args=None):
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c, grade) in enumerate(loader):

        data_WSI = data_WSI.cuda()
        data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
        data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
        data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
        data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
        data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
        data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
        label = label.type(torch.LongTensor).cuda()
        c = c.type(torch.FloatTensor).cuda()

        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            hazards, S, Y_hat, A, _ = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6) # return hazards, S, Y_hat, A_raw, 

        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(), 'survival': event_time.item(), 'censorship': c.item()}})

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg


    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    val_epoch_str = "val c-index: {:.4f}".format(c_index)
    print(val_epoch_str)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(val_epoch_str+'\n')
    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_surv, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return patient_results, c_index, True

    return patient_results, c_index, False

def train_loop_grade_coattn(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None,
                               lambda_reg=0.3, gc=16, args=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    # input(device)
    model.to(device)
    model.train()
    train_loss_grade, train_loss = 0., 0.
    grad_acc_epoch=0
    
    #debugging
    #print(loader)
    #sys.exit(0)
    
    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time,c,grade) in enumerate(loader):
        #print(f"event_time: {event_time}")
        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)

        label = label.type(torch.LongTensor).to(device)
        grade = grade.type(torch.LongTensor).to(device)
        grade=grade-2
        c = c.type(torch.FloatTensor).to(device)


        hazards, S, Y_hat, A, hazard_grade = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
        
        loss= F.nll_loss(hazard_grade, grade)
        #cross=nn.CrossEntropyLoss()
        #loss=cross(hazard_grade,grade)
        loss_value = loss.item()
        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        hazard_grade = hazard_grade.argmax(dim=1, keepdim=True)
        '''
        check[batch_idx].append(grade.item())
        check[batch_idx].append(hazard_grade.item())
        check[batch_idx].append('\n')
        '''
        grad_acc_epoch += hazard_grade.eq(grade.view_as(hazard_grade)).sum().item()

        train_loss_grade += loss_value
        train_loss += loss_value + loss_reg
    
        if (batch_idx + 1) % len(loader) == 0:
            print('batch {}, loss: {:.4f}, grade_true: {}, event_time: {:.4f}, grade_pred: {:.4f}'.format(batch_idx,
                                                                                                          loss_value + loss_reg,
                                                                                                          grade.item(),
                                                                                                          float(event_time),
                                                                                                          hazard_grade.item()))
        loss = loss/gc  + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0:
            optimizer.step()
            optimizer.zero_grad()

    
    # calculate loss and error for epoch
    train_loss_grade /= len(loader)
    train_loss /= len(loader)
    acc=grad_acc_epoch/(len(loader))
    #c_index = \
    #concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    print('Epoch: {}, train_loss_grade: {:.4f}, train_loss: {:.4f}, acc: {:.4f}'.format(epoch, train_loss_grade,
                                                                                                 train_loss, acc))
    if writer:
        writer.add_scalar('train/loss_surv', train_loss_grade, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', acc, epoch)



def validate_grade_coattn(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, args=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    #test_loss = 0.
    grad_acc_epoch=0
    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    
    val_loss_grade, val_loss = 0., 0.
    grad_acc_epoch=0
    
    grade_true=[]
    grade_pred=[]
    grade_pred_sum=[]

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time,
                    c,grade) in enumerate(loader):
        #print(label, event_time, c)
        #df = df.append({'case_id': case_id, 'dis_survival_month': label.item()}, ignore_index=True)
        data_WSI = data_WSI.to(device)
        #print(f"case id: {data_WSI.shape}")
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)
        grade=grade.type(torch.LongTensor).to(device)
        slide_id = slide_ids.iloc[batch_idx]
        grade=grade-2
        with torch.no_grad():
            hazards, S, Y_hat, A, hazards_grade = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2,
                                                x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5,
                                                x_omic6=data_omic6)  # return hazards, S, Y_hat, A_raw, results_dict

        
        loss= F.nll_loss(hazards_grade, grade)
        loss_value = loss.item()
        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        soft=nn.Softmax(dim=1)
        hazards_grade=soft(hazards_grade)
        grade_true.append(grade)
        grade_pred.append(hazards_grade)
        
        hazards_grade = hazards_grade.argmax(dim=1, keepdim=True) 

        grad_acc_epoch += hazards_grade.eq(grade.view_as(hazards_grade)).sum().item()
        grade_pred_sum.append(hazards_grade.item())

        val_loss_grade += loss_value
        val_loss += loss_value + loss_reg


    val_loss_grade /= len(loader)
    val_loss /= len(loader)
    acc=grad_acc_epoch/len(loader)
    if writer:
        writer.add_scalar('val/loss_grade', val_loss_grade, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', acc, epoch)
    try:
        grade_true=torch.cat(grade_true,dim=0).detach().cpu().numpy()
        grade_pred=torch.cat(grade_pred,dim=0).detach().cpu().numpy()
        
        grade_true_bin = label_binarize(grade_true, classes=[0, 1, 2])
        micro_auc = roc_auc_score(grade_true_bin, grade_pred,multi_class='ovr',average='micro')
        micro_ap=average_precision_score(grade_true_bin, grade_pred,average='micro')
        micro_f1=f1_score(grade_true ,grade_pred_sum,average='micro')
        acc=grad_acc_epoch/len(loader)
        print("acc:",acc,' micro_auc:',micro_auc,' micro_ap:',micro_ap,' micro_f1:',micro_f1,'\n')
    except ValueError as V:
        print(V)
    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_grade, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return patient_results, acc, micro_auc, micro_ap, micro_f1, True
            
    return patient_results, acc, micro_auc, micro_ap, micro_f1, False