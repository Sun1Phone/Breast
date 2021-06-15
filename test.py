import os
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
import torchvision.models as models
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn import model_selection
import matplotlib.pyplot as plt
import warnings
import logging
import sys

from MyDataset import MyDataset
from newRes_2_2 import newresnet50_2
from newRes_2_4 import newresnet50_4
from newRes_1_2 import resnet50_2
from ResNet import resnet50

warnings.filterwarnings("ignore")

# def datatotensor(path,batch_size):
#     data_transform = transforms.Compose([
#     transforms.Resize((224, 224), 3),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
#                              0.229, 0.224, 0.225])
#          ])
#     dataset = MyDataset(path, transform = data_transform)
#     loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
#
#     return loader
data_transform = transforms.Compose([transforms.ToTensor()])
model = newresnet50_4()
#print('model',model)
model.load_state_dict(torch.load("best_model.pt",map_location=lambda storage, loc: storage))
model = model.cuda()

model.eval()
correct = 0
total = 0
batch_size = 2
pre = []
lab = []
score = []
test_path = "../list/test5.txt"
labels_dir = "labels2.txt"
output_dir = "outputs2.txt"

test_dataset = MyDataset(txt_path=test_path, transform=data_transform, target_transform=None)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for i, data in enumerate(test_loader, 0):
    inputs1,inputs2,inputs3,inputs4, labels = data
    inputs1,inputs2,inputs3,inputs4, labels = Variable(inputs1).cuda(),Variable(inputs2).cuda(),Variable(inputs3).cuda(),Variable(inputs4).cuda(),Variable(labels).cuda()

    outputs = model(inputs1, inputs2, inputs3, inputs4)
    output = torch.softmax(outputs,1)
    _, preds = torch.max(outputs.data, 1)
    t = output.cpu().detach().numpy()
    score1 = [x[1] for x in t]
    score = np.concatenate([score,score1])
#    print(score)
    pre1 =  preds.cpu().numpy()
    pre = np.concatenate([pre,pre1])
#    print(pre)
    lab = np.concatenate([lab,labels.cpu().numpy()])
#    print(lab)
    #print(outputs.data)                             
    with open(labels_dir,'a') as f1:
        print(labels, file = f1)
    with open(output_dir,'a') as f2:
        print(preds, file = f2)

    total += labels.size(0)
    correct += (preds == labels).sum()



fpr,tpr,threshold = roc_curve(lab, score)
roc_auc = auc(fpr,tpr)
#plt.figure()
lw = 2
#plt.figure(figsize=(10,10))
#plt.plot(fpr, tpr, color='darkorange',
#         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right")
#plt.savefig("ROC curve")
#plt.close('all')
print("AUC is %.4f" % roc_auc_score(lab,score))
print("正确的数量%d,所有图片数量%d:" % (correct, total))
print('val accuracy of the %d val images:%.4f' % (total,float(correct) / total))

import seaborn as sns
cm = confusion_matrix(lab, pre, [0,1]) 

print(cm)

#sns.heatmap(cm, annot = True, annot_kws={'size':20,'weight':'bold', 'color':'blue'})

#plt.rc('font', family='Arial Unicode MS', size=14)

#plt.title('混淆矩阵',fontsize=20)

#plt.xlabel('Predict',fontsize=14)

#plt.ylabel('Actual',fontsize=14)

#plt.savefig("confusion_matrix")
#plt.close('all')

# 精确率、召回率、F1、准确率
def p_r_f1_a(acts,pres):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(acts)):
        if acts[i] == 1 and pres[i] == 1:
            TP += 1
        if acts[i] == 0 and pres[i] == 1:
            FP += 1
        if acts[i] == 1 and pres[i] == 0:
            FN += 1
        if acts[i] == 0 and pres[i] == 0:
            TN += 1    
    # 精确率Precision
    P = TP / (TP+FP)
    # 召回率Recall
    R = TP / (TP+FN)  
    # F1
    F1 = 2 / (1/P + 1/R)
    # 准确率Accuracy
    A = (TP+TN) / (TP+FP+FN+TN) 
    
    return P, R, F1, A

print("Precision, Recall, F1, Accuracy:",p_r_f1_a(lab,pre))
