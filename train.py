import os
import torch
import torch.nn as nn
import torchvision
import torch.utils.data
import torch.nn.functional
from torchvision import transforms
import argparse
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import time
import warnings

from MyDataset import MyDataset
from newRes_2_2 import newresnet50_2
from newRes_2_4 import newresnet50_4
from newRes_1_2 import resnet50_2
from ResNet import resnet50

#from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
#from sklearn.externals import joblib

# def classifier_training(feature_path, label_path, save_path):
#     features = pickle.load(open(feature_path, 'rb'))
#     labels = pickle.load(open(label_path, 'rb'))
#     #classifier = SVC(C=0.5)
#     # classifier = MLPClassifier()
#     classifier = RandomForestClassifier(n_jobs=4, criterion='entropy', n_estimators=70, min_samples_split=5)
#     # classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=4)
#     # classifier = ExtraTreesClassifier(n_jobs=4,  n_estimators=100, criterion='gini', min_samples_split=10,
#     #                        max_features=50, max_depth=40, min_samples_leaf=4)
#     # classifier = GaussianNB()
#     print(".... Start fitting this classifier ....")
#     classifier.fit(features, labels)
#     print("... Training process is down. Save the model ....")
#     joblib.dump(classifier, save_path)
#     print("... model is saved ...")

train_path = '../list/trainset.txt'
val_path = '../list/valset.txt'
loss_dir = 'loss.txt'
trainacc_dir = 'trainacc.txt'
valacc_dir = 'valacc.txt'
Maxepoch = 150
batch_size = 10
lrinit = 0.0005
logpath = "./log/"

model = newresnet50_4()
model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lrinit)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

data_transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MyDataset(txt_path=train_path, transform=data_transform, target_transform= None)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = MyDataset(txt_path=val_path, transform=data_transform, target_transform=None)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lrinit)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

Loss_list_train = []
Loss_list_val = []
Acc_list_train = []
Acc_list_val = []

best_acc =0.0
best_model = model.state_dict()

for epoch in range(Maxepoch):
    #train
    loss_sigma = 0.0
    correct = 0.0
    total = 0.0
    #scheduler.step()
    print('Epoch {} train'.format(epoch))
    model.train()

    for i, data in enumerate(train_loader, 0):
        inputs1, inputs2, inputs3, inputs4, labels = data
        inputs1, inputs2, inputs3, inputs4, labels = Variable(inputs1).cuda(), Variable(inputs2).cuda(), Variable(inputs3).cuda(),Variable(inputs4).cuda(),Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = model(inputs1,inputs2,inputs3,inputs4)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _,predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        loss_sigma += loss.item()

    epoch_loss = loss_sigma/len(train_dataset)
    Loss_list_train.append(epoch_loss)

    epoch_acc = float(correct)/len(train_dataset)
    Acc_list_train.append(epoch_acc)

    print('Loss:{:.4f} Acc:{:.4f}'.format(epoch_loss,epoch_acc))

    logFileLoc = logpath + "log.txt"
    with open(logFileLoc,'a') as f1:
        print('Epoch {} train'.format(epoch), file = f1)
        print('Loss:{:.4f} Acc:{:.4f}'.format(epoch_loss,epoch_acc), file = f1)
    #print('Epoch {} train'.format(epoch))
    #print('Loss:{:.4f} Acc:{:.4f}'.format(epoch_loss,epoch_acc))
    #valid
    if epoch >= 0:
        print('Epoch {} valid'.format(epoch))
        loss_sigma = 0.0
        correct = 0.0
        total = 0.0
        loss_sigma = 0.0
        model.eval()
        for i, data in enumerate(val_loader,0):
            inputs1, inputs2, inputs3, inputs4, labels = data
            inputs1, inputs2, inputs3, inputs4, labels = Variable(inputs1).cuda(), Variable(inputs2).cuda(), Variable(inputs3).cuda(),Variable(inputs4).cuda(), Variable(labels).cuda()

            optimizer.zero_grad()
            outputs = model(inputs1, inputs2, inputs3, inputs4)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            loss_sigma += loss.item()


        epoch_loss = loss_sigma / len(val_dataset)
        Loss_list_val.append(epoch_loss)
        epoch_acc = float(correct) / len(val_dataset)
        Acc_list_val.append(epoch_acc)

        print('Loss:{:.4f} Acc:{:.4f}'.format(epoch_loss, epoch_acc))

        logFileLoc = logpath + "log.txt"
        with open(logFileLoc, 'a') as f1:
            print('Epoch {} valid'.format(epoch), file=f1)
            print('Loss:{:.4f} Acc:{:.4f}'.format(epoch_loss, epoch_acc), file=f1)
        #print('Epoch {} valid'.format(epoch))
        #print('Loss:{:.4f} Acc:{:.4f}'.format(epoch_loss, epoch_acc))
        if epoch_acc > best_acc and epoch > 80:
            best_acc = epoch_acc

            best_model = model.state_dict()

        if (epoch+1)%100 ==0 and epoch >80:
            model.load_state_dict(model.state_dict())
            torch.save(model.state_dict,'epoch_'+str(epoch)+'bestacc_'+str(best_acc)+'_model.pt')
print('Best val acc: {:.4f}'.format(best_acc))

model.load_state_dict(best_model)
torch.save(model.state_dict(),'best_model.pt')




x = range(0, Maxepoch)
y1 = Loss_list_val
y2 = Loss_list_train

plt.plot(x, y1, color="b", linestyle="-", marker="o", linewidth=1, label="valid")
plt.plot(x, y2, color="r", linestyle="-", marker="o", linewidth=1, label="train")
plt.legend()
plt.title('train and val loss per epoches')
plt.ylabel('loss')
plt.savefig(logpath + "train and val loss per epoches.jpg")
plt.close('all') # 

y5 = Acc_list_train
y6 = Acc_list_val
plt.plot(x, y5, color="r", linestyle="-", marker=".", linewidth=1, label="train")
plt.plot(x, y6, color="b", linestyle="-", marker=".", linewidth=1, label="valid")
plt.legend()
plt.title('train and val acc per epoches')
plt.ylabel('accuracy')
plt.savefig(logpath + "train and val acc per epoches.jpg")
plt.close('all')




