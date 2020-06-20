import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms, datasets, models
import os
import time
from load_data_v3 import train_dataloader,test_dataloader
import json
# from c3d_v1 import C3D
import resnet
# import resnet_v2
# import deepmedic
# import vggnet_v1
config = json.load(open('package.json'))
num_epoch = config['epoch']

os.environ['CUDA_VISIBLE_DEVICES']='0' # 第3块GPU
DEVICE = torch.device('cuda:0')

# c3d = C3D()
c3d = resnet.resnet10(sample_size=8,sample_duration=4)
# c3d = resnet_v2.generate_model(model_depth=10)
# c3d = deepmedic.UNet3D(in_channel=2,n_classes=2)
# c3d = vggnet_v1.vgg13_bn()
c3d.to(DEVICE)

# 优化器和损失函数\
lr = 0.01
optimizer = torch.optim.SGD(c3d.parameters(),lr=lr,weight_decay=0.01)
# optimizer = torch.optim.Adam(c3d.parameters(),lr=lr,betas=(0.9,0.999))
criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.BCELoss(reduce=False, size_average=False)
tmp = []
testtmp = []
loss1 = []

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


for epoch in range(num_epoch):
    c3d = c3d.train()
    running_loss = 0.0
    correct = 0
    accuracy = 0

    print("EPOCH:", epoch + 1)
    for sample,label in train_dataloader:
        sample =sample.to(DEVICE)
        label = label.to(DEVICE)
        label[label <= 0] = 0

        inputs, targets_a, targets_b, lam = mixup_data(sample, label, alpha=1.0)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                            targets_a, targets_b))
        outputs = c3d(inputs)
        optimizer.zero_grad()
        running_loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        print_loss = running_loss.data.item()
        running_loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct = (predicted == label).sum()
        accuracy += correct.item()
        # print(accuracy)
    i = 0
    for osample,olabel in train_dataloader:
        osample = osample.to(DEVICE)
        olabel = olabel.to(DEVICE)
        olabel[olabel<=0] = 0
        optimizer.zero_grad()
        ooutput = c3d(osample)

        running_loss = criterion(ooutput,olabel)
        print_loss = running_loss.data.item()

        running_loss.backward()
        optimizer.step()
        _,predicted = torch.max(ooutput,1)
        correct = (predicted==olabel).sum()
        accuracy += correct.item()
        i = i+1
        if(i>0):
            break

    accuracy = accuracy/(len(train_dataloader)*10+i*10)
    tmp.append(accuracy)
    loss1.append(print_loss)
    print("Accuracy:", accuracy)
    print("Loss:", print_loss)

    c3d = c3d.eval()
    testloss = 0
    correct = 0
    testaccuracy = 0

    for photo, label in test_dataloader:
        # print(len(test_dataloader))
        photo = photo.to(DEVICE)
        label = label.to(DEVICE)
        label[label <= 0] = 0
        photo = Variable(photo)
        label = Variable(label)
        output = c3d(photo)

        testloss = criterion(output, label)
        print_testloss = testloss.data.item()

        _, predicted = torch.max(output, 1)
        correct = (predicted == label).sum()
        # print('correct',correct)
        testaccuracy += correct.item()

    testaccuracy = testaccuracy / len(test_dataloader) / 4
    testtmp.append(testaccuracy)
    print("Test_accuracy:", testaccuracy)
    print("Test_Loss:", print_testloss)

    if epoch == 0:
        torch.save(c3d.state_dict(), 'model_new_res1.pkl')
        print(epoch + 1, "saved")
        k = testtmp[0]
        q = tmp[0]
    # elif (epoch >= 1) and (testtmp[epoch] >= k) and (tmp[epoch] >= q):
    elif (epoch >= 1) and (testtmp[epoch] >= k)and (tmp[epoch] >= q):
        k = testtmp[epoch]
        q = tmp[epoch]
        torch.save(c3d.state_dict(), 'model_new_res1.pkl')
        print(epoch + 1, "saved")

    if (epoch + 1) / float(num_epoch) == 0.3 or (epoch + 1) / float(num_epoch) == 0.6 or (epoch + 1) / float(
            num_epoch) == 0.9:
        lr /= 10
        # print('reset learning rate to:', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            print('lr： ',param_group['lr'])

import pandas as pd
result ={'loss':loss1,'train_accu':tmp,'test_accu':testtmp}
result_df = pd.DataFrame(result)
result_df.to_csv('result_1.csv',index=False)