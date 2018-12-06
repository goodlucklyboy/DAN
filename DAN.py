import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import math
import data_loader
import ResNet as models
from torch.utils import model_zoo

import torchvision.models as mo_guanfang
import torchvision
#import matplotlib.pyplot as plt
import numpy as np
##########################################################################
#################    Result saved setting       ##########################
##########################################################################
#plt.switch_backend('agg')
save_name = 'DAN_RESNET'
save_dir = './result/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
torch.cuda.empty_cache()
##########################################################################
#################   1/  HYPER PARAMETERS        ##########################
##########################################################################
batch_size = 10
epochs = 500
lr = 0.01
momentum = 0.9
no_cuda = False
seed = 8
log_interval = 10
l2_decay = 5e-4
root_path = "./7_office/"
source_name = "amazon"
target_name = "webcam"

cuda = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


torch.cuda.manual_seed(seed)

##########################################################################
#################       DATASET         ##########################
##########################################################################
print('==> Preparing data..')

source_loader = data_loader.load_training(root_path, source_name, batch_size)
target_train_loader = data_loader.load_training(root_path, target_name, batch_size)
target_test_loader = data_loader.load_testing(root_path, target_name, batch_size)

len_source_dataset = len(source_loader.dataset)
len_target_dataset = len(target_test_loader.dataset)
len_source_loader = len(source_loader)
len_target_loader = len(target_train_loader)


##########################################################################
###########       Network architecture and optimizer        ##############
##########################################################################

def load_pretrain(model):
    net = mo_guanfang.resnet50(pretrained=True)
    model.sharedNet.layer1 = net.layer1
    model.sharedNet.layer2 = net.layer2
    model.sharedNet.layer3 = net.layer3
    model.sharedNet.layer4 = net.layer4
    model.sharedNet.avgpool = net.avgpool
    # url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    # pretrained_dict = model_zoo.load_url(url)
    # model_dict = model.state_dict()
    # for k, v in model_dict.items():
    #     if not "cls_fc" in k:
    #         model_dict[k] = pretrained_dict[k[k.find(".") + 1:]]
    # model.load_state_dict(model_dict)
    return model

model = models.DANNet(num_classes=31)
correct = 0
print(model)

model = load_pretrain(model)#最终的网络是DANnet，
if cuda:
    model.cuda()


##########################################################################
###################       Assistant function           ###################
##########################################################################
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = torchvision.utils.make_grid(inp)
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
   # plt.imshow(inp)
    #if title is not None:
     #   plt.title(title)
   # plt.pause(0.001)  # pause a bit so that plots are updated
# Get a batch of training data


##########################################################################
###################       Training and validate        ###################
##########################################################################
def train(epoch, model):
    LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
    print('learning rate{: .4f}'.format(LEARNING_RATE) )
    optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

    model.train()

    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len_source_loader

    correct = 0
    total = 0

    for batch_idx in range(1, num_iter):

        data_source, label_source = iter_source.next()
        data_target, label_target = iter_target.next()


        if batch_idx % len_target_loader == 0:
            iter_target = iter(target_train_loader)
        if cuda:
            data_source, label_source = data_source.cuda(), label_source.cuda()
            data_target = data_target.cuda()
        data_source, label_source = Variable(data_source), Variable(label_source)
        data_target, label_target = Variable(data_target), Variable(label_target)

        # print(label_source)
        # print(label_target)
        optimizer.zero_grad()
        label_source_pred, loss_mmd = model(data_source, data_target)
        loss_cls = F.nll_loss(F.log_softmax(label_source_pred, dim=1), label_source)
        gamma = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1
        loss = loss_cls + gamma * loss_mmd
        loss.backward()
        optimizer.step()

        _, predicted = label_source_pred.max(1)
        total += label_source.size(0)
        correct += predicted.eq(label_source).sum().item()
        if batch_idx % log_interval == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tSource Acc ten batch: {:.6f}\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}'.format(
                epoch, batch_idx * len(data_source), len_source_dataset,
                100. * batch_idx / len_source_loader, 100. * correct / total,loss.item(), loss_cls.item(), loss_mmd.item()))
            correct = 0
            total = 0
def test(model):
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in target_test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        target = Variable(target)
        with torch.no_grad():data  = Variable(data)
        s_output, t_output = model(data, data)
        test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), target, size_average=False).item() # sum up batch loss
        pred = s_output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len_target_dataset
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        target_name, test_loss, correct, len_target_dataset,
        100. * correct / len_target_dataset))
    return correct

##########################################################################
#########################       Main        ##############################
##########################################################################

#plt.figure(0)
for epoch in range(1, epochs + 1):
    train(epoch, model)
    t_correct = test(model)
    if t_correct > correct:
        correct = t_correct
        print('source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
        source_name, target_name, correct, 100. * correct / len_target_dataset ))