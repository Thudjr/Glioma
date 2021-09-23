import os
import logging
import time

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch import nn

from user_define import config as cf
from user_define import hyperparameter as hp

import torch.optim
import torchvision.transforms as transforms
from dataset.dataset_train import glioma
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss, DataParallel
from torchvision import models


logging.basicConfig(level=logging.INFO)

trans_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

trans_test = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])




def run():
    os.environ["CUDA_VISIBLE_DEVICES"] = hp.gpu
    save_path = cf.checkpint_path
    summary_path = cf.summary_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(summary_path):
        os.mkdir(summary_path)

    net = models.resnet50(pretrained=False)
    fc_features = net.fc.in_features
    net.fc = nn.Linear(fc_features, len(hp.targets))  # 须知
    net = DataParallel(net)
    net = net.cuda()

    loss_fn = CrossEntropyLoss().cuda()
    # optimizer = torch.optim.SGD(net.parameters(), lr=hp.default_lr, momentum=hp.momentum, weight_decay=hp.weight_decay)
    optimizer = torch.optim.Adam(net.parameters(), lr=hp.default_lr, betas=(0.9,0.990), weight_decay=hp.weight_decay)

    # load dataset
    trainset = glioma(cf.dataset_path + 'train', hp.targets, hp.train_num, hp.is_balanced, trans_train)
    valset = glioma(cf.dataset_path + 'validation', hp.targets, hp.val_num, hp.is_balanced, trans_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=hp.batch_size,
                                              shuffle=True, num_workers=hp.num_workers,drop_last=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=hp.batch_size,
                                            shuffle=False, num_workers=hp.num_workers)

    summary_train = {'epoch': 0, 'step': 0}
    summary_valid = {'loss': float('inf'), 'acc': 0}
    summary_writer = SummaryWriter(summary_path)
    acc_valid_best = float(0)


    for epoch in range(hp.epoch):
        logging.info('epoch:%d,lr:%.9f' % (epoch, optimizer.param_groups[0]['lr']))
        summary_train = train_epoch(summary_train, summary_writer, net,
                                    loss_fn, optimizer,
                                    trainloader)

        torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['step'],
                    'state_dict': net.module.state_dict()},
                   os.path.join(save_path, 'train_epoch%d.ckpt' % epoch))

        time_now = time.time()
        summary_valid = valid_epoch(summary_valid, net, loss_fn,
                                    valloader)
        time_spent = time.time() - time_now


        logging.info('{}, Epoch: {}, step: {}, Validation Loss: {:.5f}, '
                     'Validation ACC: {:.5f}, Run Time: {:.2f}'
                     .format(time.strftime("%Y-%m-%d %H:%M:%S"), summary_train['epoch'],
                             summary_train['step'], summary_valid['loss'],
                             summary_valid['acc'], time_spent))

        summary_writer.add_scalar('valid/loss',
                                  summary_valid['loss'], summary_train['step'])
        summary_writer.add_scalar('valid/acc',
                                  summary_valid['acc'], summary_train['step'])

        if summary_valid['acc'] > acc_valid_best:
            acc_valid_best = summary_valid['acc']

            torch.save({'epoch': summary_train['epoch'],
                        'step': summary_train['step'],
                        'state_dict': net.module.state_dict()},
                       os.path.join(save_path, 'best.ckpt'))

    summary_writer.close()


def train_epoch(summary, summary_writer, model, loss_fn, optimizer, dataloader_train):
    model.train()
    steps = len(dataloader_train)
    batch_size = dataloader_train.batch_size
    dataiter_train = iter(dataloader_train)

    time_now = time.time()
    for step in range(steps):
        data_train, target_train = next(dataiter_train)
        data_train = data_train.float().cuda()
        target_train = target_train.long().cuda()
        output = model(data_train)
        loss = loss_fn(output, target_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = torch.max(output, dim=1)[1]
        acc_data = torch.eq(pred, target_train).sum().item() / batch_size
        loss_data = loss.data

        time_spent = time.time() - time_now
        logging.info(
            '{}, Epoch : {}, Step : {}, Training Loss : {:.5f}, '
            'Training Acc : {:.5f}, Run Time : {:.2f}'
            .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), summary['epoch'] + 1,
                summary['step'] + 1, loss_data, acc_data, time_spent))

        summary['step'] += 1

        if summary['step'] % hp.log_every == 0:
            summary_writer.add_scalar('train/loss', loss_data, summary['step'])
            summary_writer.add_scalar('train/acc', acc_data, summary['step'])

    summary['epoch'] += 1

    return summary


def valid_epoch(summary, model, loss_fn, dataloader_valid):
    model.eval()
    steps = len(dataloader_valid)
    batch_size = dataloader_valid.batch_size
    dataiter_valid = iter(dataloader_valid)

    loss_sum = 0
    acc_sum = 0
    for step in range(steps):
        data_valid, target_valid = next(dataiter_valid)
        data_valid = data_valid.float().cuda()
        target_valid = target_valid.long().cuda()
        output = model(data_valid)
        loss = loss_fn(output, target_valid)

        pred = torch.max(output, dim=1)[1]
        acc_data = torch.eq(pred, target_valid).sum().item() / batch_size
        loss_data = loss.data
        loss_sum += loss_data
        acc_sum += acc_data

    summary['loss'] = loss_sum / steps
    summary['acc'] = acc_sum / steps
    return summary



if __name__ == '__main__':
    run()
