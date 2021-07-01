import torch.nn as nn 
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import shutil
import argparse
import os 
import PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import time
import pretrainedmodels
from main_model import MODEL

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')

# ==================================================================
# Constants
# ==================================================================
EPOCH         = 120            # number of times for each run-through
BATCH_SIZE    = 8            # number of images for each epoch
LEARNING_RATE = 0.0001          # default learning rate 
WEIGHT_DECAY  = 0             # default weight decay
N             = 256          # size of input images (512 or 640)
MOMENTUM      = (0.9, 0.997)  # momentum in Adam optimization 
GPU_IN_USE    = torch.cuda.is_available()  # whether using GPU
DIR_TRAIN_IMAGES   = ''
DIR_TEST_IMAGES    = ''
IMAGE_PATH = ''
PATH_MODEL_PARAMS  = './model/ISIAfood500.pth'
NUM_CATEGORIES     = 500
LOSS_OUTPUT_INTERVAL = 100
WEIGHT_PATH= ''

# ==================================================================
# Parser Initialization
# ==================================================================
parser = argparse.ArgumentParser(description='Pytorch Implementation of Nasnet Finetune')
parser.add_argument('--lr',              default=LEARNING_RATE,     type=float, help='learning rate')
parser.add_argument('--epoch',           default=EPOCH,             type=int,   help='number of epochs')
parser.add_argument('--trainBatchSize',  default=BATCH_SIZE,        type=int,   help='training batch size')
parser.add_argument('--testBatchSize',   default=BATCH_SIZE,        type=int,   help='testing batch size')
parser.add_argument('--weightDecay',     default=WEIGHT_DECAY,      type=float, help='weight decay')
parser.add_argument('--pathModelParams', default=PATH_MODEL_PARAMS, type=str,   help='path of model parameters')
parser.add_argument('--saveModel',       default=True,              type=bool,  help='save model parameters')
parser.add_argument('--loadModel',       default=False,             type=bool,  help='load model parameters')
parser.add_argument('--classnumble',     default=NUM_CATEGORIES,    type=int,  help='the class number of the dataset')
parser.add_argument('--weightpath',     default=WEIGHT_PATH,        type=str,  help='inint weight path')
parser.add_argument('--print_freq',     default=200,                type=int,  help='print')

args = parser.parse_args()


# ==================================================================
# Prepare Dataset(training & test)
# ==================================================================
print('***** Prepare Data ******')

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])

train_transforms = transforms.Compose([
                     transforms.RandomHorizontalFlip(p=0.5), # default value is 0.5
                     transforms.Resize((N, N)),
                     transforms.RandomCrop((224,224)),
                     transforms.ToTensor(),
                     normalize
                  ])

test_transforms = transforms.Compose([
                    transforms.Resize((N, N)), 
                    transforms.CenterCrop((224, 224)),
                    transforms.ToTensor(),
                    normalize
                  ]) 

def My_loader(path):
    return PIL.Image.open(path).convert('RGB')
 
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, txt_dir, transform=None, target_transform=None, loader=My_loader):
        data_txt = open(txt_dir, 'r')
        imgs = []
        
        for line in data_txt:
            line = line.strip()
            words = line.split()

            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = My_loader
 
    def __len__(self):
        
        return len(self.imgs)
  
    def __getitem__(self, index):
        img_name, label = self.imgs[index]

        try:

   
            img = self.loader(os.path.join(IMAGE_PATH,img_name))
            if self.transform is not None:
                img = self.transform(img)
        except:
            img = np.zeros((256,256,3),dtype=float)
            img  = PIL.Image.fromarray(np.uint8(img))
            if self.transform is not None:
                img = self.transform(img)
            print('erro picture:', img_name)
        return img, label

train_dataset = MyDataset(txt_dir=DIR_TRAIN_IMAGES , transform=train_transforms)
test_dataset = MyDataset(txt_dir=DIR_TEST_IMAGES , transform=test_transforms)
train_loader  = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=40, shuffle=True,  num_workers=2)
test_loader   = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=20,  shuffle=False, num_workers=2)
print('Data Preparation : Finished')


# ==================================================================
# Prepare Model
# ==================================================================
print('\n***** Prepare Model *****')

model_name = 'se_resnext101_32x4d'
model = MODEL( num_classes= 500 , senet154_weight = WEIGHT_PATH, multi_scale = True, learn_region=True)
model = torch.nn.DataParallel(model)
vgg16 = model
vgg16.load_state_dict(torch.load('./model/ISIAfood500.pth'))

print('\n*****  Model load the weight*****')

if GPU_IN_USE:
    print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
    print('cuda: move all model parameters and buffers to the GPU')
    vgg16.cuda()

    cudnn.benchmark = True  #

criterion = nn.CrossEntropyLoss().cuda()

concate_output = list(map(id,  vgg16.module.classifier_global.parameters()))

theta_out2 = list(map(id, vgg16.module.ha2.hard_attn.fc.parameters()))
theta_out3 = list(map(id, vgg16.module.ha3.hard_attn.fc.parameters()))
theta_out4 = list(map(id, vgg16.module.ha4.hard_attn.fc.parameters()))
theta_out = theta_out2 + theta_out3 + theta_out4 

ignored_params = concate_output + theta_out

base_params = filter(lambda p: id(p) not in ignored_params, vgg16.module.parameters())
optimizer = optim.SGD([
    {'params': base_params},
    {'params':vgg16.module.classifier_global.parameters(), 'lr': args.lr},
    # {'params': vgg16.global_out.parameters(), 'lr': args.lr*1},
    # {'params':vgg16.classifier_local.parameters(), 'lr': args.lr*1},
    # {'params':vgg16.local_fc.parameters(), 'lr': args.lr*10},
    # {'params':vgg16.x1_fc.parameters(), 'lr': args.lr*1},
    # {'params':vgg16.x2_fc.parameters(), 'lr': args.lr*1},
    # {'params':vgg16.x3_fc.parameters(), 'lr': args.lr*1},
    # {'params':vgg16.module.x4_fc.parameters(), 'lr': args.lr*10},
    # {'params':vgg16.ha1.hard_attn.fc.parameters(),'lr':args.lr*1},
    {'params':vgg16.module.ha2.hard_attn.fc.parameters(),'lr':args.lr*0.01},
    {'params':vgg16.module.ha3.hard_attn.fc.parameters(),'lr':args.lr*0.01},
    {'params':vgg16.module.ha4.hard_attn.fc.parameters(),'lr':args.lr*0.01}], lr=args.lr, momentum=0.9, weight_decay=0.0001)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
# optimizer = optim.SGD([
    # {'params': base_params},
    # {'params':vgg16.module.classifier_global.parameters(), 'lr': args.lr*10},
    # {'params':vgg16.module.ha1.hard_attn.fc.parameters(),'lr':args.lr*0.1},
    # {'params':vgg16.module.ha2.hard_attn.fc.parameters(),'lr':args.lr*0.1},
    # {'params':vgg16.module.ha3.hard_attn.fc.parameters(),'lr':args.lr*0.1},
    # {'params':vgg16.module.ha4.hard_attn.fc.parameters(),'lr':args.lr*0.1}], lr=args.lr, momentum=0.9, weight_decay=0.00001)
print('Model Preparation : Finished')
# optimizer = optim.Adam([
#     {'params': base_params},
#     {'params':vgg16.last_linear.parameters(), 'lr': args.lr*10}], lr=args.lr, weight_decay=args.weightDecay, betas=MOMENTUM)





def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()


    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output, global_c, local_c= model(input_var)
        concate_loss = criterion(output, target_var)
        global_loss = criterion(global_c, target_var)
        local_loss = criterion(local_c, target_var)
        loss = concate_loss + 0.5*(global_loss + local_loss)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output, global_c, local_c= model(input_var)
        concate_loss = criterion(output, target_var)
        global_loss = criterion(global_c, target_var)
        local_loss = criterion(local_c, target_var)
        loss = concate_loss + 0.5*(global_loss + local_loss)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg

def save():
    torch.save(vgg16.state_dict(), args.pathModelParams)
    print('Checkpoint saved to {}'.format(args.pathModelParams))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 40))
    param_groups = optimizer.state_dict()['param_groups']
    # print param_groups
    param_groups[0]['lr']=lr
    param_groups[1]['lr']=lr
    param_groups[2]['lr']=lr*0.01
    param_groups[3]['lr']=lr*0.01
    param_groups[4]['lr']=lr*0.01

    for param_group in param_groups:
        print param_group
        # param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

best_prec1 = 0
for epoch in range(0, args.epoch):
    adjust_learning_rate(optimizer, epoch)
    # train for one epoch
    train(train_loader, vgg16, criterion, optimizer, epoch)
    # evaluate on validation set
    prec1 = validate(test_loader,vgg16, criterion)
    # prec1 = test(epoch)
    # remember best prec@1 and save checkpoint
    is_best = prec1 > best_prec1
    if prec1 > best_prec1:
        save()
    best_prec1 = max(prec1, best_prec1)