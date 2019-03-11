import os
import sys
import torch
import argparse
import os.path as osp
import torch.nn as nn
import torch.optim as optim
from model import creat_model
from serialization import Logger
from train import train, evaluate
from dataset import segmentationData
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='bisenet segmentation')
parser.add_argument('--dataset_path', type = str, default = '/public/chnn/segmentation/dataset')
parser.add_argument('--batch', type = int, default = 8)
parser.add_argument('--test_batch', type = int, default = 8)
parser.add_argument('--epochs', type = int, default = 40)
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--adjust_lr', type = bool, default = True)
parser.add_argument('--momentum', type = float, default = 0.9)
parser.add_argument('--gpu', type = bool, default = True)
parser.add_argument('--shuffle', type = bool, default = True)
parser.add_argument('--resume', type = str, default = None)
parser.add_argument('--height', type = int, default = 1024)
parser.add_argument('--width', type = int, default = 2048)
parser.add_argument('--evaluate_interval', type = int, default = 1)
parser.add_argument('--save_interval', type = int, default = 5)
parser.add_argument('--save_dir', type = str, default = './models')
parser.add_argument('--test_only', type = bool, default = False)
parser.add_argument('--num_classes', type = int, default = 19)
args = parser.parse_args()

model = creat_model(not args.test_only, args.num_classes)
if args.resume:
    #model.load_state_dict(torch.load(args.resume))
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.resume).items()})

if args.gpu:
    model = nn.DataParallel(model).cuda()

data_transform = transforms.Compose(transforms = [
    transforms.ColorJitter(brightness = 0.5, contrast = 0.5, hue = 0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#segmentation
])
label_transform = transforms.Compose(transforms = [
    transforms.Resize((args.height/8, args.width/8), interpolation = 3),
    transforms.ToTensor()
])

kwargs = {'num_workers': 6, 'pin_memory': True} if args.gpu else {}

trainset = segmentationData(args.dataset_path, split = 'train', dataTransform = data_transform, labelTransform = label_transform)
testset = segmentationData(args.dataset_path, split = 'val', dataTransform = data_transform, labelTransform = label_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size = args.batch, shuffle = True, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size = args.test_batch, shuffle = True, **kwargs)

optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum)
#optimizer = optim.Adam(model.parameters(), lr = args.lr)

def adjust_lr(base_lr, optimizer, epoch):
    lr = base_lr * (0.1 ** (epoch // 20))
    for para in optimizer.param_groups:
        para['lr'] = lr

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))

print('let us begin:')
if args.test_only and args.resume is not None:
    evaluate(model, test_loader, args.num_classes)

if not args.test_only:
    start_epoch = 0

for epoch in range(start_epoch, args.epochs):
    if args.adjust_lr:
	    adjust_lr(args.lr, optimizer, epoch)

    train(epoch, model, optimizer, train_loader)

    #if epoch % args.evaluate_interval == 0 or epoch == args.epochs - 1:
        #evaluate(model, test_loader, args.num_classes)

    if epoch % args.save_interval == 0 or epoch == args.epochs - 1:
        print('saving model..')
        torch.save(model.state_dict(),  osp.join(args.save_dir, ('model_{}.pth'.format(epoch))))
