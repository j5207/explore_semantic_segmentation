import sys, os
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
import pydensecrf.densecrf as dcrf
from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.loss import *
from ptsemseg.augmentations import *
import scipy.misc as misc
def train(args):
    loss_miou = []
    loss_loss = []
    loss_acc = []   
    # Setup Augmentations
    data_aug= Compose([RandomRotate(10),                                        
                       RandomHorizontallyFlip()])

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols), augmentations=data_aug, img_norm=args.img_norm)
    v_loader = data_loader(data_path, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols), img_norm=args.img_norm)
    
    
    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=8, shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=8)

    # Setup Metrics
    running_metrics = runningScore(n_classes)
        
    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom()

        loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='minibatches',
                                     ylabel='Loss',
                                     title='Training Loss',
                                     legend=['Loss']))

    # Setup Model
    model = get_model(args.arch, n_classes)
    
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    
    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)

    if hasattr(model.module, 'loss'):
        print('Using custom loss')
        loss_fn = model.module.loss
    else:
        loss_fn = cross_entropy2d

    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume)) 

    best_iou = -100.0 
    for epoch in range(args.n_epoch):
        model.train()
        for i, (images, labels) in enumerate(trainloader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(input=outputs, target=labels)

            loss.backward()
            optimizer.step()

            if args.visdom:
                vis.line(
                    X=torch.ones((1, )).cpu() * i,
                    Y=torch.Tensor([loss.data[0]]).cpu(),
                    win=loss_window,
                    update='append')

            if (i+1) % 20 == 0:
                print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, loss.data[0]))

        model.eval()
        for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
            #print('here i am')

            images_val = Variable(images_val.cuda(), volatile=True)
            labels_val = Variable(labels_val.cuda(), volatile=True)
    #        print(image_val.shape, label_val.shape)
            outputs = model(images_val)

            unary = outputs.data.cpu().numpy()
            
            # unary = np.squeeze(unary, 0)
            unary[unary == 0] = 1
            unary = -np.log(unary)
            unary[np.isneginf(unary)]=0
            unary = unary.transpose(0, 3, 2, 1)
            #print(unary.shape)
            n, w, h, c = unary.shape
            pred = []
            for i in range(n):
                unary_t = unary[i].transpose(2, 0, 1).reshape(n_classes, -1)
                unary_t = np.ascontiguousarray(unary_t).astype(np.float32)
                img = images_val[i].data.cpu().numpy()
                try:
                    img = np.squeeze(img)
                except:
                    pass
                img = img.transpose(2, 1, 0)
                #print(img.shape)       
                resized_img = np.ascontiguousarray(img).astype(np.uint8)
                #print(resized_img.shape)
                d = dcrf.DenseCRF2D(w, h, n_classes)
                d.setUnaryEnergy(unary_t)
                d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=resized_img, compat=1)
                q = d.inference(50)
                pred.append(np.argmax(q, axis=0).reshape(w, h).transpose(1, 0))
            #print(outputs.data.shape)
            #pred = outputs.data.max(1)[1].cpu().numpy()
            pred = np.stack(pred)
            pred1 = outputs.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()
           # print('update',pred.shape, gt.shape)
            #print('here i am11')
            running_metrics.update(gt, pred)

        score, class_iou = running_metrics.get_scores()
        for k, v in score.items():
            print(k, v)
        running_metrics.reset()
        loss_miou.append(score['Mean IoU : \t'])
        loss_acc.append(score['Overall Acc: \t'])
        loss_loss.append(loss.data[0])
        np.savetxt( '/home/jhan5307/log_miou_crf.txt', np.array(loss_miou))
        np.savetxt('/home/jhan5307/log_acc_crf.txt',np.array(loss_acc))
        np.savetxt('/home/jhan5307/log_loss_crf.txt',np.array(loss_loss))
        if score['Mean IoU : \t'] >= best_iou:
            best_iou = score['Mean IoU : \t']
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            torch.save(state, "/home/jhan5307/{}_{}_best_model.pkl".format(args.arch, args.dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='segnet',
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256, 
                        help='Width of the input image')

    parser.add_argument('--img_norm', dest='img_norm', action='store_true', 
                        help='Enable input image scales normalization [0, 1] | True by default')
    parser.add_argument('--no-img_norm', dest='img_norm', action='store_false', 
                        help='Disable input image scales normalization [0, 1] | True by default')
    parser.set_defaults(img_norm=True)

    parser.add_argument('--n_epoch', nargs='?', type=int, default=100, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5, 
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1, 
                        help='Divider for # of features to use')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')

    parser.add_argument('--visdom', dest='visdom', action='store_true',default = True, 
                        help='Enable visualization(s) on visdom | False by default')
    parser.add_argument('--no-visdom', dest='visdom', action='store_false', 
                        help='Disable visualization(s) on visdom | False by default')
    parser.set_defaults(visdom=False)

    args = parser.parse_args()
    train(args)
