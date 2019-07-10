import os
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import *
import utils
from models import *
from losses.multi import MultiClassCriterion
import argparse
import yaml
import numpy as np
from PIL import Image
import math
import SimpleITK as sitk

torch.cuda.set_device(0)

### Load parameters
Task = 'Task1' #{Task1, Task2, Task3, Task4}
ckpt_path = './ckpt_' + Task
exp_name = 'UNET'
# encoder = 'ResNeXt101' #{ResNeXt101, Resnet18}
args_config = os.path.join('./models', exp_name, 'config.yaml')
args = yaml.load(open(args_config))

parser = argparse.ArgumentParser()
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--save_pred', '-s', action='store_true', help='save prediction sample')
parser.add_argument('--test', '-t', action='store_true', help='test model')
flag = parser.parse_args()
best_eval = 0 # best test evaluation
start_epoch = 0 # start from epoch 0 or last checkpoint epoch

# Load dynamic dataset
dataset_exec = 'from datasets.'+Task+' import dataset'
exec(dataset_exec)
training_root, testing_root = config_path(task='Task1')
train_set = dataset(training_root)
test_set = dataset(testing_root)
train_loader = DataLoader(train_set, batch_size=1, num_workers=4, shuffle=True)

### Model
print('==> Building model..')
outputs_channels = train_set.outputs_channels
net_exec = 'net='+exp_name+'(n_class='+str(outputs_channels)+').cuda()'
exec(net_exec)

# resume from pretrained model
if flag.resume or flag.test:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint_path = os.path.join(ckpt_path, exp_name)
    print(checkpoint_path)
    # assert os.path.isdir('checkpoint_path'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(checkpoint_path, 'model.pth'))
    net.load_state_dict(checkpoint['net'])
    best_eval = checkpoint['eval']
    eval_type = checkpoint['eval_type']

optimizer = optim.SGD([
    {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
     'lr': 2 * args['lr']},
    {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
     'lr': args['lr'], 'weight_decay': args['weight_decay']}
], momentum=args['momentum'])

check_mkdir(ckpt_path)
check_mkdir(os.path.join(ckpt_path, exp_name))

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss_record = AvgMeter()
    for batch_idx, data in enumerate(train_loader):
        if epoch % args['lr_step'] == 0 and epoch != 0:
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] / args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] / args['lr_decay']
        inputs_volume, labels_volume = data
        sub_batch_len = math.ceil(inputs_volume.shape[1]/args['train_batch_size'])
        for sub_batch_idx in range(sub_batch_len):
            #split volume to batchs
            start = sub_batch_idx*args['train_batch_size']
            end = (sub_batch_idx+1)*args['train_batch_size'] if (sub_batch_idx+1)*args['train_batch_size']<inputs_volume.shape[1] else inputs_volume.shape[1]
            inputs = inputs_volume[:, start:end, :, :].permute(1, 0, 2, 3)
            inputs = inputs.expand(torch.Size((inputs.shape[0], 3, inputs.shape[2], inputs.shape[3]))) #adjust to net input channel
            # labels = labels_volume[:, start:end, :, :].permute(1, 0, 2, 3)
            labels = labels_volume[:, start:end, :, :].squeeze(0)
            # training trick
            tmp = np.array(labels)
            tmp[0, :128, :] = 255
            tmp[0, 384:, :] = 255
            tmp[0, :, :128] = 255
            tmp[0, :, 384:] = 255
            labels = torch.from_numpy(tmp)


            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
            optimizer.zero_grad()
            outputs = net(inputs)

            # pred=outputs.argmax(dim=1)
            # prediction = np.array(pred.detach().cpu())
            # l = np.array(labels.detach().cpu())
            # index = np.where(l==np.max(l))
            # a=np.count_nonzero(prediction)
            # b=np.count_nonzero(l)
            # print(str(a))
            # print(str(b))

            criterion = MultiClassCriterion('OhemCrossEntropy')
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            train_loss_record.update(loss.item(), batch_size)
            log = 'iter: %d | [Total loss: %.5f], [lr: %.8f]' % \
                  (epoch, train_loss_record.avg, optimizer.param_groups[1]['lr'])
            progress_bar(batch_idx, len(train_loader), log)


# Testing
def test(epoch):
    global best_eval
    net.eval()
    evaluator_hd95 = Evaluator_hd95()
    evaluator_dice = Evaluator_dice()
    with torch.no_grad():
        for idx, sample_name in enumerate(os.listdir(os.path.join(testing_root))):
            volume_path = os.path.join(testing_root, sample_name, 'data.nii.gz')
            gt_path = os.path.join(testing_root, sample_name, 'label.nii.gz')
            itk_CT = sitk.ReadImage(volume_path)
            itk_gt = sitk.ReadImage(gt_path)
            torch_CT = test_set._img_transfor(itk_CT)
            torch_gt = test_set._label_transfor(itk_gt)
            sub_batch_len = torch_CT.shape[0]
            pred_list = []
            for sub_batch_idx in range(sub_batch_len):
                img = torch_CT[sub_batch_idx, :, :]
                img = img.expand(torch.Size((1, 3, *img.shape)))
                img_var = Variable(img).cuda()
                outputs = net(img_var)
                outputs = outputs.argmax(dim=1)
                prediction = np.array(outputs.detach().cpu())
                # a=np.count_nonzero(prediction)
                # print(str(a))
                # prediction = crf_refine(np.array(img.permute(0,2,3,1)).squeeze(0).astype(np.uint8), prediction.astype(np.uint8))

                pred_list.append(prediction)
            pred_volume = np.concatenate(pred_list).astype(np.uint8)
            gt_volume = np.array(torch_gt, dtype=np.uint8)
            # cal each class dice and hd95
            for class_id in range(1, test_set.outputs_channels):
                pred_class_volume = pred_volume == class_id
                gt_class_volume = gt_volume == class_id
                # print('pred %.4f'%(np.count_nonzero(pred_class_volume)))
                # print('gt %.4f' % (np.count_nonzero(gt_class_volume)))

                # evaluator_hd95.add_volume(pred_class_volume, gt_class_volume)
                evaluator_dice.add_volume(pred_class_volume, gt_class_volume)

            current_dice = evaluator_dice.get_eval()
            # current_hd95 = evaluator_hd95.get_eval()
            # progress_bar(idx, len(os.listdir(os.path.join(testing_root))), 'Dice: %.4f, hd95: %.4f'% (current_dice, current_hd95))
            progress_bar(idx, len(os.listdir(os.path.join(testing_root))),
                         'Dice: %.4f' % (current_dice))

        dice = evaluator_dice.get_eval()
        # hd95 = evaluator_hd95.get_eval()
        # print('Mean dice is %.4f | Mean hd95 is %.4f'%(dice, hd95))
        print('Mean dice is %.4f' % (dice))

        # Save checkpoint.
        if dice > best_eval and not flag.test:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'eval': dice,
                'epoch': epoch,
                'eval_type': 'dice'
            }
            checkpoint_path = os.path.join(ckpt_path, exp_name)
            if not os.path.isdir(checkpoint_path):
                os.mkdir(checkpoint_path)
            torch.save(state, os.path.join(checkpoint_path, 'model.pth'))
            best_eval = dice

for epoch in range(start_epoch, start_epoch+args['iter_num']):
    if not flag.test:
        train(epoch)
        test(epoch)
    else:
        test(epoch)
        break


