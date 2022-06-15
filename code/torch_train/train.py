import os
import random
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler
from dataset import trainSet
from same_pad import SamePad2d
from torch.utils.tensorboard import SummaryWriter

class Model(nn.Module):

    def __init__(self, E=16):
        super(Model, self).__init__()
        self.k_sizes = [3, 3, 3, 3, 2]
        self.heights = [1, 72, 108, 162, 243, 256]
        self.strides = 2
        self.dropout = nn.Dropout(p=0.5)
        l = []
        for i in range(self.k_sizes.__len__()):
            l.append(SamePad2d((self.k_sizes[i], 1), (self.strides, 1)))
            l.append(nn.Conv2d(self.heights[i], self.heights[i+1], (self.k_sizes[i], 1), (self.strides, 1)))
            l.append(nn.ReLU(True))
        self.encoder = nn.ModuleList(l)
        l = []
        strides = [2, 4, 8, 16, 64]
        l.append(nn.Sequential(nn.Conv2d(1, E, (3, 1), (32, 1)),
                               nn.ReLU(True)))

        for s in strides:
            l.append(nn.Sequential(SamePad2d((1, 3), (1, s)),
                     nn.Conv2d(E, E, (1, 3), (1, s)),
                     nn.ReLU(True)))
        self.emotion_module = nn.ModuleList(l)
        l = []
        strides = [2, 2, 2, 4]
        kernels = [3, 3, 3, 4]
        l.append(nn.Sequential(SamePad2d((1, 3), (1, 2)),
                 nn.Conv2d(256, 256, (1, 3), (1, 2)),
                 nn.ReLU(True)))
        for i in range(len(strides)):
            k = kernels[i]
            s = strides[i]
            l.append(nn.Sequential(SamePad2d((1, k), (1, s)),
                     nn.Conv2d(256+E, 256, (1, k), (1, s)),
                     nn.ReLU(True)))
        self.decoder = nn.ModuleList(l)
        self.fc = nn.Sequential(
            nn.Linear(256+E, 150),
            self.dropout,
            nn.Linear(150, 61) # train_label_var第二维的长度
        )
    def forward(self, x):
        enmotion_vectors = []
        enmotion_vectors.append(self.emotion_module[0](x))
        for i in range(1, len(self.emotion_module)):
            modu = self.emotion_module[i]
            enmotion_vectors.append(modu(enmotion_vectors[0]))
        enmotion_vectors.pop(0)
        for layer in self.encoder:
            x = layer(x)
        for i in range(len(enmotion_vectors)):
            x = self.decoder[i](x)
            x = torch.cat([x, enmotion_vectors[i]], 1)
        x = torch.flatten(x, start_dim=1)
        out = self.fc(x)
        return out, enmotion_vectors[0]


def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if rank == 0:
            tensor /= world_size


def evaluate(valid_loader):
    model.eval()
    with torch.no_grad():
        cnt = 0
        total = 0
        for inputs, labels in valid_loader:
            inputs, labels = inputs.unsqueeze(1).float().cuda(), labels.long().cuda()
            output = model(inputs)
            predict = torch.argmax(output, dim=1)
            cnt += (predict == labels).sum().item()
            total += len(labels)
            # print(f'right = {(predict == labels).sum()}')
        cnt = torch.Tensor([cnt]).to(inputs.device)
        total = torch.Tensor([total]).to(inputs.device)
        reduced_param = torch.cat((cnt.view(1), total.view(1)))
        cnt = reduced_param[0].item()
        total = reduced_param[1].item()
    return cnt, total


def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('resume', type=int, default=0, help="resume or not")
    parser.add_argument('--local_rank', type=int, default=-1, help="local gpu id")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learn rate")
    parser.add_argument('--epochs', type=int, default=250, help="train epoch")
    parser.add_argument('--seed', type=int, default=42, help="train epoch")
    args = parser.parse_args()
    return args

def criterion2(y, y_, emotion_input):

    ### 计算loss_M
    split_y = torch.split(y,y.shape[0]//2,0) #参数分别为：tensor，拆分数，维度
    split_y_ = torch.split(y_,y.shape[0]//2,0) #参数分别为：tensor，拆分数，维度
    # print(10)
    y0 = split_y[0]
    y1 = split_y[1]
    y_0 = split_y_[0]
    y_1 = split_y_[1]
    loss_M = 2 * torch.mean(torch.square(y0 - y1 -y_0 + y_1))

    ### 计算loss_R
    # 拆分tensor。https://blog.csdn.net/liuweiyuxiang/article/details/81192547
    split_emotion_input = torch.split(emotion_input, emotion_input.shape[0]//2,0) #参数分别为：tensor，拆分数，维度
    # print(10)
    emotion_input0 = split_emotion_input[0]
    emotion_input1 = split_emotion_input[1]

    # 公式(3),Rx3即R'(x)
    Rx0 = torch.square(emotion_input0-emotion_input1) #计算m[·]
    Rx1 = torch.sum(Rx0,2) #4维。按shape(1)计算和，即：高
    Rx2 = torch.sum(Rx1,2) #3维。按shape(1)计算和，即：4维的shape(2)，宽
    Rx3 = 2 * torch.mean(Rx2,1) #2维。按shape(1)计算均值，即：4维的shape(3)，E

    # 公式(4),Rx是长度为batch_size/2的tensor
    e_mean0 = torch.reduce_sum(torch.square(emotion_input0),3) #4维。按shape(2)计算和，即：宽。因为高为1，所以只算一次sum
    e_mean1 = torch.mean(e_mean0) #2维。按shape(1)计算均值，即：4维的shape(3)，E
    Rx = Rx1/e_mean1

    # 公式(5)
    # R_vt = beta * R_vt_input + (1-beta) * tf.reduce_mean(tf.square(Rx)) #每个batch运行一次
    # R_vt_ = R_vt/(1-tf.pow(beta, step))

    # 公式(6)
    # loss_R = tf.reduce_mean(Rx)/(tf.sqrt(R_vt_)+epsilon)
    loss_R = torch.mean(Rx)

    # loss_R = tf.reduce_mean(tf.square(emotion_input1 - emotion_input0), name='loss_R')
    loss =  loss_M + loss_R

    return loss

if __name__ == "__main__":
    checkpoint_path = "/mnt/sdf/zzl/FACEGOOD-Audio2Face/code/torch_train/checkpoints/70.pth"
    writer = SummaryWriter()
    args = get_args()
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    set_random_seed(args.seed)

    # initilization
    # world_size = node_number * gpu_number_per_node
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    global_rank = dist.get_rank()
    print(f'global_rank = {global_rank} local_rank = {args.local_rank} world_size = {args.world_size}')

    # build a model
    model = Model().cuda()
    # DDP setting
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    print('ddp')
    model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    print('reading data')
    # construct fake datasets
    trainset = trainSet()
    print('read finished')
    # DDP samplers
    train_sampler = DistributedSampler(trainset)
    #train_sampler = SequentialSampler(trainset)
    # build dataloaders
    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              pin_memory=True,
                              sampler=train_sampler)

    # optmizer
    criterion1 = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.98)
    # training mode
    if args.resume==1:
        if global_rank == 0:
            print("Resuming from checkpoint file.")
        checkpoint = torch.load(checkpoint_path)  # 加载断点

        model.load_state_dict(checkpoint['model'])  # 加载模型可学习参数

        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
    else:
        start_epoch = 0
    model.train()

    # main process
    idx_whole = 0
    time_used = np.array([])
    for e in range(start_epoch+1, int(args.epochs)):
        loss = 0
        if global_rank == 0:
            start_time = time.time()
        for idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.float().cuda()
            labels = labels.float().cuda()
            output, emo = model(inputs)
            #loss = criterion1(output, labels)+ criterion2(output, labels, emo)
            loss = criterion1(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            reduce_loss(loss, global_rank, args.world_size)
            if idx%50 == 0:
                if global_rank == 0:
                    print(f'iter {idx} || loss: {loss}')
                    writer.add_scalar('Loss/train', loss, idx_whole)
            idx_whole+=1
        scheduler.step()
        if global_rank == 0:
            lr_temp = optimizer.state_dict()['param_groups'][0]['lr']
            time_current = (time.time()-start_time)/60/60
            time_used = np.append(time_used, [time_current])
            time_remain = (time_used.mean())*(args.epochs-e)
            print(f'iter {idx} || lr: {lr_temp}, loss: {loss}, eta: {time_remain} hours.')
            writer.add_scalar('Loss/train/epoch', loss, e)

            state = {'epoch': e,
                     'model': model.module.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict()}
            torch.save(state, '/mnt/sdf/zzl/FACEGOOD-Audio2Face/code/torch_train/checkpoints/'+str(e)+'.pth')
        # cnt, total = evaluate(valid_loader)
        '''if global_rank == 0:
            print(f'epoch {e} || eval accuracy: {cnt / total}')'''
