import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataloader import *
from model import *

##########
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler as DS
from torch.nn.parallel import DistributedDataParallel as DDP
##########

class Run_train():
    def __init__(self, args):
        super(Run_train, self).__init__()
        self.args = args
        
    def train(self):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)
            
        if self.args.cuda == -1:
            device = torch.device('cpu')

        else:
            ##########
            # device = torch.device(f'cuda:{self.args.cuda}' if torch.cuda.is_available() else 'cpu')
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '9999'
            device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
            dist.init_process_group(backend=self.args.backend, init_method="env://", rank=0, world_size=1*torch.cuda.current_device())  # rank should be 0 ~ world_size-1, init_method 임의 포트 지정
            ##########

        if self.args.condition == 'all':
            key_lst = ['01', '02', '03', '04', '05', '06', '07', '09', '10', '11']
            model = DensNet_10(self.args).to(device)
            
        elif self.args.condition == 'local':
            key_lst = ['01', '02', '03', '04', '05', '06', '07']
            ###########
            # model = DensNet_7(self.args).to(device)
            model = DensNet_7(self.args)
            model.cuda(self.args.local_rank)
            model = DDP(model, device_ids=[self.args.local_rank])
            ###########
            
        elif self.args.condition == 'global':
            key_lst = ['09', '10', '11']
            model = DensNet_3(self.args).to(device)

        elif self.args.condition == "global3_local3":
            key_lst = ['01', '04', '07', '09', '10', '11']
            model = DensNet_6(self.args).to(device)
        
        labels = pd.read_csv(self.args.data_dir + '/real_labels.csv', encoding='cp949')  # 라벨 경로 설정

        x_train_lst, x_valid_lst, x_test_lst, y_train, y_valid, y_test, scaler = data_split_multi(self.args.data_dir, key_lst, labels, self.args.seed)

        # 캡션을 지운 이미지들을 resizing(224x346x3) & normalize
        imgtransResize = 224
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        transformList = []
        transformList.append(transforms.ToPILImage())
        transformList.append(transforms.Resize(imgtransResize))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        transformTrain = transforms.Compose(transformList)
        
        transformList = []
        transformList.append(transforms.ToPILImage())
        transformList.append(transforms.Resize(imgtransResize))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        transformVal = transforms.Compose(transformList)
        
        ## train/val로 활용될 dataset 및 loader 생성
        print('building datasets')

        if self.args.dataloader == 'fast':
            train_dataset = MultiImageDataset_fast(x_train_lst, y_train, transformTrain)
            val_dataset = MultiImageDataset_fast(x_valid_lst, y_valid, transformVal)

        elif self.args.dataloader == 'slow':
            train_dataset = MultiImageDataset(x_train_lst, y_train, transformTrain)
            val_dataset = MultiImageDataset(x_valid_lst, y_valid, transformVal)

        print('finished')
        ##########
        train_sampler = DS(train_dataset)
        val_sampler = DS(val_dataset)

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4, pin_memory=True, sampler=train_sampler)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.args.batch_size, num_workers=4, sampler=val_sampler) # val은 절대적 크기가 작으므로 pin_memory=False
        ##########

        print('train_loader 길이 :', len(train_loader))
        print('val_loader 길이 :', len(val_loader))

        optimizer = optim.Adam(model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.args.l2_reg)
        criterion = nn.MSELoss()

        best_loss = 1000

        train_loss_list = []
        val_loss_list = []

        for epoch in range(self.args.total_epoch):
            tr_loss = 0.0
            val_loss = 0.0

            model.train()
            for i, data in enumerate(tqdm(train_loader)):
                inputs, labels = data

                ##############
                print('====== 1 ============')
                print('len(key_lst):', len(key_lst))
                print('inputs[0].shape:', inputs[0].shape)
                print('inputs[0].size(0):', inputs[0].size(0))
                print('imgtransResize:', imgtransResize)
                ##############
                
                inputs_torch = torch.cat(inputs, 0).reshape(len(key_lst), inputs[0].size(0), 3, imgtransResize, -1)

                ##############
                print('====== 2 ============')
                print('len(key_lst):', len(key_lst))
                print('inputs[0].shape:', inputs[0].shape)
                print('inputs[0].size(0):', inputs[0].size(0))
                print('imgtransResize:', imgtransResize)
                ##############

                inputs_torch = inputs_torch.to(device)
                labels = labels.to(device).type(torch.float32)

                attn_score, outputs = model(inputs_torch)  ## outputs : [500hz / 2000hz / 1000hz] 순으로 출력
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                tr_loss += loss.item()

            print(f'{epoch} train_loss :', tr_loss / (i + 1))
            train_loss_list.append((tr_loss / (i + 1)))

            model.eval()
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    inputs, labels = data

                    inputs_torch = torch.cat(inputs, 0).reshape(len(key_lst), inputs[0].size(0), 3, imgtransResize, -1)

                    inputs_torch = inputs_torch.to(device)
                    labels = labels.to(device).type(torch.float32)

                    attn_score, outputs = model(inputs_torch)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                print(f'{epoch} val_loss :', val_loss / (i + 1))
                val_loss_list.append((val_loss / (i + 1)))

            print("epoch {} finished".format(epoch))

            # 10번 째마다 train_loss 및 val_loss plot 저장
            if epoch % 10 == 0:
                save_fig_dir = './loss_figure/{}/{}'.format(self.args.save_f, self.args.condition)
                os.makedirs(save_fig_dir, exist_ok=True)

                plt.plot(train_loss_list, label='train')
                plt.plot(val_loss_list, label='valid')
                plt.legend()
                plt.savefig(os.path.join(save_fig_dir, 'loss_plot.png'))
                plt.clf()
                plt.close('all')

            ## val_loss가 감소하게 되면 해당 모델의 weight들 저장
            ## val_loss가 patience 횟수만큼 감소하지 않으면 학습 종료(early_stopping)
            if val_loss < best_loss:
                best_loss = val_loss
                es = 0

                save_dir = './save_models/{}/{}'.format(self.args.save_f, self.args.condition)
                os.makedirs(save_dir, exist_ok=True)
                save_dir_lst = os.listdir(save_dir)

                if len(save_dir_lst) == 0:
                    torch.save(model.state_dict(),
                               os.path.join(save_dir, str(epoch) + ' epoch ' + str(round(val_loss, 3)) + ' best_model.pt'))

                else:
                    past_best = sorted(save_dir_lst)[0]
                    os.remove(os.path.join(save_dir, past_best))

                    torch.save(model.state_dict(),
                               os.path.join(save_dir, str(epoch) + ' epoch ' + str(round(val_loss, 3)) + ' best_model.pt'))

            else:
                es += 1
                print("Counter {} of {}".format(es, self.args.patience))

                if es == self.args.patience:

                    # Epoch에 관계 없이 마지막 train_loss 및 val_loss plot 저장
                    save_fig_dir = './loss_figure/{}/{}'.format(self.args.save_f, self.args.condition)
                    os.makedirs(save_fig_dir, exist_ok=True)

                    plt.plot(train_loss_list, label='train')
                    plt.plot(val_loss_list, label='valid')
                    plt.legend()
                    plt.savefig(os.path.join(save_fig_dir, 'loss_plot.png'))
                    plt.clf()
                    plt.close('all')

                    print("Early stopping with best loss : ", best_loss, "and val_loss for this epoch: ", val_loss, "...")
                    break
