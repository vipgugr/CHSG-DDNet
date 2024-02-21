import sys
from os import makedirs
from os.path import join

import torch
import torch.optim as optim
from torch.autograd import Variable

from models.aff import ColorWrapper, DDNet, AAplusModel
from models.fade2black_matrix import FadeToBlackMatrixNoPad
from models.tse.loggers import console_logger, file_logger
from models.tse.lr_schedulers import lr_step_epoch_scheduler
from utils.read import RestorationsColorOnlyDataset
from utils.losses import CharbonnierLoss
from utils.metrics import PSNRWrapper, SSIMWrapper

import config


class Preprocess_deblur():
    def __init__(self, epsilon, fd_black):
        self.epsilon = epsilon
        self.fd_black = fd_black

    def __call__(self, X, gpu):
        with torch.no_grad():
            x, y, H = X
            H = H.unsqueeze(dim=1)

            if gpu:
                x = Variable(x.float().cuda()/255.0)
                y = Variable(y.float().cuda()/255.0)
                H = Variable(H.float().cuda())
            else:
                x = Variable(x.float()/255.0)
                y = Variable(y.float()/255.0)
                H = Variable(H.float())

            amortised_model = AAplusModel(H, fd_black=self.fd_black,
                                          epsilon=self.epsilon)
            x_r = amortised_model.Aplus_only(y)
            y = y[:, :, config.MAX_PSF_SIZE:-config.MAX_PSF_SIZE, config.MAX_PSF_SIZE:-config.MAX_PSF_SIZE].contiguous()
            x = x[:, :, config.MAX_PSF_SIZE:-config.MAX_PSF_SIZE, config.MAX_PSF_SIZE:-config.MAX_PSF_SIZE].contiguous()
            data = [x_r, y]

            return data, x, x.size(0)


if __name__ == '__main__':
    '''
    Parameters for data
    '''
    batch_size = {}
    batch_size['train'] = config.BATCH_SIZE
    batch_size['val'] = 1
    #torch.backends.cudnn.benchmark = True
    #torch.set_num_threads(4)

    '''
    Read train and test data
    '''
    print('Reading train...')
    train_dataset = RestorationsColorOnlyDataset(config.TRAIN_FILE_PATH, ['.npy', ], config.IM_SIZE, config.SIGMA_NOISE)
    sampler = torch.utils.data.sampler.\
            WeightedRandomSampler(weights=torch.ones(len(train_dataset)).double(),
                                  num_samples=config.BATCH_SIZE*config.ITERS_PER_EPOCH)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size['train'],
                                                sampler=sampler,
                                                num_workers=4)

    print(f'{len(train_dataset.filelist)} files read')

    print('Reading eval...')
    eval_dataset = RestorationsColorOnlyDataset(config.EVAL_FILE_PATH, ['.npy', ], config.IM_SIZE, config.SIGMA_NOISE)
    eval_data_loader = torch.utils.data.DataLoader(eval_dataset,
                                                   batch_size=batch_size['val'],
                                                   shuffle=False,
                                                   num_workers=4)
    print(f'{len(eval_dataset.filelist)} files read')

    '''
    Train model
    '''
    #Make model
    G = ColorWrapper(
            DDNet(channels_c=2, 
                  n_features=config.N_FEATURES, 
                  nb=config.N_DENSE_BLOCKS_COLOR
            )
        ).cuda()
    
    #Create output dir
    makedirs(config.W_COLOR_PATH_SAVE, exist_ok=True)

    #Losses and metrics to print
    keys_loss = [
        'train_loss',
        'eval_loss',
        'eval_PSNR',
        'eval_SSIM',
    ]

    '''
    Train bicubic only first
    '''
    #Preprocessing of train data
    fd_black = FadeToBlackMatrixNoPad(
        [config.IM_SIZE+2*config.MAX_PSF_SIZE, config.IM_SIZE+2*config.MAX_PSF_SIZE], 
        [config.MAX_PSF_SIZE, config.MAX_PSF_SIZE]
    )
    fd_black = fd_black.cuda()
    preprocess_train = Preprocess_deblur(epsilon=config.EPS_WIENER, fd_black=fd_black)

    #Train
    G.fit(loss_fn=CharbonnierLoss(eps=config.EPS), 
          opt_class=optim.Adam, 
          opt_args={'lr': config.LR_1_COLOR, 'weight_decay': config.W_DECAY},
          epochs=config.EPOCHS_COLOR,
          train_loader=train_data_loader, 
          eval_loader=eval_data_loader,
          preprocess_func=preprocess_train,
          ini_epoch=0,
          lr_scheduler=\
            lr_step_epoch_scheduler(
                steps=[config.LR_STEP_1_COLOR, config.LR_STEP_2_COLOR, config.EPOCHS],
                lrs=[config.LR_1_COLOR, config.LR_2_COLOR, config.LR_3_COLOR]
            ),
          metrics=\
            {
              'eval_loss': CharbonnierLoss(eps=config.EPS), 
              'eval_PSNR': PSNRWrapper(config.BORDER),
              'eval_SSIM': SSIMWrapper(config.BORDER, channel=2)
            },
          log_interval=30,
          step_loggers=[console_logger(['step',  'train']),],
          epoch_loggers=\
            [
                console_logger(keys_loss),
                file_logger(keys_loss, join(config.W_COLOR_PATH_SAVE, 'epoch_log.txt'), False),
            ],
          path_save=config.W_COLOR_PATH_SAVE)
