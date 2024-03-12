import sys
from os import makedirs
from os.path import join, basename, splitext
from time import time

from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from skimage.color import ycbcr2rgb

from models.aff import ColorWrapper, AffWrapper, DDNet, AAplusModel
from models.fft_blur_deconv_pytorch import psf2otf
from models.fade2black_matrix import FadeToBlackMatrix
from utils.read import RestorationsEvalColorDataset

import config


class Preprocess_deblur():
    def __call__(self, X, gpu):
        with torch.no_grad():
            y, y_color, psf = X

            y = y.float()
            y_color = y_color.float()

            hi, wi = y.shape[-2:]
            hi_ext = (hi//4 + 1)*4-hi
            wi_ext = (wi//4 + 1)*4-wi
            or_size = [hi_ext, wi_ext]
            l_pad = config.BORDER
            r_pad = config.BORDER+wi_ext
            u_pad = config.BORDER
            b_pad = config.BORDER+hi_ext
            y = torch.nn.functional.pad(y, (l_pad, r_pad, u_pad, b_pad), mode='replicate')
            y_color = torch.nn.functional.pad(y_color, (l_pad, r_pad, u_pad, b_pad), mode='replicate')

            H = psf2otf(psf, (y.size(2)+config.MAX_PSF_SIZE*2, y.size(3)+config.MAX_PSF_SIZE*2)).float()
            H = H.squeeze().unsqueeze(dim=0).unsqueeze(dim=1)

            if gpu:
                y = Variable(y.cuda()/255.0)
                y_color = Variable(y_color.cuda()/255.0)
                H = Variable(H.cuda())
            else:
                y = Variable(y/255.0)
                y_color = Variable(y_color/255.0)
                H = Variable(H)

            psf_size = psf.shape[-2:]

            fd_black = FadeToBlackMatrix(y.shape[-2:], [config.MAX_PSF_SIZE, config.MAX_PSF_SIZE]).cuda()
            amortised_model = AAplusModel(H, fd_black=fd_black,
                                          epsilon=config.EPS_WIENER)
            x_r       = amortised_model.Aplus_only(y)
            x_r_color = amortised_model.Aplus_only(y_color)
            data       = [amortised_model, x_r, y]
            data_color = [x_r_color, y_color]

            return data, data_color, 0, psf_size, [l_pad, r_pad, u_pad, b_pad]


if __name__ == '__main__':
    '''
    Parameters for data
    '''
    eval_file_path      = sys.argv[1]
    eval_file_psf_path  = sys.argv[2]
    path_out            = sys.argv[3]
    w_path_save         = sys.argv[4]
    w_color_path_save   = sys.argv[5]

    batch_size = {}
    batch_size['test'] = 1

    print('Loading model...')

    '''
    Make model
    '''
    #Generator g_theta(x)
    G = AffWrapper(
            DDNet(
                channels_c=1, 
                n_features=config.N_FEATURES, 
                nb=config.N_DENSE_BLOCKS, 
                df_size=config.DYNAMIC_FILTER_SIZE
            ), 
        df_size=config.DYNAMIC_FILTER_SIZE
    )

    #Generator color
    Gc = ColorWrapper(
            DDNet(channels_c=2, 
                  n_features=config.N_FEATURES, 
                  nb=config.N_DENSE_BLOCKS_COLOR
            )
        )
    
    G.load(w_path_save)
    Gc.load(w_color_path_save)

    #Move to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    G = G.to(device)
    Gc = Gc.to(device)

    '''
    Read test data
    '''
    print('Reading test files...')
    test_dataset = RestorationsEvalColorDataset(
        eval_file_path,
        eval_file_psf_path,
        ['.png', '.jpg'],
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size['test'],
        shuffle=True,
        num_workers=4
    )
    print(f'{len(test_dataset.filelist)} files read')

    #Create output dir
    makedirs(path_out, exist_ok=True)

    '''
    Test model
    '''
    #Preprocessing of test data
    preprocess_test = Preprocess_deblur()

    m_time = 0.0

    for i in range(len(test_dataset.filelist)):
        data = [d.unsqueeze(dim=0) for d in test_dataset.__getitem__(i)]
        data_y, data_color, label, psf_size, pads = preprocess_test(data, device)
        
        start_time = time()
        p, _ = G(data_y)
        p_color, _ = Gc(data_color)
        m_time += time()-start_time
        x_r_red = p.squeeze(dim=0).cpu().detach();
        x_r_red_color = p_color.squeeze(dim=0).cpu().detach();
        l_pad, r_pad, u_pad, b_pad = pads
        x_r_red = torch.cat([x_r_red, x_r_red_color], dim=0)[..., u_pad:-b_pad, l_pad:-r_pad].contiguous()

        f = test_dataset.__filename__(i)
        name = splitext(basename(f))[0]
        fname = join(path_out, name+'_ddnet.png')
        img = np.clip(ycbcr2rgb(np.clip(x_r_red.numpy().transpose((1, 2, 0)) * 255, 0, 255)) * 255, 0, 255)#[:, :, ::-1]
        img = img.astype(np.uint8)
        Image.fromarray(img).save(fname)

    m_time /= float(len(test_dataset.filelist))

    print("---------------------------------------------------------------------------------")
    print(f"Time per image: {m_time:8.4f}\t")
