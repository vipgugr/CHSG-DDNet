from os import makedirs
from os.path import exists, join

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from cv2 import imwrite


class console_logger():
    def __init__(self, keys, ndec=6):
        self.keys = keys
        self.ndec = int(ndec)

    def make_string_epoch(self, history_log):
        epoch_log = history_log['info_epochs'][-1]
        string_epoch = '\nFinished epoch {} in {:.2f} seconds.'\
                        .format(epoch_log['epoch'], epoch_log['epoch_time'])

        for k in self.keys:
            string_epoch += ('\t{}:{:.'+str(self.ndec)+'f}').format(k, epoch_log[k])

        return string_epoch

    def make_string_iter(self, epoch_log):
        iter_log = epoch_log['iters_info'][-1]
        string_step = 'Train Epoch: {:04d} [{:06d}/{:06d} ({:03d}%)]'\
                      .format(epoch_log['epoch'],
                              iter_log['data_idx'],
                              iter_log['len_data'],
                              (100*iter_log['iter'])//iter_log['n_batches'])

        for k in self.keys:
            string_step += ('\t{}:{:.'+str(self.ndec)+'f}').format(k, iter_log[k])

        return string_step

    def epoch_log(self, history_log):
        string_epoch = self.make_string_epoch(history_log)
        end = '\n'

        print (string_epoch, end=end)

    def iter_log(self, epoch_log):
        string_iter = self.make_string_iter(epoch_log)
        string_iter = '\r' + string_iter
        end = ''

        print (string_iter, end=end)

    def iter_log_end_epoch(self, epoch_log):
        return


class file_logger(console_logger):
    def __init__(self, keys, file_path, append=True):
        super(file_logger, self).__init__(keys)
        self.file_path = file_path

        if not append:
            open(self.file_path, 'w').close()

    def epoch_log(self, history_log):
        string_epoch = self.make_string_epoch(history_log) + '\n'

        f = open(self.file_path, 'a')
        f.write(string_epoch)
        f.close()

    def iter_log(self, epoch_log):
        string_iter = self.make_string_iter(epoch_log) + '\n'

        f = open(self.file_path, 'a')
        f.write(string_iter)
        f.close()

    def iter_log_end_epoch(self, epoch_log):
        return


class plot_logger():
    def __init__(self, file_path, keys_and_color,
                xlabel='Epochs', ylabel='', title=''):
        self.file_path = file_path
        self.keys_and_color = keys_and_color
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

    def plot(self, log, key_x, key_iter, path):
        info_iters = log[key_iter]
        data = {}

        data['x'] = []

        for iter_log in info_iters:
            data['x'].append(iter_log[key_x])

        keys_and_color = self.keys_and_color

        for i, (sub_name, keys_and_color_sub) in enumerate(keys_and_color):
            plt.clf()

            for (k, c) in keys_and_color_sub:
                data[k] = []

                for iter_log in info_iters:
                    data[k].append(iter_log[k])

                plt.plot(data['x'], data[k], c, label=k)
                plt.legend(loc='best')

            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)
            plt.title('%s %s'%(self.title, sub_name))
            plt.tight_layout()
            plt.savefig('%s_%s.png'%(path, sub_name),
                        dpi=550.0, bbox_inches='tight')

    def epoch_log(self, history_log):
        self.plot(history_log, 'epoch', 'info_epochs', self.file_path)

    def iter_log(self, epoch_log):
        return

    def iter_log_end_epoch(self, epoch_log):
        path = self.file_path % epoch_log['epoch']
        self.plot(epoch_log, 'iter', 'iters_info', path)


'''
class image_logger():
    def __init__(self, file_path):
        self.file_path = file_path

        #Create output dir
        if not exists(self.file_path):
            makedirs(self.file_path)

    def epoch_log(self, history_log):
        epoch = history_log['info_epochs'][-1]['epoch']
        images = history_log.eval_out
        outdir = join(self.file_path, '%04d'%epoch)

        #Create output dir
        if not exists(outdir):
            makedirs(outdir)

        images = torch.unsqueeze(images, dim=4)
        images = images.transpose(1, 4)
        images = torch.squeeze(images, dim=1)
        images = (images*255).clamp(0, 255).numpy().astype('uint8')

        for i, im in enumerate(images):
            imwrite(join(outdir, '%04d.png'%i), im)
'''
