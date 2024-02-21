from os import getpid
from os.path import join
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .history_log import History_Log, Epoch_Log
from .lr_schedulers import lr_dummy_scheduler


def preprocess_data_iter(X, gpu):
    X, Y = X

    if gpu:
        X, Y = Variable(X.cuda()), Variable(Y.cuda())
    else:
        X, Y = Variable(X), Variable(Y)

    len_data = len(X)

    return X, Y, len_data


def preprocess_data_multiple_input_output_iter(X, gpu):
    X, Y = X
    if gpu:
        X, Y = [Variable(x.cuda()) for x in X], [Variable(y.cuda()) for y in Y]
    else:
        X, Y = [Variable(x) for x in X], [Variable(y) for y in Y]

    len_data = len(X[0])

    return X, Y, len_data


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.grad_clip = 0
        self.training_params_and_info = {}

    def fit_epoch(self, loss_fn, epoch, data_loader, opt, lr_scheduler,
                  step_loggers, log_interval, epoch_log, preprocess_func):
        #Set mode to training
        self.train()

        #Are we going to run on gpu?
        gpu = next(self.parameters()).is_cuda

        #Information variables
        pid = getpid()
        train_loss = 0.0
        step_loss = 0.0
        data_idx = 0

        #Gradients
        grad_of_params = {}

        for name, parameter in self.named_parameters():
            grad_of_params[name] = 0.0

        #Main loop
        for batch_idx, X in enumerate(data_loader):
            #Sample X, Y
            X, Y, len_data = preprocess_func(X, gpu)

            #Train
            self.zero_grad()
            pY = self(X)
            loss = loss_fn(pY, Y)
            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

            #Gradients
            #for name, parameter in self.named_parameters():
            #    if parameter.requires_grad:
            #        grad_of_params[name] += parameter.grad.norm(2).cpu().numpy()/float(log_interval)

            if isinstance(opt, dict):
                for k in opt.keys():
                    opt[k].step()
            else:
                opt.step()

            #Inform
            step_loss += loss.item()
            train_loss += loss.item()
            data_idx += len_data

            if (batch_idx+1) % log_interval == 0:
                #Log step info
                info = {'data_idx': data_idx,
                        'len_data': len(data_loader.sampler),
                        'n_batches': len(data_loader),
                        'step': step_loss/float(log_interval),
                        'train': train_loss/float(batch_idx+1)
                        }
                info = {**info, **grad_of_params}
                epoch_log.log_iter_info(batch_idx+1, info)

                #Exect step loggers
                for logger in step_loggers:
                    logger.iter_log(epoch_log)

                #Reset information variables
                step_loss = 0.0

                #Gradients
                #for name, parameter in self.named_parameters():
                #    grad_of_params[name] = 0.0

            #Update lr
            if isinstance(opt, dict):
                for k in opt.keys():
                    pre_lr = opt[k].param_groups[0]['lr']
                    lr = lr_scheduler[k].update(pre_lr, train_loss/float(batch_idx+1))

                    if pre_lr != lr:
                        for param_group in opt[k].param_groups:
                            param_group['lr'] = lr
            else:
                pre_lr = opt.param_groups[0]['lr']
                lr = lr_scheduler.update(pre_lr, train_loss/float(batch_idx+1))

                if pre_lr != lr:
                    for param_group in opt.param_groups:
                        param_group['lr'] = lr

        #Inform train loss
        train_loss /= float(len(data_loader))
        epoch_log.log_info([('train_loss', train_loss), ])

        for logger in step_loggers:
            logger.iter_log_end_epoch(epoch_log)

        return train_loss

    def eval_epoch(self, data_loader, metrics, preprocess_func, return_py=0):
        self.eval()
        gpu = next(self.parameters()).is_cuda

        with torch.no_grad():
            def eval_epoch_1_dataset(data_loader_local, return_py_local, preprocess_func_l):
                eval_losses_local = {}

                for key in metrics.keys():
                    eval_losses_local[key] = 0.0

                data_target = data_loader_local.dataset[0]

                if return_py_local:
                    data, target = data_target
                    py = torch.zeros([len(data_loader_local.dataset), ] + list(target.size()))

                ini = 0

                for data_target in data_loader_local:
                    data, target, len_data = preprocess_func_l(data_target, gpu)

                    output = self(data)
                    end = ini+len_data

                    if return_py_local!=0:
                        if type(output) is list:
                            py[ini:end] = output[return_py].data.cpu()
                        else:
                            py[ini:end] = output.data.cpu()

                    ini = end

                    for key, metric in metrics.items():
                        eval_losses_local[key] += metric(output, target).cpu().numpy() if gpu else metric(output, target).numpy()

                for key in metrics.keys():
                    if len(eval_losses_local[key].shape) == 0:
                        eval_losses_local[key] /= float(len(data_loader_local))
                    else:
                        eval_losses_local[key] = metrics[key].combine(eval_losses_local[key])

                if return_py_local:
                    return eval_losses_local, py
                else:
                    return eval_losses_local

            tipo = 0

            if isinstance(data_loader, list):
                tipo = 1
            elif isinstance(data_loader, dict):
                tipo = 2

            if tipo==0:
                return eval_epoch_1_dataset(data_loader, return_py, preprocess_func)
            elif tipo==1:
                eval_losses_temp = [eval_epoch_1_dataset(data_loader_local, False, preprocess_func) for data_loader_local in data_loader]

                eval_losses = {}

                for i, eval_losses_key in enumerate(eval_losses_temp):
                    for key in metrics.keys():
                        eval_losses['evalset %d %s' % (i+1, key)] = eval_losses_key[key]

                return eval_losses
            else:
                eval_losses = {}

                for key_data, data_loader_local in data_loader.items():
                    eval_losses_key = eval_epoch_1_dataset(data_loader_local, False, preprocess_func[key_data])

                    for key in metrics.keys():
                        eval_losses['%s %s' % (key_data, key)] = eval_losses_key[key]

                return eval_losses

    def fit(self, loss_fn, opt_class, opt_args, epochs,
              train_loader, eval_loader=None, ini_epoch=0,
              preprocess_func=preprocess_data_iter,
              preprocess_func_eval=None,
              lr_scheduler=lr_dummy_scheduler(),
              metrics={}, log_interval=1, step_loggers=[], epoch_loggers=[],
              path_save=None, return_py=0, loss_reset=999999):

        if isinstance(opt_class, dict):
            opt = {}

            for k in opt_class.keys():
                opt[k] = opt_class[k](self.get_parameters()[k], **opt_args[k])
        else:
            opt = opt_class(self.get_parameters(), **opt_args)

        eval_set = eval_loader is not None
        preprocess_func_eval = preprocess_func if preprocess_func_eval is None else preprocess_func_eval

        history_log = History_Log(getpid())
        history_log.log_info(opt_args.items())
        history_log.log_info([('loss fn', str(loss_fn)),
                              ('eval_set', eval_set)])

        self.training_params_and_info['loss_fn'] = loss_fn
        self.training_params_and_info['opt'] = opt
        self.training_params_and_info['opt_class'] = opt_class
        self.training_params_and_info['opt_args'] = opt_args
        self.training_params_and_info['preprocess_func'] = preprocess_func
        self.training_params_and_info['preprocess_func_eval'] = preprocess_func_eval
        self.training_params_and_info['epochs'] = epochs
        self.training_params_and_info['current_epoch'] = ini_epoch
        self.training_params_and_info['train_loader'] = train_loader
        self.training_params_and_info['eval_loader'] = eval_loader
        self.training_params_and_info['eval_set'] = eval_set
        self.training_params_and_info['lr_scheduler'] = lr_scheduler
        self.training_params_and_info['metrics'] = metrics
        self.training_params_and_info['log_interval'] = log_interval
        self.training_params_and_info['step_loggers'] = step_loggers
        self.training_params_and_info['epoch_loggers'] = epoch_loggers
        self.training_params_and_info['path_save'] = path_save
        self.training_params_and_info['return_py'] = return_py
        self.training_params_and_info['history_log'] = history_log
        self.training_params_and_info['loss_reset'] = loss_reset

        self.__fit()

    def __fit(self):
        loss_fn = self.training_params_and_info['loss_fn']
        opt = self.training_params_and_info['opt']
        opt_class = self.training_params_and_info['opt_class']
        opt_args = self.training_params_and_info['opt_args']
        preprocess_func = self.training_params_and_info['preprocess_func']
        preprocess_func_eval = self.training_params_and_info['preprocess_func_eval']
        epochs = self.training_params_and_info['epochs']
        current_epoch = self.training_params_and_info['current_epoch']
        train_loader = self.training_params_and_info['train_loader']
        eval_loader = self.training_params_and_info['eval_loader']
        eval_set = self.training_params_and_info['eval_set']
        lr_scheduler = self.training_params_and_info['lr_scheduler']
        metrics = self.training_params_and_info['metrics']
        log_interval = self.training_params_and_info['log_interval']
        step_loggers = self.training_params_and_info['step_loggers']
        epoch_loggers = self.training_params_and_info['epoch_loggers']
        path_save = self.training_params_and_info['path_save']
        return_py = self.training_params_and_info['return_py']
        history_log = self.training_params_and_info['history_log']
        loss_reset = self.training_params_and_info['loss_reset']

        epoch = current_epoch + 1
        #for epoch in range(current_epoch + 1, epochs + 1):
        while epoch < epochs + 1:
            epoch_log = Epoch_Log(epoch)

            if isinstance(opt, dict):
                for k in opt.keys():
                    epoch_log.log_info([('lr_ini_'+k, opt[k].param_groups[0]['lr']),])
            else:
                epoch_log.log_info([('lr_ini', opt.param_groups[0]['lr']),])

            epoch_time = time()
            train_loss = self.fit_epoch(loss_fn, epoch, train_loader, opt, lr_scheduler,
                           step_loggers, log_interval, epoch_log, preprocess_func)
            epoch_time = time() - epoch_time

            if isinstance(opt, dict):
                for k in opt.keys():
                    epoch_log.log_info([('lr_end_'+k, opt[k].param_groups[0]['lr']),
                                        ('epoch_time', epoch_time)])
            else:
                epoch_log.log_info([('lr_end', opt.param_groups[0]['lr']),
                                    ('epoch_time', epoch_time)])

            if eval_set:
                if return_py == 1:
                    eval_losses, pY_eval = self.eval_epoch(eval_loader, metrics,
                                                  preprocess_func_eval,
                                                  return_py=1)

                    epoch_log.log_info(eval_losses.items())
                    history_log.eval_out = pY_eval
                else:
                    eval_losses = self.eval_epoch(eval_loader, metrics,
                                                  preprocess_func_eval,
                                                  return_py=0)

                    epoch_log.log_info(eval_losses.items())
                    #history_log.eval_out = pY_eval

            history_log.log_info_epoch(epoch_log)

            for logger in epoch_loggers:
                logger.epoch_log(history_log)

            if path_save is not None:
                history_log.save(join(path_save, 'history_log.pkl'))
                self.save(join(path_save, 'w.'+str(epoch)+'.pt'))

            #Update lr
            self.training_params_and_info['current_epoch'] = epoch

            #Reset
            if (loss_reset <= train_loss) and (epoch > 1):
                if (path_save is not None):
                    epoch -= 1
                    self.load(join(path_save, 'w.'+str(epoch)+'.pt'))

                    if isinstance(opt, dict):
                        new_opt = {}

                        for k in opt_class.keys():
                            opt_args_aux = {}

                            for k2 in opt_args[k].keys():
                                opt_args_aux[k2] = opt_args[k][k2]
                            opt_args_aux['lr'] = lr

                            new_opt[k] = opt_class[k](self.get_parameters()[k], **opt_args_aux)

                        opt = new_opt
                    else:
                        opt_args_aux = {}
                        for k in opt_args.keys():
                            opt_args_aux[k] = opt_args[k]
                        opt_args_aux['lr'] = lr

                        opt = opt_class(self.get_parameters(), **opt_args_aux)
            else:
                if isinstance(opt, dict):
                    for k in opt.keys():
                        pre_lr = opt[k].param_groups[0]['lr']
                        lr = lr_scheduler[k].epoch_end(pre_lr, epoch_log['train_loss'])

                        if pre_lr != lr:
                            for param_group in opt[k].param_groups:
                                param_group['lr'] = lr
                else:
                    pre_lr = opt.param_groups[0]['lr']
                    lr = lr_scheduler.epoch_end(pre_lr, epoch_log['train_loss'])

                    if pre_lr != lr:
                        for param_group in opt.param_groups:
                            param_group['lr'] = lr

            epoch += 1

    def predict(self, X, out_shape, batch_size=64):
        self.eval()
        Y = torch.zeros([X.size(0),] + list(out_shape))
        n = X.size(0)//batch_size

        try:
            gpu = next(self.parameters()).is_cuda
        except StopIteration:
            gpu = True

        with torch.no_grad():
            for i in range(n):
                ini = i*batch_size
                end = ini+batch_size

                if gpu:
                    Y[ini:end] = self(Variable(X[ini:end].cuda())).data.cpu()
                else:
                    Y[ini:end] = self(Variable(X[ini:end])).data

            end = batch_size*n

            if end < X.size(0):
                ini = end
                end = X.size(0)

                if gpu:
                    Y[ini:end] = self(Variable(X[ini:end].cuda())).data.cpu()
                else:
                    Y[ini:end] = self(Variable(X[ini:end])).data

        return Y

    def initialize_kernel(self, kernel_initializers):
        def initialize_kernel(module, kernel_initializers):
            modules = [m for m in module.modules()]

            for m in modules[1:]:
                m_ls = [m_l for m_l in m.modules()]

                if len(m_ls) > 1:
                    initialize_kernel(m, kernel_initializers)
                else:
                    if hasattr(m, 'weight'):
                        key = m.__class__.__name__
                        if key in kernel_initializers:
                            kernel_initializers[key](m.weight)

        initialize_kernel(self, kernel_initializers)

    def initialize_bias(self, bias_initializer):
        def initialize_bias(module, bias_initializers):
            modules = [m for m in module.modules()]

            for m in modules[1:]:
                m_ls = [m_l for m_l in m.modules()]

                if len(m_ls) > 1:
                    initialize_bias(m, bias_initializers)
                else:
                    if hasattr(m, 'bias'):
                        key = m.__class__.__name__
                        if key in bias_initializers:
                            bias_initializers[key](m.weight)

        initialize_bias(self, bias_initializer)

    def save(self, path):
        state = self.state_dict()
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path)#, map_location=torch.device('cpu'))
        self.load_state_dict(state)

    def set_trainable(self, trainable):
        for p in self.parameters():
            p.requires_grad = trainable

    def set_trainable_module(self, module_name, trainable):
        module = getattr(self, module_name)

        for p in module.parameters():
            p.requires_grad = trainable

    def clone_layer_weights(self, model, layers_names):
        def copy_check(l1, l2):
            parameters1 = [p for p in l1.parameters()]
            parameters2 = [p for p in l2.parameters()]

            for i in range(len(parameters1)):
                p1, p2 = parameters1[i], parameters2[i]
                p1.data = p2.data.clone()

            if isinstance(l1, nn.BatchNorm1d) or isinstance(l1, nn.BatchNorm2d) or isinstance(l1, nn.BatchNorm3d):
                if l2.running_mean is not None:
                    l1.running_mean = l2.running_mean.clone()

                if l2.running_var is not None:
                    l1.running_var = l2.running_var.clone()

        for l in layers_names:
            layer_self = getattr(self, l)
            layer_model = getattr(model, l)
            modules1 = [m for m in layer_self.modules()]
            modules2 = [m for m in layer_model.modules()]

            if len(modules1) > 1:
                for l1, l2 in zip(modules1[1:], modules2[1:]):
                    copy_check(l1, l2)
            else:
                copy_check(layer_self, layer_model)

    def get_parameters(self):
        parameters = []

        for name, p in self.named_parameters():
            if p.requires_grad:
                parameters.append(p)

        return parameters


class BaseParrallelModel(nn.DataParallel):
    def __init__(self, model):
        super(BaseParrallelModel, self).__init__(model)
        self.grad_clip = 0
        self.training_params_and_info = {}

    def fit_epoch(self, loss_fn, epoch, data_loader, opt, lr_scheduler,
                  step_loggers, log_interval, epoch_log, preprocess_func):
        #Set mode to training
        self.train()

        #Are we going to run on gpu?
        gpu = next(self.parameters()).is_cuda

        #Information variables
        pid = getpid()
        train_loss = 0.0
        step_loss = 0.0
        data_idx = 0

        #Gradients
        grad_of_params = {}

        for name, parameter in self.named_parameters():
            grad_of_params[name] = 0.0

        #Main loop
        for batch_idx, X in enumerate(data_loader):
            #Sample X, Y
            X, Y, len_data = preprocess_func(X, gpu)

            #Train
            self.zero_grad()
            pY = self(X)
            loss = loss_fn(pY, Y)
            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

            #Gradients
            #for name, parameter in self.named_parameters():
            #    if parameter.requires_grad:
            #        grad_of_params[name] += parameter.grad.norm(2).cpu().numpy()/float(log_interval)
            if isinstance(opt, dict):
                for k in opt.keys():
                    opt[k].step()
            else:
                opt.step()

            #Inform
            step_loss += loss.item()
            train_loss += loss.item()
            data_idx += len_data

            if (batch_idx+1) % log_interval == 0:
                #Log step info
                info = {'data_idx': data_idx,
                        'len_data': len(data_loader.sampler),
                        'n_batches': len(data_loader),
                        'step': step_loss/float(log_interval),
                        'train': train_loss/float(batch_idx+1)
                        }
                info = {**info, **grad_of_params}
                epoch_log.log_iter_info(batch_idx+1, info)

                #Exect step loggers
                for logger in step_loggers:
                    logger.iter_log(epoch_log)

                #Reset information variables
                step_loss = 0.0

                #Gradients
                #for name, parameter in self.named_parameters():
                #    grad_of_params[name] = 0.0

            #Update lr
            if isinstance(opt, dict):
                for k in opt.keys():
                    pre_lr = opt[k].param_groups[0]['lr']
                    lr = lr_scheduler[k].update(pre_lr, train_loss/float(batch_idx+1))

                    if pre_lr != lr:
                        for param_group in opt[k].param_groups:
                            param_group['lr'] = lr
            else:
                pre_lr = opt.param_groups[0]['lr']
                lr = lr_scheduler.update(pre_lr, train_loss/float(batch_idx+1))

                if pre_lr != lr:
                    for param_group in opt.param_groups:
                        param_group['lr'] = lr

        #Inform train loss
        train_loss /= float(len(data_loader))
        epoch_log.log_info([('train_loss', train_loss), ])

        for logger in step_loggers:
            logger.iter_log_end_epoch(epoch_log)

        return train_loss

    def eval_epoch(self, data_loader, metrics, preprocess_func, return_py=0):
        self.eval()
        gpu = next(self.parameters()).is_cuda

        with torch.no_grad():
            def eval_epoch_1_dataset(data_loader_local, return_py_local, preprocess_func_l):
                eval_losses_local = {}

                for key in metrics.keys():
                    eval_losses_local[key] = 0.0

                data_target = data_loader_local.dataset[0]

                if return_py_local:
                    data, target = data_target
                    py = torch.zeros([len(data_loader_local.dataset), ] + list(target.size()))

                ini = 0

                for data_target in data_loader_local:
                    data, target, len_data = preprocess_func_l(data_target, gpu)

                    output = self(data)
                    end = ini+len_data

                    if return_py_local!=0:
                        if type(output) is list:
                            py[ini:end] = output[return_py].data.cpu()
                        else:
                            py[ini:end] = output.data.cpu()

                    ini = end

                    for key, metric in metrics.items():
                        eval_losses_local[key] += metric(output, target).cpu().numpy() if gpu else metric(output, target).numpy()

                for key in metrics.keys():
                    if len(eval_losses_local[key].shape) == 0:
                        eval_losses_local[key] /= float(len(data_loader_local))
                    else:
                        eval_losses_local[key] = metrics[key].combine(eval_losses_local[key])

                if return_py_local:
                    return eval_losses_local, py
                else:
                    return eval_losses_local

            tipo = 0

            if isinstance(data_loader, list):
                tipo = 1
            elif isinstance(data_loader, dict):
                tipo = 2

            if tipo==0:
                return eval_epoch_1_dataset(data_loader, return_py, preprocess_func)
            elif tipo==1:
                eval_losses_temp = [eval_epoch_1_dataset(data_loader_local, False, preprocess_func) for data_loader_local in data_loader]

                eval_losses = {}

                for i, eval_losses_key in enumerate(eval_losses_temp):
                    for key in metrics.keys():
                        eval_losses['evalset %d %s' % (i+1, key)] = eval_losses_key[key]

                return eval_losses
            else:
                eval_losses = {}

                for key_data, data_loader_local in data_loader.items():
                    eval_losses_key = eval_epoch_1_dataset(data_loader_local, False, preprocess_func[key_data])

                    for key in metrics.keys():
                        eval_losses['%s %s' % (key_data, key)] = eval_losses_key[key]

                return eval_losses

    def fit(self, loss_fn, opt_class, opt_args, epochs,
              train_loader, eval_loader=None, ini_epoch=0,
              preprocess_func=preprocess_data_iter,
              preprocess_func_eval=None,
              lr_scheduler=lr_dummy_scheduler(),
              metrics={}, log_interval=1, step_loggers=[], epoch_loggers=[],
              path_save=None, return_py=0, loss_reset=999999):
        opt = opt_class(self.get_parameters(), **opt_args)

        eval_set = eval_loader is not None
        preprocess_func_eval = preprocess_func if preprocess_func_eval is None else preprocess_func_eval

        history_log = History_Log(getpid())
        history_log.log_info(opt_args.items())
        history_log.log_info([('loss fn', str(loss_fn)),
                              ('eval_set', eval_set)])

        self.training_params_and_info['loss_fn'] = loss_fn
        self.training_params_and_info['opt'] = opt
        self.training_params_and_info['opt_class'] = opt_class
        self.training_params_and_info['opt_args'] = opt_args
        self.training_params_and_info['preprocess_func'] = preprocess_func
        self.training_params_and_info['preprocess_func_eval'] = preprocess_func_eval
        self.training_params_and_info['epochs'] = epochs
        self.training_params_and_info['current_epoch'] = ini_epoch
        self.training_params_and_info['train_loader'] = train_loader
        self.training_params_and_info['eval_loader'] = eval_loader
        self.training_params_and_info['eval_set'] = eval_set
        self.training_params_and_info['lr_scheduler'] = lr_scheduler
        self.training_params_and_info['metrics'] = metrics
        self.training_params_and_info['log_interval'] = log_interval
        self.training_params_and_info['step_loggers'] = step_loggers
        self.training_params_and_info['epoch_loggers'] = epoch_loggers
        self.training_params_and_info['path_save'] = path_save
        self.training_params_and_info['return_py'] = return_py
        self.training_params_and_info['history_log'] = history_log
        self.training_params_and_info['loss_reset'] = loss_reset

        self.__fit()

    def __fit(self):
        loss_fn = self.training_params_and_info['loss_fn']
        opt = self.training_params_and_info['opt']
        opt_class = self.training_params_and_info['opt_class']
        opt_args = self.training_params_and_info['opt_args']
        preprocess_func = self.training_params_and_info['preprocess_func']
        preprocess_func_eval = self.training_params_and_info['preprocess_func_eval']
        epochs = self.training_params_and_info['epochs']
        current_epoch = self.training_params_and_info['current_epoch']
        train_loader = self.training_params_and_info['train_loader']
        eval_loader = self.training_params_and_info['eval_loader']
        eval_set = self.training_params_and_info['eval_set']
        lr_scheduler = self.training_params_and_info['lr_scheduler']
        metrics = self.training_params_and_info['metrics']
        log_interval = self.training_params_and_info['log_interval']
        step_loggers = self.training_params_and_info['step_loggers']
        epoch_loggers = self.training_params_and_info['epoch_loggers']
        path_save = self.training_params_and_info['path_save']
        return_py = self.training_params_and_info['return_py']
        history_log = self.training_params_and_info['history_log']
        loss_reset = self.training_params_and_info['loss_reset']

        epoch = current_epoch + 1
        #for epoch in range(current_epoch + 1, epochs + 1):
        while epoch < epochs + 1:
            epoch_log = Epoch_Log(epoch)
            epoch_log.log_info([('lr_ini', opt.param_groups[0]['lr']),])

            epoch_time = time()
            train_loss = self.fit_epoch(loss_fn, epoch, train_loader, opt, lr_scheduler,
                           step_loggers, log_interval, epoch_log, preprocess_func)
            epoch_time = time() - epoch_time

            epoch_log.log_info([('lr_end', opt.param_groups[0]['lr']),
                                ('epoch_time', epoch_time)])

            if eval_set:
                if return_py == 1:
                    eval_losses, pY_eval = self.eval_epoch(eval_loader, metrics,
                                                  preprocess_func_eval,
                                                  return_py=1)

                    epoch_log.log_info(eval_losses.items())
                    history_log.eval_out = pY_eval
                else:
                    eval_losses = self.eval_epoch(eval_loader, metrics,
                                                  preprocess_func_eval,
                                                  return_py=0)

                    epoch_log.log_info(eval_losses.items())
                    #history_log.eval_out = pY_eval

            history_log.log_info_epoch(epoch_log)

            for logger in epoch_loggers:
                logger.epoch_log(history_log)

            if path_save is not None:
                history_log.save(join(path_save, 'history_log.pkl'))
                self.save(join(path_save, 'w.'+str(epoch)+'.pt'))

            #Update lr
            self.training_params_and_info['current_epoch'] = epoch

            #Reset
            if (loss_reset <= train_loss) and (epoch > 1):
                if (path_save is not None):
                    epoch -= 1
                    self.load(join(path_save, 'w.'+str(epoch)+'.pt'))

                    opt_args_aux = {}
                    for k in opt_args.keys():
                        opt_args_aux[k] = opt_args[k]
                    opt_args_aux['lr'] = lr

                    opt = opt_class(self.get_parameters(), **opt_args_aux)
            else:
                pre_lr = opt.param_groups[0]['lr']
                lr = lr_scheduler.epoch_end(pre_lr, epoch_log['train_loss'])

                if pre_lr != lr:
                    for param_group in opt.param_groups:
                        param_group['lr'] = lr

            epoch += 1

    def predict(self, X, out_shape, batch_size=64):
        self.eval()
        Y = torch.zeros([X.size(0),] + list(out_shape))
        n = X.size(0)//batch_size

        try:
            gpu = next(self.parameters()).is_cuda
        except StopIteration:
            gpu = True

        with torch.no_grad():
            for i in range(n):
                ini = i*batch_size
                end = ini+batch_size

                if gpu:
                    Y[ini:end] = self(Variable(X[ini:end].cuda())).data.cpu()
                else:
                    Y[ini:end] = self(Variable(X[ini:end])).data

            end = batch_size*n

            if end < X.size(0):
                ini = end
                end = X.size(0)

                if gpu:
                    Y[ini:end] = self(Variable(X[ini:end].cuda())).data.cpu()
                else:
                    Y[ini:end] = self(Variable(X[ini:end])).data

        return Y

    def initialize_kernel(self, kernel_initializers):
        def initialize_kernel(module, kernel_initializers):
            modules = [m for m in module.modules()]

            for m in modules[1:]:
                m_ls = [m_l for m_l in m.modules()]

                if len(m_ls) > 1:
                    initialize_kernel(m, kernel_initializers)
                else:
                    if hasattr(m, 'weight'):
                        key = m.__class__.__name__
                        if key in kernel_initializers:
                            kernel_initializers[key](m.weight)

        initialize_kernel(self, kernel_initializers)

    def initialize_bias(self, bias_initializer):
        def initialize_bias(module, bias_initializers):
            modules = [m for m in module.modules()]

            for m in modules[1:]:
                m_ls = [m_l for m_l in m.modules()]

                if len(m_ls) > 1:
                    initialize_bias(m, bias_initializers)
                else:
                    if hasattr(m, 'bias'):
                        key = m.__class__.__name__
                        if key in bias_initializers:
                            bias_initializers[key](m.weight)

        initialize_bias(self, bias_initializer)

    def save(self, path):
        state = self.state_dict()
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path)
        self.load_state_dict(state)

    def set_trainable(self, trainable):
        for p in self.parameters():
            p.requires_grad = trainable

    def set_trainable_module(self, module_name, trainable):
        module = getattr(self, module_name)

        for p in module.parameters():
            p.requires_grad = trainable

    def clone_layer_weights(self, model, layers_names):
        def copy_check(l1, l2):
            parameters1 = [p for p in l1.parameters()]
            parameters2 = [p for p in l2.parameters()]

            for i in range(len(parameters1)):
                p1, p2 = parameters1[i], parameters2[i]
                p1.data = p2.data.clone()

            if isinstance(l1, nn.BatchNorm1d) or isinstance(l1, nn.BatchNorm2d) or isinstance(l1, nn.BatchNorm3d):
                if l2.running_mean is not None:
                    l1.running_mean = l2.running_mean.clone()

                if l2.running_var is not None:
                    l1.running_var = l2.running_var.clone()

        for l in layers_names:
            layer_self = getattr(self, l)
            layer_model = getattr(model, l)
            modules1 = [m for m in layer_self.modules()]
            modules2 = [m for m in layer_model.modules()]

            if len(modules1) > 1:
                for l1, l2 in zip(modules1[1:], modules2[1:]):
                    copy_check(l1, l2)
            else:
                copy_check(layer_self, layer_model)

    def get_parameters(self):
        return self.module.get_parameters()
