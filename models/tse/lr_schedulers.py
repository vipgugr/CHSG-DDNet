class lr_dummy_scheduler():
    def update(self, lr, train_error):
        return lr

    def epoch_end(self, lr, train_error):
        return lr


class lr_step_scheduler():
    def __init__(self, steps, alfa, min_lr):
        self.steps = steps
        self.alfa = alfa
        self.min_lr = min_lr
        self.update_idx = 0

    def update(self, lr, train_error):
        self.update_idx += 1

        if self.update_idx%self.steps == 0:
            return max(lr*self.alfa, self.min_lr)

        return lr

    def epoch_end(self, lr, train_error):
        return lr


class lr_lamb_scheduler():
    def __init__(self, lamb, min_lr):
        self.lamb = lamb
        self.min_lr = min_lr

    def update(self, lr, train_error):
        return lr

    def epoch_end(self, lr, train_error):
        return max(lr-self.lamb, self.min_lr)

        return lr


class lr_step_epoch_scheduler():
    def __init__(self, steps, lrs):
        self.steps = steps
        self.steps_idx = 0
        self.lrs = lrs
        self.update_idx = 0

    def update(self, lr, train_error):
        return lr

    def epoch_end(self, lr, train_error):
        self.update_idx += 1

        if self.update_idx - self.steps[self.steps_idx] == 0:
            self.steps_idx = min(self.steps_idx + 1, len(self.steps) - 1)

        return self.lrs[self.steps_idx]


class lr_traingular_epoch_scheduler():
    def __init__(self, cicle, lr_min, lr_max):
        self.cicle_2 = int(cicle/2)
        self.lr_ini = lr_min
        self.lr = lr_min
        self.lr_next = lr_max
        self.update_step = (self.lr_next-self.lr)/float(self.cicle_2)
        self.epoch = 0

    def update(self, lr, train_error):
        return lr

    def epoch_end(self, lr, train_error):
        self.epoch += 1
        self.lr = self.lr + self.update_step

        if self.epoch >= self.cicle_2:
            self.lr = self.lr_next
            self.lr_next = self.lr_ini
            self.lr_ini = self.lr
            self.update_step = (self.lr_next-self.lr)/float(self.cicle_2)
            self.epoch = 0

        return self.lr


class lr_step_stop_tr_scheduler():
    def __init__(self, times=1, factor= 1.0/10.0, epsilon=0.0001, patience=3):
        self.times = times
        self.factor = factor
        self.epsilon = epsilon
        self.train_error = 99999999
        self.patience = patience
        self.patience_cont = 0

    def update(self, lr, train_error):
        return lr

    def epoch_end(self, lr, train_error):
        if self.times > 0:
            diff = self.train_error - train_error
            self.train_error = min(train_error, self.train_error)

            if diff <= self.epsilon:
                if self.patience_cont < self.patience:
                    self.patience_cont += 1
                else:
                    self.patience_cont = 0

                    return lr*self.factor

        return lr
