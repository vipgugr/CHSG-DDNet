import abc
import pickle

class Event_Log(abc.ABC):
    def __init__(self, init_data):
        self.info = init_data

    @abc.abstractmethod
    def get_reserved_keys(self):
        pass

    def log_info(self, keys_and_values):
        reserved = self.get_reserved_keys()

        for key, value in keys_and_values:
            if not (key in reserved):
                self.info[key] = value

    def __getitem__(self, key):
        value = self.info[key]

        return value

    def save(self, path):
        with open(path, 'wb') as handle:
            pickle.dump(self.info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, 'rb') as handle:
            self.info = pickle.load(handle)


class Epoch_Log(Event_Log):
    RESERVED_KEYS = ['epoch', 'iters_info']
    def __init__(self, epoch):
        super(Epoch_Log, self).__init__({Epoch_Log.RESERVED_KEYS[0]: epoch,
                                         Epoch_Log.RESERVED_KEYS[1]: []})

    def get_reserved_keys(self):
        return Epoch_Log.RESERVED_KEYS

    def log_iter_info(self, iter, dict):
        new_dict = dict.copy()
        new_dict['iter'] = iter
        self.info[Epoch_Log.RESERVED_KEYS[1]].append(new_dict)


class History_Log(Event_Log):
    RESERVED_KEYS = ['pid', 'info_epochs']

    def __init__(self, pid):
        super(History_Log, self).__init__({History_Log.RESERVED_KEYS[0]: pid,
                                           History_Log.RESERVED_KEYS[1]: []})
        self.eval_out = None

    def get_reserved_keys(self):
        return History_Log.RESERVED_KEYS

    def log_info_epoch(self, epoch_info):
        if isinstance(epoch_info, Epoch_Log):
            self.info['info_epochs'].append(epoch_info)
