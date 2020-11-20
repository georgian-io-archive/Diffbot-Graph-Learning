from collections import defaultdict
import datetime
import logging
from os import makedirs
from os.path import dirname, abspath, join, exists
import re

import numpy as np
import pytz
import torch

logger = logging.getLogger(__name__)


def get_root_path():
    return dirname(abspath(__file__))


def get_logs_path():
    return join(get_root_path(), 'logs')


def get_data_path():
    return join(get_root_path(), 'data')


def get_temp_path():
    return join(get_root_path(), 'temp')


def create_dir_if_not_exists(folder):
    if not exists(folder):
        makedirs(folder)


def get_current_ts(zone='US/Pacific'):
    return datetime.datetime.now(pytz.timezone(zone)).strftime(
        '%Y-%m-%dT%H-%M-%S.%f')


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, want_increase, metric_name=None, verbose=False, delta=1e-7, save_path='checkpoint.pt'):
        """
        Args:
            want_increase(bool): Whether we cant the validation metric to increase
            patience (int): How long to wait after last time validation metric improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation metric improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.want_increase = want_increase
        self.metric_name = metric_name if metric_name is not None else "metric"
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_val_metric = np.Inf if not want_increase else 0
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_metric, model, ):

        if not self.want_increase:
            score = -val_metric
        else:
            score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            logger.debug(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            str_inc_or_dec = 'increased' if self.want_increase else 'decreased'
            logger.debug(f'Validation {self.metric_name} {str_inc_or_dec} ({self.best_val_metric:.6f} --> {val_metric:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.best_val_metric = val_metric


def get_args_info_as_str(config_flags):
    rtn = []
    if type(config_flags) is not dict:
        d = vars(config_flags)
    else:
        d = config_flags
    for k in sorted_nicely(d.keys()):
        v = d[k]
        if type(v) is dict:
            for k2, v2 in v.items():
                s = '{0:26} : {1}'.format(k + '_' + k2, v2)
                rtn.append(s)
        else:
            s = '{0:26} : {1}'.format(k, v)
            rtn.append(s)
    return '\n'.join(rtn)


def sorted_nicely(l, reverse=False):
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        if type(s) is not str:
            raise ValueError('{} must be a string in l: {}'.format(s, l))
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    rtn = sorted(l, key=alphanum_key)
    if reverse:
        rtn = reversed(rtn)
    return rtn


def underscore_to_camelcase(word):
    return ''.join(x.capitalize() or '_' for x in word.split('_'))


def string_to_underscore(sentence):
    return '_'.join(x.lower() or '_' for x in ' '.join(sentence.split()).split(' '))


def get_graph_diffbot_category_count(nx_graph, min_count=200):

    all_diffbot_cats = defaultdict(int)
    for i, ndata in nx_graph.nodes(data=True):
        if 'diffbot_categories' in ndata:
            for cat in ndata['diffbot_categories'].split(','):
                all_diffbot_cats[cat.strip()] += 1
    all_diffbot_cats = {k: v for k, v in all_diffbot_cats.items() if v > min_count}
    return all_diffbot_cats


if __name__ == '__main__':
    print(string_to_underscore('DsDSFafasd sdfasdf sdfasd    dsaf'))