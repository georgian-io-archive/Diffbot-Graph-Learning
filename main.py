from collections import defaultdict, Counter
import json
import logging
import os
from statistics import mean, stdev
import sys
import time

import numpy as np
from os.path import join
import pandas as pd
from pprint import pformat
from scipy.special import softmax
from sklearn.model_selection import KFold, train_test_split
import torch
import torch.nn.functional as F
from transformers import HfArgumentParser

from graph_dataset import DiffbotGraphDatasetNodeLabels
from util import EarlyStopping, create_dir_if_not_exists, get_args_info_as_str
from evaluation import calc_classification_metrics
from experiment_args import ExperimentArguments, TrainingArguments
from trainer import Trainer

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ExperimentArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        exp_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        exp_args, training_args = parser.parse_args_into_dataclasses()

    create_dir_if_not_exists(exp_args.log_dir)
    stream_handler = logging.StreamHandler(sys.stderr)
    file_handler = logging.FileHandler(filename=os.path.join(exp_args.log_dir, 'train_log.txt'),
                                       mode='w+')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG if exp_args.debug else logging.INFO,
        handlers=[stream_handler, file_handler]
    )

    logger.info(f"======== Experiment Args ========\n{get_args_info_as_str(exp_args)}\n")
    logger.info(f"======== Training Args ========\n{get_args_info_as_str(training_args)}\n")

    dataset = DiffbotGraphDatasetNodeLabels(exp_args.dataset_name,
                                            exp_args.market,
                                            exp_args.market_source,
                                            exp_args.natts_path,
                                            metapaths=exp_args.metapaths)
    dataset._process()
    natts_path = exp_args.natts_path if exp_args.natts_path else dataset.natts_path
    with open(natts_path, 'rb') as f:
        natts_dict = json.load(f)

    device = torch.device(f"cuda:{training_args.device}" if torch.cuda.is_available() and training_args.device >= 0 else "cpu")

    logger.info(f'======== Dataset Feature Info ========\n{pformat(natts_dict)}\n')
    logger.info(f'\n Node Feat Dimensions: {pformat({name: feat.shape for name, feat in dataset.graph.ndata["feats"].items()})}')

    logger.info(f'Number of nodes {dataset.homo_graph.number_of_nodes()}\n')
    logger.info(f'Number of edges {dataset.homo_graph.number_of_edges()}\n')
    logger.info(f'Canonical Edge Types: {pformat(dataset.graph.canonical_etypes)}\n')
    logger.info(f'Node Types: {pformat(dataset.graph.ntypes)}\n')
    if dataset.metapaths:
        logger.info(f'Metapaths: {pformat(dataset.metapaths)}\n')

    labels = torch.LongTensor(dataset.labels).to(device)
    if training_args.repeat == 1:
        train_inds, val_inds, test_inds = [dataset.train_idx], [dataset.val_idx], [dataset.test_idx]
    else:
        kf = KFold(n_splits=training_args.repeat, shuffle=True)
        splits = list(kf.split(dataset.labels))
        train_inds, val_inds, test_inds = [], [], []
        for train_idxs, test_idxs in splits:
            test_inds.append(test_idxs)
            train_idxs, val_idxs = train_test_split(train_idxs, test_size=0.2)
            train_inds.append(train_idxs), val_inds.append(val_idxs)
    logger.info(f"Labels Count {Counter(dataset.labels)}")
    logger.info(f"Num train {len(train_inds[0])}")
    logger.info(f"Num val {len(val_inds[0])}")
    logger.info(f"Num test {len(test_inds[0])}")

    test_metrics_all = defaultdict(list)
    for run_num in range(training_args.repeat):
        train_idx, val_idx, test_idx = train_inds[run_num], val_inds[run_num], test_inds[run_num]
        logger.info(f"======== Run {run_num} ========")
        trainer = Trainer(dataset, exp_args.model, training_args, device)
        if run_num == 0:
            logger.info(f"======== MODEL ========\n{trainer.model}")
        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=training_args.lr,
                                     weight_decay=training_args.weight_decay)
        trainer.train()
        early_stopping = EarlyStopping(patience=training_args.patience,
                                       want_increase=training_args.val_metric != 'loss',
                                       metric_name=training_args.val_metric,
                                       verbose=True,
                                       save_path=join(exp_args.log_dir, 'checkpoint.pt'))
        dur1 = []
        dur2 = []
        dur3 = []
        for epoch in range(training_args.num_epochs):
            t0 = time.time()
            # training forward
            trainer.train()
            logits, embeddings = trainer.forward()

            train_loss = F.cross_entropy(logits[train_idx], labels[train_idx])
            t1 = time.time()
            dur1.append(t1 - t0)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            t2 = time.time()
            dur2.append(t2 - t1)

            trainer.eval()

            with torch.no_grad():
                logits, embeddings = trainer.forward()
                val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
                logits = logits.cpu().detach().numpy()
                embeddings = embeddings.cpu().detach().numpy()
                pred_scores = softmax(logits[val_idx], axis=1)[:, 1]
                preds_labels = np.argmax(logits[val_idx], axis=1)
                val_metrics = calc_classification_metrics(pred_scores, preds_labels, labels[val_idx].cpu().detach().numpy())
                val_metrics['loss'] = val_loss.item()

            t3 = time.time()
            dur3.append(t3 - t2)

            if epoch % training_args.print_val_epochs == 0:
                logging.info('\nValidation metrics\n' + pformat(val_metrics, indent=3))
                logging.info(
                    "Epoch {:05d} | Train_Loss {:.4f} | Val_Loss {:.4f} | TimeFoward(s) {:.4f} | TimeBackward(s) {:.4f} | TimeValidation(s) {:.4f}".format(
                        epoch, train_loss.item(), val_loss.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))
            else:
                logging.debug('\nValidation metrics\n' + pformat(val_metrics, indent=3))
                logging.debug(
                    "Epoch {:05d} | Train_Loss {:.4f} | Val_Loss {:.4f} | TimeFoward(s) {:.4f} | TimeBackward(s) {:.4f} | TimeValidation(s) {:.4f}".format(
                        epoch, train_loss.item(), val_loss.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))



            early_stopping(val_metrics[training_args.val_metric], trainer.model)
            if early_stopping.early_stop:
                logging.info('Early stopping!')
                break

        trainer.model.load_state_dict(torch.load(join(exp_args.log_dir, 'checkpoint.pt')))
        trainer.eval()
        with torch.no_grad():
            logits, _ = trainer.forward()
            logits = logits.cpu().detach().numpy()
            pred_scores = softmax(logits[test_idx], axis=1)[:, 1]
            preds_labels = np.argmax(logits[test_idx], axis=1)
            test_metrics = calc_classification_metrics(pred_scores, preds_labels, labels[test_idx].cpu().detach().numpy())
            for k, metric in test_metrics.items():
                test_metrics_all[k].append(metric)
        for ind, pred_score, pred_label in zip(test_idx, pred_scores, preds_labels):
            dataset.labels_info[ind]['pred_score'] = pred_score
            dataset.labels_info[ind]['pred_label'] = pred_label

        test_res_df = pd.DataFrame([label_info for i, label_info in enumerate(dataset.labels_info) if i in test_idx])
        test_res_df.to_csv(join(exp_args.log_dir, f"test_res_run{run_num}.csv"))

    logging.info('\nTesting\n' + pformat(dict(test_metrics_all), indent=3))
    avg_results = {}
    for k, v in dict(test_metrics_all).items():
        avg_results[k + '_mean'] = mean(v)
        avg_results[k + '_std'] = stdev(v)
    logger.info(f'{pformat(avg_results, indent=3)}')
    with open(join(exp_args.log_dir, 'test_results.json'), 'w') as f:
        json.dump(dict(test_metrics_all), f, indent=4)
    with open(join(exp_args.log_dir, 'test_results_avg.json'), 'w') as f:
        json.dump(avg_results, f, indent=4)
    logger.info(f'\nResults saved to {exp_args.log_dir}')


if __name__ == '__main__':
    main()