# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import argparse
import logging
import sys

import numpy as np
import torch
import tqdm
from torch.backends import cudnn
from sklearn.manifold import TSNE
import os
import matplotlib.pyplot as plt

sys.path.append('.')

from fastreid.evaluation.rank import evaluate_rank
from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.data import build_reid_test_loader, build_reid_train_loader
from predictor import FeatureExtractionDemo
from fastreid.utils.visualizer import Visualizer

# import some modules added in project
# for example, add partial reid like this below
# sys.path.append("projects/PartialReID")
# from partialreid import *

cudnn.benchmark = True
setup_logger(name="fastreid")

logger = logging.getLogger('fastreid.visualize_result')


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='if use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--dataset-name",
        help="a test dataset name for visualizing ranking list."
    )
    parser.add_argument(
        "--output",
        default="./vis_rank_list",
        help="a file or directory to save rankling list result.",

    )
    parser.add_argument(
        "--vis-label",
        action='store_true',
        help="if visualize label of query instance"
    )
    parser.add_argument(
        "--num-vis",
        default=100,
        help="number of query images to be visualized",
    )
    parser.add_argument(
        "--rank-sort",
        default="ascending",
        help="rank order of visualization images by AP metric",
    )
    parser.add_argument(
        "--label-sort",
        default="ascending",
        help="label order of visualization images by cosine similarity metric",
    )
    parser.add_argument(
        "--max-rank",
        default=10,
        help="maximum number of rank list to be visualized",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    # test_loader, num_query = build_reid_test_loader(cfg, dataset_name=args.dataset_name)
    train_loader = build_reid_train_loader(cfg, dataset_name=args.dataset_name)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)

    logger.info("Start extracting image features")
    feats = []
    pids = []
    camids = []
    viewids = []

    for i, (feat, pid, camid, viewid) in enumerate(tqdm.tqdm(demo.run_on_loader(train_loader), total=100)):
        if i> 4: break
        feats.append(feat)
        pids.extend(pid)
        camids.extend(camid)
        viewids.extend(viewid)
    view1_index = [index for index, content in enumerate(viewids) if content == 'Aerial']
    view2_index = [index for index, content in enumerate(viewids) if content == 'Ground']
    # print(viewid[0])

    feats = torch.cat(feats, dim=0)
    tsne = TSNE(n_components=2, learning_rate=200, metric='cosine', perplexity=100, random_state=1)
    # tsne = TSNE(n_components=2, learning_rate=200, metric='cosine',init="pca", n_jobs=-1)
    tsne = tsne.fit_transform(feats)
    
    colors={0:'b', 1:'c', 2:'y', 3:'m', 4:'r', 5:'g', 6:'k', 7:'yellow', 8:'yellowgreen', 9:'wheat'}
    pid_container = set()
    for pid in sorted(pids):
        pid_container.add(pid)
    
    pid2label = {pid: label for label, pid in enumerate(pid_container)}
    pids = [colors[pid2label[p]] for p in pids]
    vis_x = tsne[:, 0]#0维
    vis_y = tsne[:, 1]#1维
    pids = np.array(pids)
    # print(pids[view1_index])
    plt.scatter(vis_x[view1_index], vis_y[view1_index], c=pids[view1_index], marker='o', alpha=0.5, label='Aerial')
    plt.scatter(vis_x[view2_index], vis_y[view2_index], c=pids[view2_index], marker='+', label='Ground')
    # plt.scatter(vis_x[view2_index], vis_y[view2_index], c='w', marker='<', linewidths=1)
    plt.legend(loc='upper left')
    
    os.makedirs(args.output, exist_ok=True)
    filepath = os.path.join(args.output, "{}".format('tsne'))
    try:
        plt.savefig(filepath)
    except:
        print(filepath)
    # q_feat = feats[:num_query]
    # g_feat = feats[num_query:]
    # q_pids = np.asarray(pids[:num_query])
    # g_pids = np.asarray(pids[num_query:])
    # q_camids = np.asarray(camids[:num_query])
    # g_camids = np.asarray(camids[num_query:])

    # # compute cosine distance
    # distmat = 1 - torch.mm(q_feat, g_feat.t())
    # distmat = distmat.numpy()

    # logger.info("Computing APs for all query images ...")
    # cmc, all_ap, all_inp = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids)
    # logger.info("Finish computing APs for all query images!")

    # visualizer = Visualizer(test_loader.dataset)
    # visualizer.get_model_output(all_ap, distmat, q_pids, g_pids, q_camids, g_camids)

    # logger.info("Start saving ROC curve ...")
    # fpr, tpr, pos, neg = visualizer.vis_roc_curve(args.output)
    # visualizer.save_roc_info(args.output, fpr, tpr, pos, neg)
    # logger.info("Finish saving ROC curve!")

    # logger.info("Saving rank list result ...")
    # # query_indices = visualizer.vis_rank_list(args.output, args.vis_label, args.num_vis,
    # #                                          args.rank_sort, args.label_sort, args.max_rank)
    # logger.info("Finish saving rank list results!")
