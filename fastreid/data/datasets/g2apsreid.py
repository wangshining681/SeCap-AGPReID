# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from . import DATASET_REGISTRY


@DATASET_REGISTRY.register()

class G2APS_ReID(ImageDataset):

    dataset_dir = ''
    dataset_name = "g2aps_reid"

    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'GA-ReID')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"AG-ReID".')

        
        
        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        # self.setting_text = osp.join(self.data_dir, 'exp1_A2G.txt')
        # self.setting_text = osp.join(self.data_dir, 'exp2_G2A.txt')
        self.setting_text = osp.join(self.data_dir, 'exp3_A2G+.txt')
        # self.setting_text = osp.join(self.data_dir, 'exp4_G2A+.txt')
        # self.setting_text = osp.join(self.data_dir, 'exp5_G2A+G.txt')

        required_files = [
            self.data_dir,
            self.train_dir,
            self.setting_text
        ]

        self.check_before_run(required_files)

        train = lambda: self.process_dir(self.train_dir)
        query, gallery = self.process_setting_txt(self.data_dir,self.setting_text)

        super(G2APS_ReID, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '**/*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)v(\d)')

        data = []
        for img_path in img_paths:
            # print(img_path)
            pid, camid, _ = map(int, pattern.search(img_path).groups())
            if camid: 
                camid = 1
                viewid = 'Ground'
            else:
                viewid = 'Aerial'
            if pid == -1:
                continue  # junk images are just ignored

            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid, viewid))

        return data
    
    def process_setting_txt(self, path, text_path, is_train=True):
        pattern = re.compile(r'([-\d]+)_c(\d)v(\d)')
        with open(text_path,'r') as f:
            query = []
            gallery = []
            for img_path in f:
                # print(img_path)
                split_p = img_path.split('/')[0]
                split = split_p.split('_')[0]
                img_path = osp.join(path, img_path[:-1])
                if split == 'query':
                    pid, camid, _  = map(int, pattern.search(img_path).groups())
                    if camid: 
                        camid = 1
                        viewid = 'Ground'
                    else:
                        viewid = 'Aerial'
                    # if pid == -1:
                    #     continue  # junk images are just ignored
                                       
                    query.append((img_path, pid, camid, viewid))
                else:
                    pid, camid, _ = map(int, pattern.search(img_path).groups())
                    if camid: 
                        camid = 1
                        viewid = 'Ground'
                    else:
                        viewid = 'Aerial'
                    # if pid == -1:
                    #     continue  # junk images are just ignored
                                     
                    gallery.append((img_path, pid, camid, viewid))

        return query, gallery
    
@DATASET_REGISTRY.register()
class G2APS_ReID_A2G(ImageDataset):

    dataset_dir = ''
    dataset_name = "g2aps_reid"

    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'GA-ReID')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"AG-ReID".')

        
        
        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        # self.setting_text = osp.join(self.data_dir, 'exp1_A2G.txt')
        # self.setting_text = osp.join(self.data_dir, 'exp2_G2A.txt')
        self.setting_text = osp.join(self.data_dir, 'exp3_A2G+.txt')
        # self.setting_text = osp.join(self.data_dir, 'exp4_G2A+.txt')
        # self.setting_text = osp.join(self.data_dir, 'exp5_G2A+G.txt')

        required_files = [
            self.data_dir,
            self.train_dir,
            self.setting_text
        ]

        self.check_before_run(required_files)

        train = lambda: self.process_dir(self.train_dir)
        query, gallery = self.process_setting_txt(self.data_dir,self.setting_text)

        super(G2APS_ReID_A2G, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '**/*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)v(\d)')

        data = []
        for img_path in img_paths:
            # print(img_path)
            pid, camid, _ = map(int, pattern.search(img_path).groups())
            if camid: 
                camid = 1
                viewid = 'Ground'
            else:
                viewid = 'Aerial'
            if pid == -1:
                continue  # junk images are just ignored

            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid, viewid))

        return data
    
    def process_setting_txt(self, path, text_path, is_train=True):
        pattern = re.compile(r'([-\d]+)_c(\d)v(\d)')
        with open(text_path,'r') as f:
            query = []
            gallery = []
            for img_path in f:
                # print(img_path)
                split_p = img_path.split('/')[0]
                split = split_p.split('_')[0]
                img_path = osp.join(path, img_path[:-1])
                if split == 'query':
                    pid, camid, _  = map(int, pattern.search(img_path).groups())
                    if camid: 
                        camid = 1
                        viewid = 'Ground'
                    else:
                        viewid = 'Aerial'
                    # if pid == -1:
                    #     continue  # junk images are just ignored
                                       
                    query.append((img_path, pid, camid, viewid))
                else:
                    pid, camid, _ = map(int, pattern.search(img_path).groups())
                    if camid: 
                        camid = 1
                        viewid = 'Ground'
                    else:
                        viewid = 'Aerial'
                    # if pid == -1:
                    #     continue  # junk images are just ignored
                                     
                    gallery.append((img_path, pid, camid, viewid))

        return query, gallery

@DATASET_REGISTRY.register()
class G2APS_ReID_G2A(ImageDataset):

    dataset_dir = ''
    dataset_name = "g2aps_reid"

    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'GA-ReID')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"AG-ReID".')

        
        
        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        # self.setting_text = osp.join(self.data_dir, 'exp1_A2G.txt')
        # self.setting_text = osp.join(self.data_dir, 'exp2_G2A.txt')
        # self.setting_text = osp.join(self.data_dir, 'exp3_A2G+.txt')
        self.setting_text = osp.join(self.data_dir, 'exp4_G2A+.txt')
        # self.setting_text = osp.join(self.data_dir, 'exp5_G2A+G.txt')

        required_files = [
            self.data_dir,
            self.train_dir,
            self.setting_text
        ]

        self.check_before_run(required_files)

        train = lambda: self.process_dir(self.train_dir)
        query, gallery = self.process_setting_txt(self.data_dir,self.setting_text)

        super(G2APS_ReID_G2A, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '**/*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)v(\d)')

        data = []
        for img_path in img_paths:
            # print(img_path)
            pid, camid, _ = map(int, pattern.search(img_path).groups())
            if camid: 
                camid = 1
                viewid = 'Ground'
            else:
                viewid = 'Aerial'
            if pid == -1:
                continue  # junk images are just ignored

            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid, viewid))

        return data
    
    def process_setting_txt(self, path, text_path, is_train=True):
        pattern = re.compile(r'([-\d]+)_c(\d)v(\d)')
        with open(text_path,'r') as f:
            query = []
            gallery = []
            for img_path in f:
                # print(img_path)
                split_p = img_path.split('/')[0]
                split = split_p.split('_')[0]
                img_path = osp.join(path, img_path[:-1])
                if split == 'query':
                    pid, camid, _  = map(int, pattern.search(img_path).groups())
                    if camid: 
                        camid = 1
                        viewid = 'Ground'
                    else:
                        viewid = 'Aerial'
                    # if pid == -1:
                    #     continue  # junk images are just ignored
                                       
                    query.append((img_path, pid, camid, viewid))
                else:
                    pid, camid, _ = map(int, pattern.search(img_path).groups())
                    if camid: 
                        camid = 1
                        viewid = 'Ground'
                    else:
                        viewid = 'Aerial'
                    # if pid == -1:
                    #     continue  # junk images are just ignored
                                     
                    gallery.append((img_path, pid, camid, viewid))

        return query, gallery
    

