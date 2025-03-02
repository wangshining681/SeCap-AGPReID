import glob
import re
import mat4py
import pandas as pd
import torch
import re
import warnings
import os.path as osp

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class AG_ReID_v2(ImageDataset):

    def __init__(self, root='datasets',
                 verbose=True, **kwargs):
        
        self.dataset_dir = root
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'AG-ReID.v2')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"AG-ReID".')
        self.train_dir = osp.join(self.data_dir, 'AG-ReID.v2', 'train_all')

        self.query_dir = osp.join(self.data_dir, 'AG-ReID.v2', 'query')
        self.gallery_dir = osp.join(self.data_dir, 'AG-ReID.v2', 'gallery')
        self.setting_text = osp.join(self.data_dir, 'exp1_aerial_to_cctv.txt')
        self.qut_attribute_path = osp.join(self.data_dir, 'qut_attribute_v8.mat')
        self.attribute_dict_all = self.generate_attribute_dict(self.qut_attribute_path, "qut_attribute")
        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
            self.qut_attribute_path
        ]
        self.check_before_run(required_files)

        train = self._process_dir(self.train_dir, is_train=True)
        query, gallery = self.process_setting_txt(osp.join(self.data_dir, 'AG-ReID.v2'), self.setting_text)
        # query = self._process_dir(self.query_dir, is_train=False)
        # gallery = self._process_dir(self.gallery_dir, is_train=False)

        # if verbose:
        #     print("=> AG-ReID loaded")

        #     self.print_dataset_statistics(train, query, gallery)

        # self.train = train

        # self.query = query
        # self.gallery = gallery
        # self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)

        # self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        # self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
        super(AG_ReID_v2, self).__init__(train, query, gallery, **kwargs)
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))

        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '**/*.jpg'), recursive=True)
        pattern_pid = re.compile(r'P([-\d]+)T([-\d]+)A([-\d]+)')
        pattern_camid = re.compile(r'C([-\d]+)F([-\d]+)')

        data = []

        for img_path in img_paths:
            fname = osp.split(img_path)[-1]

            pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
            pid = int(pid_part1 + pid_part2 + pid_part3)

            camid, frameid = pattern_camid.search(fname).groups()
            camid = int(camid)
            if camid: 
                camid -= 1
                viewid = 'Ground'
            else:
                viewid = 'Aerial'
            p_attribute = self.attribute_dict_all[str(pid)]
            p_attribute = p_attribute.float()

            if is_train:
                pid = "ag-reid" + "_" + str(pid)
            data.append((img_path, pid, camid, viewid, p_attribute))

        return data

    def generate_attribute_dict(self, dir_path: str, dataset: str):

        mat_attribute_train = mat4py.loadmat(dir_path)[dataset]["train"]
        mat_attribute_train = pd.DataFrame(mat_attribute_train, index=mat_attribute_train['image_index']).astype(int)

        mat_attribute_test = mat4py.loadmat(dir_path)[dataset]["test"]
        mat_attribute_test = pd.DataFrame(mat_attribute_test, index=mat_attribute_test['image_index']).astype(int)

        mat_attribute = mat_attribute_train.add(mat_attribute_test, fill_value=0)
        mat_attribute = mat_attribute.drop(['image_index'], axis=1)

        self.key_attribute = list(mat_attribute.keys())

        h, w = mat_attribute.shape
        dict_attribute = dict()

        for i in range(h):
            row = mat_attribute.iloc[i:i + 1, :].values.reshape(-1)
            dict_attribute[str(int(mat_attribute.index[i]))] = torch.tensor(row[0:].astype(int)) * 2 - 3

        return dict_attribute

    def process_setting_txt(self, path, text_path, is_train=True):
        pattern_pid = re.compile(r'P([-\d]+)T([-\d]+)A([-\d]+)')
        pattern_camid = re.compile(r'C([-\d]+)F([-\d]+)')
        with open(text_path,'r') as f:
            query = []
            gallery = []
            for img_path in f:
                # print(img_path)
                split = img_path.split('/')[0]
                # split = split_p.split('_')[0]
                img_path = osp.join(path, img_path[:-1])
                if split == 'query':
                    pid_part1, pid_part2, pid_part3 = pattern_pid.search(img_path).groups()
                    pid = int(pid_part1 + pid_part2 + pid_part3)

                    camid, frameid = pattern_camid.search(img_path).groups()
                    camid = int(camid)
                    p_attribute = self.attribute_dict_all[str(pid)]
                    p_attribute = p_attribute.float()
                    if camid: 
                        camid -= 1
                        viewid = 'Ground'
                    else:
                        viewid = 'Aerial'
                    # if pid == -1:
                    #     continue  # junk images are just ignored
                                       
                    query.append((img_path, pid, camid, viewid))
                else:
                    pid_part1, pid_part2, pid_part3 = pattern_pid.search(img_path).groups()
                    pid = int(pid_part1 + pid_part2 + pid_part3)

                    camid, frameid = pattern_camid.search(img_path).groups()
                    camid = int(camid)
                    p_attribute = self.attribute_dict_all[str(pid)]
                    p_attribute = p_attribute.float()
                    if camid: 
                        camid = 1
                        viewid = 'Ground'
                    else:
                        viewid = 'Aerial'
                    # if pid == -1:
                    #     continue  # junk images are just ignored
                                     
                    gallery.append((img_path, pid, camid, viewid))

        return query, gallery

    def name_of_attribute(self):
        if self.key_attribute:
            print(self.key_attribute)
            return self.key_attribute
        else:
            assert False
            
@DATASET_REGISTRY.register()
class AG_ReID_v2_G2A(ImageDataset):

    def __init__(self, root='datasets',
                 verbose=True, **kwargs):
        
        self.dataset_dir = root
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'AG-ReID.v2')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"AG-ReID".')
        self.train_dir = osp.join(self.data_dir, 'AG-ReID.v2', 'train_all')

        self.query_dir = osp.join(self.data_dir, 'AG-ReID.v2', 'query')
        self.gallery_dir = osp.join(self.data_dir, 'AG-ReID.v2', 'gallery')
        self.setting_text = osp.join(self.data_dir, 'exp4_cctv_to_aerial.txt')
        self.qut_attribute_path = osp.join(self.data_dir, 'qut_attribute_v8.mat')
        self.attribute_dict_all = self.generate_attribute_dict(self.qut_attribute_path, "qut_attribute")
        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
            self.qut_attribute_path
        ]
        self.check_before_run(required_files)

        train = self._process_dir(self.train_dir, is_train=True)
        query, gallery = self.process_setting_txt(osp.join(self.data_dir, 'AG-ReID.v2'),self.setting_text)
        # query = self._process_dir(self.query_dir, is_train=False)
        # gallery = self._process_dir(self.gallery_dir, is_train=False)

        # if verbose:
        #     print("=> AG-ReID loaded")

        #     self.print_dataset_statistics(train, query, gallery)

        # self.train = train

        # self.query = query
        # self.gallery = gallery
        # self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)

        # self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        # self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
        super(AG_ReID_v2_G2A, self).__init__(train, query, gallery, **kwargs)
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))

        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '**/*.jpg'), recursive=True)
        pattern_pid = re.compile(r'P([-\d]+)T([-\d]+)A([-\d]+)')
        pattern_camid = re.compile(r'C([-\d]+)F([-\d]+)')

        data = []

        for img_path in img_paths:
            fname = osp.split(img_path)[-1]

            pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
            pid = int(pid_part1 + pid_part2 + pid_part3)

            camid, frameid = pattern_camid.search(fname).groups()
            camid = int(camid)
            if camid: 
                camid -= 1
                viewid = 'Ground'
            else:
                viewid = 'Aerial'
            p_attribute = self.attribute_dict_all[str(pid)]
            p_attribute = p_attribute.float()

            if is_train:
                pid = "ag-reid" + "_" + str(pid)
            data.append((img_path, pid, camid, viewid, p_attribute))

        return data

    def generate_attribute_dict(self, dir_path: str, dataset: str):

        mat_attribute_train = mat4py.loadmat(dir_path)[dataset]["train"]
        mat_attribute_train = pd.DataFrame(mat_attribute_train, index=mat_attribute_train['image_index']).astype(int)

        mat_attribute_test = mat4py.loadmat(dir_path)[dataset]["test"]
        mat_attribute_test = pd.DataFrame(mat_attribute_test, index=mat_attribute_test['image_index']).astype(int)

        mat_attribute = mat_attribute_train.add(mat_attribute_test, fill_value=0)
        mat_attribute = mat_attribute.drop(['image_index'], axis=1)

        self.key_attribute = list(mat_attribute.keys())

        h, w = mat_attribute.shape
        dict_attribute = dict()

        for i in range(h):
            row = mat_attribute.iloc[i:i + 1, :].values.reshape(-1)
            dict_attribute[str(int(mat_attribute.index[i]))] = torch.tensor(row[0:].astype(int)) * 2 - 3

        return dict_attribute

    def process_setting_txt(self, path, text_path, is_train=True):
        pattern_pid = re.compile(r'P([-\d]+)T([-\d]+)A([-\d]+)')
        pattern_camid = re.compile(r'C([-\d]+)F([-\d]+)')
        with open(text_path,'r') as f:
            query = []
            gallery = []
            for img_path in f:
                # print(img_path)
                split = img_path.split('/')[0]
                # split = split_p.split('_')[0]
                img_path = osp.join(path, img_path[:-1])
                if split == 'query':
                    pid_part1, pid_part2, pid_part3 = pattern_pid.search(img_path).groups()
                    pid = int(pid_part1 + pid_part2 + pid_part3)

                    camid, frameid = pattern_camid.search(img_path).groups()
                    camid = int(camid)
                    p_attribute = self.attribute_dict_all[str(pid)]
                    p_attribute = p_attribute.float()
                    if camid: 
                        camid -= 1
                        viewid = 'Ground'
                    else:
                        viewid = 'Aerial'
                    # if pid == -1:
                    #     continue  # junk images are just ignored
                                       
                    query.append((img_path, pid, camid, viewid))
                else:
                    pid_part1, pid_part2, pid_part3 = pattern_pid.search(img_path).groups()
                    pid = int(pid_part1 + pid_part2 + pid_part3)

                    camid, frameid = pattern_camid.search(img_path).groups()
                    camid = int(camid)
                    p_attribute = self.attribute_dict_all[str(pid)]
                    p_attribute = p_attribute.float()
                    if camid: 
                        camid = 1
                        viewid = 'Ground'
                    else:
                        viewid = 'Aerial'
                    # if pid == -1:
                    #     continue  # junk images are just ignored
                                     
                    gallery.append((img_path, pid, camid, viewid))

        return query, gallery

    def name_of_attribute(self):
        if self.key_attribute:
            print(self.key_attribute)
            return self.key_attribute
        else:
            assert False

@DATASET_REGISTRY.register()
class AG_ReID_v2_A2W(ImageDataset):

    def __init__(self, root='datasets',
                 verbose=True, **kwargs):
        
        self.dataset_dir = root
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'AG-ReID.v2')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"AG-ReID".')
        self.train_dir = osp.join(self.data_dir, 'AG-ReID.v2', 'train_all')

        self.query_dir = osp.join(self.data_dir, 'AG-ReID.v2', 'query')
        self.gallery_dir = osp.join(self.data_dir, 'AG-ReID.v2', 'gallery')
        self.setting_text = osp.join(self.data_dir, 'exp2_aerial_to_wearable.txt')
        self.qut_attribute_path = osp.join(self.data_dir, 'qut_attribute_v8.mat')
        self.attribute_dict_all = self.generate_attribute_dict(self.qut_attribute_path, "qut_attribute")
        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
            self.qut_attribute_path
        ]
        self.check_before_run(required_files)

        train = self._process_dir(self.train_dir, is_train=True)
        query, gallery = self.process_setting_txt(osp.join(self.data_dir, 'AG-ReID.v2'),self.setting_text)
        # query = self._process_dir(self.query_dir, is_train=False)
        # gallery = self._process_dir(self.gallery_dir, is_train=False)

        # if verbose:
        #     print("=> AG-ReID loaded")

        #     self.print_dataset_statistics(train, query, gallery)

        # self.train = train

        # self.query = query
        # self.gallery = gallery
        # self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)

        # self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        # self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
        super(AG_ReID_v2_A2W, self).__init__(train, query, gallery, **kwargs)
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))

        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '**/*.jpg'), recursive=True)
        pattern_pid = re.compile(r'P([-\d]+)T([-\d]+)A([-\d]+)')
        pattern_camid = re.compile(r'C([-\d]+)F([-\d]+)')

        data = []

        for img_path in img_paths:
            fname = osp.split(img_path)[-1]

            pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
            pid = int(pid_part1 + pid_part2 + pid_part3)

            camid, frameid = pattern_camid.search(fname).groups()
            camid = int(camid)
            if camid: 
                camid -= 1
                viewid = 'Ground'
            else:
                viewid = 'Aerial'
            p_attribute = self.attribute_dict_all[str(pid)]
            p_attribute = p_attribute.float()

            if is_train:
                pid = "ag-reid" + "_" + str(pid)
            data.append((img_path, pid, camid, viewid, p_attribute))

        return data

    def generate_attribute_dict(self, dir_path: str, dataset: str):

        mat_attribute_train = mat4py.loadmat(dir_path)[dataset]["train"]
        mat_attribute_train = pd.DataFrame(mat_attribute_train, index=mat_attribute_train['image_index']).astype(int)

        mat_attribute_test = mat4py.loadmat(dir_path)[dataset]["test"]
        mat_attribute_test = pd.DataFrame(mat_attribute_test, index=mat_attribute_test['image_index']).astype(int)

        mat_attribute = mat_attribute_train.add(mat_attribute_test, fill_value=0)
        mat_attribute = mat_attribute.drop(['image_index'], axis=1)

        self.key_attribute = list(mat_attribute.keys())

        h, w = mat_attribute.shape
        dict_attribute = dict()

        for i in range(h):
            row = mat_attribute.iloc[i:i + 1, :].values.reshape(-1)
            dict_attribute[str(int(mat_attribute.index[i]))] = torch.tensor(row[0:].astype(int)) * 2 - 3

        return dict_attribute

    def process_setting_txt(self, path, text_path, is_train=True):
        pattern_pid = re.compile(r'P([-\d]+)T([-\d]+)A([-\d]+)')
        pattern_camid = re.compile(r'C([-\d]+)F([-\d]+)')
        with open(text_path,'r') as f:
            query = []
            gallery = []
            for img_path in f:
                # print(img_path)
                split = img_path.split('/')[0]
                # split = split_p.split('_')[0]
                img_path = osp.join(path, img_path[:-1])
                if split == 'query':
                    pid_part1, pid_part2, pid_part3 = pattern_pid.search(img_path).groups()
                    pid = int(pid_part1 + pid_part2 + pid_part3)

                    camid, frameid = pattern_camid.search(img_path).groups()
                    camid = int(camid)
                    p_attribute = self.attribute_dict_all[str(pid)]
                    p_attribute = p_attribute.float()
                    if camid: 
                        camid -= 1
                        viewid = 'Ground'
                    else:
                        viewid = 'Aerial'
                    # if pid == -1:
                    #     continue  # junk images are just ignored
                                       
                    query.append((img_path, pid, camid, viewid))
                else:
                    pid_part1, pid_part2, pid_part3 = pattern_pid.search(img_path).groups()
                    pid = int(pid_part1 + pid_part2 + pid_part3)

                    camid, frameid = pattern_camid.search(img_path).groups()
                    camid = int(camid)
                    p_attribute = self.attribute_dict_all[str(pid)]
                    p_attribute = p_attribute.float()
                    if camid: 
                        camid = 1
                        viewid = 'Ground'
                    else:
                        viewid = 'Aerial'
                    # if pid == -1:
                    #     continue  # junk images are just ignored
                                     
                    gallery.append((img_path, pid, camid, viewid))

        return query, gallery

    def name_of_attribute(self):
        if self.key_attribute:
            print(self.key_attribute)
            return self.key_attribute
        else:
            assert False


@DATASET_REGISTRY.register()
class AG_ReID_v2_W2A(ImageDataset):

    def __init__(self, root='datasets',
                 verbose=True, **kwargs):
        
        self.dataset_dir = root
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'AG-ReID.v2')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"AG-ReID".')
        self.train_dir = osp.join(self.data_dir, 'AG-ReID.v2', 'train_all')

        self.query_dir = osp.join(self.data_dir, 'AG-ReID.v2', 'query')
        self.gallery_dir = osp.join(self.data_dir, 'AG-ReID.v2', 'gallery')
        self.setting_text = osp.join(self.data_dir, 'exp5_wearable_to_aerial.txt')
        self.qut_attribute_path = osp.join(self.data_dir, 'qut_attribute_v8.mat')
        self.attribute_dict_all = self.generate_attribute_dict(self.qut_attribute_path, "qut_attribute")
        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
            self.qut_attribute_path
        ]
        self.check_before_run(required_files)

        train = self._process_dir(self.train_dir, is_train=True)
        query, gallery = self.process_setting_txt(osp.join(self.data_dir, 'AG-ReID.v2'),self.setting_text)
        # query = self._process_dir(self.query_dir, is_train=False)
        # gallery = self._process_dir(self.gallery_dir, is_train=False)

        # if verbose:
        #     print("=> AG-ReID loaded")

        #     self.print_dataset_statistics(train, query, gallery)

        # self.train = train

        # self.query = query
        # self.gallery = gallery
        # self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)

        # self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        # self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
        super(AG_ReID_v2_W2A, self).__init__(train, query, gallery, **kwargs)
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))

        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '**/*.jpg'), recursive=True)
        pattern_pid = re.compile(r'P([-\d]+)T([-\d]+)A([-\d]+)')
        pattern_camid = re.compile(r'C([-\d]+)F([-\d]+)')

        data = []

        for img_path in img_paths:
            fname = osp.split(img_path)[-1]

            pid_part1, pid_part2, pid_part3 = pattern_pid.search(fname).groups()
            pid = int(pid_part1 + pid_part2 + pid_part3)

            camid, frameid = pattern_camid.search(fname).groups()
            camid = int(camid)
            if camid: 
                camid -= 1
                viewid = 'Ground'
            else:
                viewid = 'Aerial'
            p_attribute = self.attribute_dict_all[str(pid)]
            p_attribute = p_attribute.float()

            if is_train:
                pid = "ag-reid" + "_" + str(pid)
            data.append((img_path, pid, camid, viewid, p_attribute))

        return data

    def generate_attribute_dict(self, dir_path: str, dataset: str):

        mat_attribute_train = mat4py.loadmat(dir_path)[dataset]["train"]
        mat_attribute_train = pd.DataFrame(mat_attribute_train, index=mat_attribute_train['image_index']).astype(int)

        mat_attribute_test = mat4py.loadmat(dir_path)[dataset]["test"]
        mat_attribute_test = pd.DataFrame(mat_attribute_test, index=mat_attribute_test['image_index']).astype(int)

        mat_attribute = mat_attribute_train.add(mat_attribute_test, fill_value=0)
        mat_attribute = mat_attribute.drop(['image_index'], axis=1)

        self.key_attribute = list(mat_attribute.keys())

        h, w = mat_attribute.shape
        dict_attribute = dict()

        for i in range(h):
            row = mat_attribute.iloc[i:i + 1, :].values.reshape(-1)
            dict_attribute[str(int(mat_attribute.index[i]))] = torch.tensor(row[0:].astype(int)) * 2 - 3

        return dict_attribute

    def process_setting_txt(self, path, text_path, is_train=True):
        pattern_pid = re.compile(r'P([-\d]+)T([-\d]+)A([-\d]+)')
        pattern_camid = re.compile(r'C([-\d]+)F([-\d]+)')
        with open(text_path,'r') as f:
            query = []
            gallery = []
            for img_path in f:
                # print(img_path)
                split = img_path.split('/')[0]
                # split = split_p.split('_')[0]
                img_path = osp.join(path, img_path[:-1])
                if split == 'query':
                    pid_part1, pid_part2, pid_part3 = pattern_pid.search(img_path).groups()
                    pid = int(pid_part1 + pid_part2 + pid_part3)

                    camid, frameid = pattern_camid.search(img_path).groups()
                    camid = int(camid)
                    p_attribute = self.attribute_dict_all[str(pid)]
                    p_attribute = p_attribute.float()
                    if camid: 
                        camid -= 1
                        viewid = 'Ground'
                    else:
                        viewid = 'Aerial'
                    # if pid == -1:
                    #     continue  # junk images are just ignored
                                       
                    query.append((img_path, pid, camid, viewid))
                else:
                    pid_part1, pid_part2, pid_part3 = pattern_pid.search(img_path).groups()
                    pid = int(pid_part1 + pid_part2 + pid_part3)

                    camid, frameid = pattern_camid.search(img_path).groups()
                    camid = int(camid)
                    p_attribute = self.attribute_dict_all[str(pid)]
                    p_attribute = p_attribute.float()
                    if camid: 
                        camid = 1
                        viewid = 'Ground'
                    else:
                        viewid = 'Aerial'
                    # if pid == -1:
                    #     continue  # junk images are just ignored
                                     
                    gallery.append((img_path, pid, camid, viewid))

        return query, gallery

    def name_of_attribute(self):
        if self.key_attribute:
            print(self.key_attribute)
            return self.key_attribute
        else:
            assert False