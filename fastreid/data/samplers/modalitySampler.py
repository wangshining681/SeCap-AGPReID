import itertools
from typing import Optional
import copy
import numpy as np
from torch.utils.data import Sampler
import random
from fastreid.utils import comm
from collections import defaultdict

class RandomIdentityModalitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        self.index_dic_aerial = defaultdict(list) #dict with list value
        self.index_dic_ground = defaultdict(list) #dict with list value
        for index, (_, pid, camid, view_id) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
            if view_id == 'Aerial':
                self.index_dic_aerial[pid].append(index)
            else:
                self.index_dic_ground[pid].append(index)
        self.pids = list(self.index_dic.keys())
        # tmp = []
        # for pid in self.pids:
        #     idxs = self.index_dic[pid]
        #     idx_g = self.index_dic_ground[pid]
        #     idx_a = self.index_dic_aerial[pid]
        #     if len(idxs) > num_instances and len(idx_a) > num_instances/2 and len(idx_g) > num_instances/2:
        #         tmp.append(pid)
        # self.pids = tmp
        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num % self.num_instances < self.num_instances:
                num = num - num % self.num_instances + self.num_instances
            self.length += num

    def __iter__(self):
        avai_pids = copy.deepcopy(self.pids)
        batch_idxs_dict = defaultdict(list)
        batch_idxs_aerial_dict = defaultdict(list)
        batch_idxs_ground_dict = defaultdict(list)
        batch_indices = []
        batch_idxs_aerial = []
        batch_idxs_ground = []

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])

            idxs_aerial = copy.deepcopy(self.index_dic_aerial[pid])
            idxs_ground = copy.deepcopy(self.index_dic_ground[pid])

            idx_rem = self.num_instances - len(idxs) % self.num_instances
            a_rem = int(self.num_instances / 2 - len(idxs_aerial) % (self.num_instances / 2))
            g_rem = int(self.num_instances / 2 - len(idxs_ground) % (self.num_instances / 2))
            # 图片数目向上取整，保证为instance_num的倍数

            if int(idx_rem) < self.num_instances:
                rem = np.random.choice(idxs, size=int(idx_rem), replace=True)
                idxs.extend(rem.tolist())

            if int(a_rem) < (self.num_instances/2):
                rem = np.random.choice(idxs_aerial, size=int(a_rem), replace=True)
                idxs_aerial.extend(rem)
            
            if int(g_rem) < (self.num_instances/2):
                rem = np.random.choice(idxs_ground, size=int(g_rem), replace=True)
                idxs_ground.extend(rem)

            # 保证两种模态的图片数目相等
            if len(idxs_ground) < len(idxs_aerial):
                rem = np.random.choice(idxs_ground, size=(len(idxs_aerial)-len(idxs_ground)), replace=True)
                idxs_ground.extend(rem)

            elif len(idxs_ground) > len(idxs_aerial):
                rem = np.random.choice(idxs_aerial, size=(len(idxs_ground)-len(idxs_aerial)), replace=True)
                idxs_aerial.extend(rem)
                
            random.shuffle(idxs)
            random.shuffle(idxs_aerial)
            random.shuffle(idxs_ground)
            
            batch_idxs_aerial = []
            batch_idxs_ground = []
            batch_idxs = []
            for i in range(len(idxs_aerial)):
                batch_idxs_aerial.append(idxs_aerial[i])
                batch_idxs_ground.append(idxs_ground[i])
                # num_instance中一半为vis，一半为the
                if len(batch_idxs_aerial) == self.num_instances / 2:
                    batch_idxs.extend(batch_idxs_aerial)
                    batch_idxs.extend(batch_idxs_ground)
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs_aerial_dict[pid].append(batch_idxs_aerial)
                    batch_idxs_ground_dict[pid].append(batch_idxs_ground)
                    batch_idxs_aerial = []
                    batch_idxs_ground = []
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)

            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
            # for pid in selected_pids:
            #     batch_idxs = batch_idxs_thermal_dict[pid].pop(0)
            #     final_idxs.extend(batch_idxs)
            #     if len(batch_idxs_thermal_dict[pid]) == 0 
            #         avai_pids.remove(pid)
            # for pid in selected_pids:
            #     batch_idxs = batch_idxs_visible_dict[pid].pop(0)
            #     final_idxs.extend(batch_idxs)
        return iter(final_idxs)
    def __len__(self):
        return self.length