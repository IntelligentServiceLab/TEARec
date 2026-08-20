import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import sys
import os

_cpp_sampler = None
_cpp_sampler_loaded = False


class Data:
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        self.cache_dir = os.path.join(path, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        self._init_cpp_sampler()

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []
        self.recommendResult = {}

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R_Item_Interacts = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)

        self.train_items, self.test_set = {}, {}

        self.head_items = set()
        self.tail_items = set()
        self.is_tail_item = {}

        head_file = path + '/head.txt'
        tail_file = path + '/tail.txt'

        if os.path.exists(head_file):
            with open(head_file) as f:
                for l in f.readlines():
                    if len(l.strip()) > 0:
                        item_id = int(l.strip())
                        self.head_items.add(item_id)

        if os.path.exists(tail_file):
            with open(tail_file) as f:
                for l in f.readlines():
                    if len(l.strip()) > 0:
                        item_id = int(l.strip())
                        self.tail_items.add(item_id)
                        self.is_tail_item[item_id] = 1

        print(f'Loaded {len(self.head_items)} head items and {len(self.tail_items)} tail items')
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for idx, i in enumerate(train_items):
                        self.R[uid, i] = 1.

                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items

        if self.cpp_sampler is not None:
            self._train_items_dict = []
            for u in range(self.n_users):
                if u in self.train_items and len(self.train_items[u]) > 0:
                    self._train_items_dict.append(self.train_items[u])
                else:
                    self._train_items_dict.append([])
        else:
            self._train_items_dict = None

    def _init_cpp_sampler(self):
        global _cpp_sampler, _cpp_sampler_loaded
        if not _cpp_sampler_loaded:
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                sys.path.insert(0, parent_dir)
                import cppimport
                if hasattr(sys.modules.get('__main__'), 'args'):
                    seed = sys.modules['__main__'].args.seed
                else:
                    seed = 2025
                sample_module = cppimport.imp("source.sample")
                _cpp_sampler = sample_module.FastSampler(seed)
            except Exception as e:
                print(f"C++ sampler load failed: {e}, using Python fallback")
                _cpp_sampler = None
            finally:
                _cpp_sampler_loaded = True
        self.cpp_sampler = _cpp_sampler

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(os.path.join(self.cache_dir, 's_adj_mat.npz'))
            norm_adj_mat = sp.load_npz(os.path.join(self.cache_dir, 's_norm_adj_mat.npz'))
            mean_adj_mat = sp.load_npz(os.path.join(self.cache_dir, 's_mean_adj_mat.npz'))
            print('already load adj matrix', adj_mat.shape, time() - t1)
        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(os.path.join(self.cache_dir, 's_adj_mat.npz'), adj_mat)
            sp.save_npz(os.path.join(self.cache_dir, 's_norm_adj_mat.npz'), norm_adj_mat)
            sp.save_npz(os.path.join(self.cache_dir, 's_mean_adj_mat.npz'), mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)
        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        mean_adj_mat = normalized_adj_single(adj_mat)
        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def negative_pool(self):
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools

    def sample(self):
        return self._sample_original()

    def _sample_original(self):
        if self.cpp_sampler is not None:
            try:
                return self.cpp_sampler.batch_sample(
                    self.exist_users, self._train_items_dict,
                    self.batch_size, self.n_items,
                    pos_samples_per_user=1, neg_samples_per_user=1)
            except Exception:
                pass

        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos(u, num):
            pos_items = self.train_items[u]
            pos_batch = []
            while len(pos_batch) < num:
                pos_id = np.random.randint(0, len(pos_items))
                if pos_items[pos_id] not in pos_batch:
                    pos_batch.append(pos_items[pos_id])
            return pos_batch

        def sample_neg(u, num):
            neg_items = []
            while len(neg_items) < num:
                neg_id = np.random.randint(0, self.n_items)
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos(u, 1)
            neg_items += sample_neg(u, 1)
        return users, pos_items, neg_items

    def sample_all_users_pos_items(self):
        self.all_train_users = []
        self.all_train_pos_items = []
        for u in self.exist_users:
            self.all_train_users += [u] * len(self.train_items[u])
            self.all_train_pos_items += self.train_items[u]

    def epoch_sample(self):
        def sample_neg_items_for_u(u, num):
            neg_items = []
            while len(neg_items) < num:
                neg_id = np.random.randint(0, self.n_items)
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        neg_items = []
        for u in self.all_train_users:
            neg_items += sample_neg_items_for_u(u, 1)
        perm = np.random.permutation(len(self.all_train_users))
        users = np.array(self.all_train_users)[perm]
        pos_items = np.array(self.all_train_pos_items)[perm]
        neg_items = np.array(neg_items)[perm]
        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()
            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
        return split_uids, split_state

    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()
        for uid in all_users_to_test:
            n_iids = len(self.train_items[uid]) + len(self.test_set[uid])
            if n_iids not in user_n_iid:
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)

        split_uids, split_state = [], []
        temp, count, n_rates = [], 1, 0
        n_count = self.n_train + self.n_test
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])
            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)
                split_state.append('#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates))
                temp, n_rates = [], 0
            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)
                split_state.append('#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates))
        return split_uids, split_state
