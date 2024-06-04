import os
from collections import defaultdict

from torch.utils.data.dataset import Dataset


class KnowledgeGraph:
    def __init__(self, path, dataset):
        entity_path = os.path.join(path, dataset, 'entity2id.txt')
        relation_path = os.path.join(path, dataset, 'relation2id.txt')
        train_path = os.path.join(path, dataset, 'train.txt')
        valid_path = os.path.join(path, dataset, 'valid.txt')
        test_path = os.path.join(path, dataset, 'test.txt')

        self.entity2id = {}
        self.relation2id = {}
        with open(entity_path, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                entity, i = line.strip().split('\t')
                self.entity2id[str(entity)] = int(i)
        with open(relation_path, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                relation, i = line.strip().split('\t')
                self.relation2id[str(relation)] = int(i)
        self.num_entities = len(self.entity2id)
        self.num_relations = len(self.relation2id)

        self.train_data = self.read_data(train_path)
        self.valid_data = self.read_data(valid_path)
        self.test_data = self.read_data(test_path)

        self.train_hr_vocab = defaultdict(list)
        self.train_ht_vocab = defaultdict(list)
        for triplet in self.train_data:
            self.train_hr_vocab[(triplet[0], triplet[1])].append(triplet[2])
            self.train_ht_vocab[(triplet[0], triplet[2])].append(triplet[1])
        for i in range(len(self.train_data)):
            self.train_data[i][3] = len(self.train_hr_vocab[(self.train_data[i][0], self.train_data[i][1])])
            self.train_data[i][4] = len(self.train_ht_vocab[(self.train_data[i][0], self.train_data[i][2])])

        self.valid_hr_vocab = defaultdict(list)
        self.test_hr_vocab = defaultdict(list)
        for triplet in self.train_data + self.valid_data:
            self.valid_hr_vocab[(triplet[0], triplet[1])].append(triplet[2])
        for triplet in self.train_data + self.valid_data + self.test_data:
            self.test_hr_vocab[(triplet[0], triplet[1])].append(triplet[2])

    @staticmethod
    def read_data(file_path):
        data = []
        with open(file_path, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                h, r, t = line.strip().split('\t')
                data.append([int(h), int(r), int(t), 0, 0])
        return data


class KGDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(data):
        heads = [_[0] for _ in data]
        relations = [_[1] for _ in data]
        tails = [_[2] for _ in data]
        tails_degree = [_[3] for _ in data]
        relations_degree = [_[4] for _ in data]
        return heads, relations, tails, tails_degree, relations_degree
