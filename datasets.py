import os
import torch
from torch.utils.data.dataset import Dataset


class KnowledgeGraphQA:
    def __init__(self, path, dataset):
        entity_path = os.path.join(path, dataset, 'entity2id.txt')
        relation_path = os.path.join(path, dataset, 'relation2id.txt')
        operator_path = os.path.join(path, dataset, 'operator2id.txt')
        query_path = os.path.join(path, dataset, 'query2id.txt')
        train_path = os.path.join(path, dataset, 'train.txt')
        valid_path = os.path.join(path, dataset, 'valid.txt')
        test_path = os.path.join(path, dataset, 'test.txt')
        train_queries_path = os.path.join(path, dataset, 'train-queries.txt')
        valid_queries_path = os.path.join(path, dataset, 'valid-queries.txt')
        test_queries_path = os.path.join(path, dataset, 'test-queries.txt')

        self.entity2id = {}
        self.relation2id = {}
        self.operator2id = {}
        self.operator_name2id = {}
        self.query2id = {}
        self.query_name2id = {}
        with open(entity_path, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                entity, i = line.strip().split('\t')
                self.entity2id[str(entity)] = int(i)
        with open(relation_path, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                relation, i = line.strip().split('\t')
                self.relation2id[str(relation)] = int(i)
        with open(operator_path, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                name, operator, i = line.strip().split('\t')
                self.operator2id[str(operator)] = int(i)
                self.operator_name2id[str(name)] = int(i)
        with open(query_path, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                name, query, i = line.strip().split('\t')
                self.query2id[str(query)] = int(i)
                self.query_name2id[str(name)] = int(i)
        self.id2entity = {value: key for key, value in self.entity2id.items()}
        self.id2relation = {value: key for key, value in self.relation2id.items()}
        self.id2operator = {value: key for key, value in self.operator2id.items()}
        self.id2operator_name = {value: key for key, value in self.operator_name2id.items()}
        self.id2query = {value: key for key, value in self.query2id.items()}
        self.id2query_name = {value: key for key, value in self.query_name2id.items()}
        self.num_entities = len(self.entity2id)
        self.num_relations = len(self.relation2id)
        self.num_operators = len(self.operator2id)
        self.num_queries = len(self.query2id)

        self.train = self.read_data(train_path)
        self.valid = self.read_data(valid_path)
        self.test = self.read_data(test_path)

        self.train_queries = self.read_queries(train_queries_path, is_train=True)
        self.valid_queries = self.read_queries(valid_queries_path, is_train=False)
        self.test_queries = self.read_queries(test_queries_path, is_train=False)

    @staticmethod
    def read_data(file_path):
        data = []
        with open(file_path, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                h, r, t = line.strip().split('\t')
                data.append([int(h), int(r), int(t)])
        return data

    def read_queries(self, file_path, is_train=False):
        data = []
        with open(file_path, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                structure, query, easy_answers, hard_answers = line.strip().split('\t')
                if is_train:
                    if int(structure) in [0, 3, 4, 9, 10]:
                        structure = int(structure)
                        query = [int(i) for i in query.strip().split(' ')]
                        easy_answers = [] if easy_answers.strip() == '' else [int(i) for i in
                                                                              easy_answers.strip().split(' ')]
                        hard_answers = [] if hard_answers.strip() == '' else [int(i) for i in
                                                                              hard_answers.strip().split(' ')]
                        data.append([structure, query, easy_answers, hard_answers, self.num_entities])
                else:
                    structure = int(structure)
                    query = [int(i) for i in query.strip().split(' ')]
                    easy_answers = [] if easy_answers.strip() == '' else [int(i) for i in
                                                                          easy_answers.strip().split(' ')]
                    hard_answers = [] if hard_answers.strip() == '' else [int(i) for i in
                                                                          hard_answers.strip().split(' ')]
                    data.append([structure, query, easy_answers, hard_answers, self.num_entities])
        return data


class QADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def train_collate_fn(data):
        structures = [_[0] for _ in data]
        queries = [_[1] for _ in data]
        labels = torch.zeros(len(data), data[0][4])
        for i in range(len(data)):
            labels[i, data[i][3]] = 1
        return structures, queries, labels

    @staticmethod
    def test_collate_fn(data):
        structures = [_[0] for _ in data]
        queries = [_[1] for _ in data]
        easy_answers = [_[2] for _ in data]
        hard_answers = [_[3] for _ in data]
        return structures, queries, easy_answers, hard_answers
