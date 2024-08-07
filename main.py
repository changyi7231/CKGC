import os
import time
import random
import argparse
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Model
from datasets import KnowledgeGraphQA, QADataset


def main(args):
    if args.mode == 'train':
        device = torch.device(args.device)
        if args.save_path is None:
            args.save_path = os.path.join('save', time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)
        print(args.save_path)
        dataset = KnowledgeGraphQA(args.data_path, args.dataset)
        degree = torch.zeros(dataset.num_entities, dataset.num_relations).to(device)
        for (h, r, t) in dataset.train:
            degree[h, r] += 1
        degree = torch.clamp(degree, min=args.alpha)
        model = Model(dataset.num_entities, dataset.num_relations, args.dimension, degree, args.fractions).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train(args, device, dataset, model, optimizer)
    elif args.mode == 'test':
        device = torch.device(args.device)
        dataset = KnowledgeGraphQA(args.data_path, args.dataset)
        degree = torch.zeros(dataset.num_entities, dataset.num_relations).to(device)
        for (h, r, t) in dataset.train:
            degree[h, r] += 1
        degree = torch.clamp(degree, min=args.alpha)
        model = Model(dataset.num_entities, dataset.num_relations, args.dimension, degree, args.fractions).to(device)
        state_file = os.path.join(args.save_path, 'epoch_best.pth')
        if not os.path.isfile(state_file):
            raise RuntimeError('file {} is not found'.format(state_file))
        print('load checkpoint {}'.format(state_file))
        checkpoint = torch.load(state_file, device)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        test(args, device, dataset, model, epoch, is_test=True)
    else:
        raise RuntimeError('wrong mode')


def train(args, device, dataset, model, optimizer):
    if args.pretrain_path is not None:
        state_file = os.path.join(args.pretrain_path, 'epoch_best.pth')
        if not os.path.isfile(state_file):
            raise RuntimeError('file {} is not found'.format(state_file))
        print('load checkpoint {}'.format(state_file))
        checkpoint = torch.load(state_file, device)
        model.kgc_model.load_state_dict(checkpoint['model'])
        model.kgc_model.entity.weight.requires_grad = False
        model.kgc_model.relation.weight.requires_grad = False

    data_loader = DataLoader(QADataset(dataset.train_queries), batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, collate_fn=QADataset.train_collate_fn)
    best_mrr, best_epoch = 0.0, 0
    save(args.save_path, 0, model)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        total_loss = 0.
        model.train()
        model.recomputing = True
        for data in data_loader:
            structures, queries, labels = data[0], data[1], data[2]
            queries_dict = defaultdict(list)
            indexes_dict = defaultdict(list)
            for i, query in enumerate(queries):
                queries_dict[dataset.id2query[structures[i]]].append(query)
                indexes_dict[dataset.id2query[structures[i]]].append(i)
            for query_structure in queries_dict.keys():
                queries_dict[query_structure] = torch.LongTensor(queries_dict[query_structure]).to(device)
                indexes_dict[query_structure] = torch.LongTensor(indexes_dict[query_structure]).to(device)
            labels = labels.to(device)

            scores = model(queries_dict, indexes_dict)
            weight = (1 + labels * (dataset.num_entities/labels.sum(-1).unsqueeze(-1)-2))/(dataset.num_entities-labels.sum(-1).unsqueeze(-1))
            loss = (weight * F.binary_cross_entropy(scores, labels, reduction='none')).sum(-1).mean()
            total_loss += loss.item() * len(structures)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss /= len(dataset.train_queries)
        t1 = time.time()
        print('\n[train: epoch {}], loss: {}, time: {}s'.format(epoch, total_loss, t1 - t0))
        if not (epoch % args.save_interval):
            mrr = test(args, device, dataset, model, epoch, is_test=False)
            if mrr > best_mrr:
                best_mrr, best_epoch = mrr, epoch
                save(args.save_path, epoch, model)
    print('best mrr: {} at epoch {}'.format(best_mrr, best_epoch))


def test(args, device, dataset, model, epoch, is_test=True):
    if is_test:
        data_loader = DataLoader(QADataset(dataset.test_queries), batch_size=args.test_batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=QADataset.test_collate_fn)
        adj_list = [[] for _ in range(dataset.num_relations)]
        for (h, r, t) in dataset.train + dataset.valid:
            adj_list[r].append([h, t])
    else:
        data_loader = DataLoader(QADataset(dataset.valid_queries), batch_size=args.test_batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=QADataset.test_collate_fn)
        adj_list = [[] for _ in range(dataset.num_relations)]
        for (h, r, t) in dataset.train:
            adj_list[r].append([h, t])
    t0 = time.time()
    model.eval()
    filename = os.path.join(args.save_path, str(int(is_test)) + '_' + 'epoch_' + str(epoch) + '_' + str(args.fractions)
                            + '_' + str(args.alpha) + '_' + str(args.epsilon) + '.pth')
    if os.path.exists(filename):
        print('load file {}'.format(filename))
        model.adj = torch.load(filename, map_location=device)
    else:
        ratio = 0.0
        model.recomputing = True
        for i in tqdm(range(dataset.num_relations)):
            with torch.no_grad():
                relation_embedding = torch.zeros(dataset.num_entities, dataset.num_entities).to(torch.float)
                r = torch.LongTensor([i]).to(device)
                batch_size = 1000
                for s in range(0, dataset.num_entities, batch_size):
                    t = min(dataset.num_entities, s + batch_size)
                    h = torch.arange(s, t).to(device)
                    scores = F.softmax(model.kgc_model(h, r), dim=-1)
                    scores = scores * model.degree[s:t, i].unsqueeze(-1)
                    scores = torch.clamp(scores, max=1.0)
                    scores = scores * torch.clamp(1 + model.weight[s:t, i], min=1e-5).unsqueeze(-1)
                    scores = torch.clamp(scores, max=1.0)
                    scores = scores.masked_fill(scores <= args.epsilon, 0.0)
                    relation_embedding[s:t, :] = scores.cpu()
            for (h, t) in adj_list[i]:
                relation_embedding[h, t] = 1.0
            ratio = ratio + ((relation_embedding == 0).sum() / (
                    relation_embedding.size(0) * relation_embedding.size(1))).item()
            print('sparsity level: {}'.format(ratio / (i + 1)))
            # add fractions
            dim = dataset.num_entities // args.fractions
            rest = dataset.num_entities - args.fractions * dim
            for j in range(args.fractions):
                s = j * dim
                t = (j + 1) * dim
                if j == (args.fractions - 1):
                    t += rest
                model.adj.append(relation_embedding[s:t, :].to_sparse().to(device))
        torch.save(model.adj, filename)
        print('save file {}'.format(filename))
    metrics = defaultdict(lambda: defaultdict(float))
    average_metrics_p = defaultdict(float)
    average_metrics_n = defaultdict(float)
    model.recomputing = False
    with torch.no_grad():
        for data in data_loader:
            structures, queries, easy_answers, hard_answers = data[0], data[1], data[2], data[3]
            queries_dict = defaultdict(list)
            indexes_dict = defaultdict(list)
            for i, query in enumerate(queries):
                queries_dict[dataset.id2query[structures[i]]].append(query)
                indexes_dict[dataset.id2query[structures[i]]].append(i)
            for query_structure in queries_dict.keys():
                queries_dict[query_structure] = torch.LongTensor(queries_dict[query_structure]).to(device)
                indexes_dict[query_structure] = torch.LongTensor(indexes_dict[query_structure]).to(device)
            scores = model(queries_dict, indexes_dict)
            scores = scores.detach().cpu().numpy()

            for i, score in enumerate(scores):
                target = score[hard_answers[i]]
                score[easy_answers[i] + hard_answers[i]] = -1e8
                rank = np.greater_equal(np.expand_dims(score, 0), np.expand_dims(target, -1)).sum(-1) + 1
                metrics[dataset.id2query_name[structures[i]]]['mrr'] += (1.0 / rank).mean()
                metrics[dataset.id2query_name[structures[i]]]['hit@1'] += (rank <= 1).mean()
                metrics[dataset.id2query_name[structures[i]]]['hit@3'] += (rank <= 3).mean()
                metrics[dataset.id2query_name[structures[i]]]['hit@10'] += (rank <= 10).mean()
                metrics[dataset.id2query_name[structures[i]]]['number'] += 1
    model.adj = list()
    num_query_structures_p = 0
    for query_name in ['1p', '2p', '3p', '2i', '3i', 'pi', 'ip', '2u', 'up']:
        for metric in ['mrr', 'hit@1', 'hit@3', 'hit@10']:
            metrics[query_name][metric] /= metrics[query_name]['number']
            average_metrics_p[metric] += metrics[query_name][metric]
        num_query_structures_p += 1
        print('[test: epoch {}], query name: {}, mrr: {}, hit@1: {}, hit@3: {}, hit@10: {}'.format(epoch, query_name, metrics[query_name]['mrr'], metrics[query_name]['hit@1'], metrics[query_name]['hit@3'], metrics[query_name]['hit@10']))
    for metric in average_metrics_p.keys():
        average_metrics_p[metric] /= num_query_structures_p

    num_query_structures_n = 0
    for query_name in ['2in', '3in', 'inp', 'pin', 'pni']:
        for metric in ['mrr', 'hit@1', 'hit@3', 'hit@10']:
            metrics[query_name][metric] /= metrics[query_name]['number']
            average_metrics_n[metric] += metrics[query_name][metric]
        num_query_structures_n += 1
        print('[test: epoch {}], query name: {}, mrr: {}, hit@1: {}, hit@3: {}, hit@10: {}'.format(epoch, query_name, metrics[query_name]['mrr'], metrics[query_name]['hit@1'], metrics[query_name]['hit@3'], metrics[query_name]['hit@10']))
    for metric in average_metrics_n.keys():
        average_metrics_n[metric] /= num_query_structures_n
    t1 = time.time()
    print('[test: epoch {}], mrr: {}, hit@1: {}, hit@3: {}, hit@10: {}, time: {}s'.format(epoch, average_metrics_p['mrr'], average_metrics_p['hit@1'], average_metrics_p['hit@3'], average_metrics_p['hit@10'], t1 - t0))
    print('[test: epoch {}], mrr: {}, hit@1: {}, hit@3: {}, hit@10: {}, time: {}s'.format(epoch, average_metrics_n['mrr'], average_metrics_n['hit@1'], average_metrics_n['hit@3'], average_metrics_n['hit@10'], t1 - t0))
    return (average_metrics_p['mrr']*9+average_metrics_n['mrr']*5)/14


def save(save_path, epoch, model):
    state = {
        'epoch': epoch,
        'model': model.state_dict()
    }
    state_path = os.path.join(save_path, 'epoch_best.pth')
    torch.save(state, state_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Complex Logical Query Answering')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='mode')
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cpu'], help='device')
    parser.add_argument('--dataset', type=str, default='FB15k-237', choices=['FB15k', 'FB15k-237', 'NELL995'], help='dataset')
    parser.add_argument('--data_path', type=str, default='datasets', help='data path')
    parser.add_argument('--save_path', type=str, default=None, help='save path')
    parser.add_argument('--pretrain_path', type=str, default='save/FB15k-237', help='pretrain path')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--save_interval', type=int, default=1, help='number of epochs to save')

    parser.add_argument('--dimension', type=int, default=2000, help='embedding dimension')
    parser.add_argument('--epochs', type=int, default=5, help='epochs')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='test batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--alpha', type=float, default=0.1, help='minimum degree')
    parser.add_argument('--epsilon', type=float, default=5e-5, help='minimum value')
    parser.add_argument('--fractions', type=int, default=5, help='fractions for adjacent tensor')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parse_args = parser.parse_args()
    random.seed(parse_args.seed)
    np.random.seed(parse_args.seed)
    torch.manual_seed(parse_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(parse_args.seed)

    print(parse_args.__dict__)
    main(parse_args)
    # hyper-parameters settings
    # FB15k: epsilon: 0.0005
    # FB15k-237: epsilon: 0.00005
    # NELL995: epsilon: 0.00001
