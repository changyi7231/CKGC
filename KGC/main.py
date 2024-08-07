import os
import time
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from model import Model
from datasets import KnowledgeGraph, KGDataset


def main(args):
    if args.mode == 'train':
        device = torch.device(args.device)
        if args.save_path is None:
            args.save_path = os.path.join('save', time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)
        print(args.save_path)
        dataset = KnowledgeGraph(args.data_path, args.dataset)
        model = Model(dataset.num_entities, dataset.num_relations, args.dimension).to(device)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
        train(args, device, dataset, model, optimizer)
    elif args.mode == 'test':
        device = torch.device(args.device)
        dataset = KnowledgeGraph(args.data_path, args.dataset)
        model = Model(dataset.num_entities, dataset.num_relations, args.dimension).to(device)
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
    data_loader = DataLoader(KGDataset(dataset.train_data), batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, collate_fn=KGDataset.collate_fn)
    best_mrr, best_epoch = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        total_loss = 0.
        model.train()
        for data in data_loader:
            heads = torch.LongTensor(data[0]).to(device)
            relations = torch.LongTensor(data[1]).to(device)
            tails = torch.LongTensor(data[2]).to(device)
            tails_degree = torch.FloatTensor(data[3]).to(device)
            relations_degree = torch.FloatTensor(data[4]).to(device)
            t_scores, r_scores, factor = model(heads, relations, tails)
            loss = (torch.clamp(F.cross_entropy(t_scores, tails, reduction='none') - tails_degree.log(), min=0).mean()
                    + args.lambda1 * torch.clamp(F.cross_entropy(r_scores, relations, reduction='none') - relations_degree.log(), min=0).mean()
                    + args.lambda2 * factor)
            total_loss += loss.item() * heads.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss /= len(dataset.train_data)
        t1 = time.time()
        print('\n[train: epoch {}], loss: {}, time: {}s'.format(epoch, total_loss, t1 - t0))
        if not (epoch % args.save_interval):
            metrics = test(args, device, dataset, model, epoch, is_test=False)
            if metrics['mrr'] > best_mrr:
                best_mrr, best_epoch = metrics['mrr'], epoch
                save(args.save_path, epoch, model)
    print('best mrr: {} at epoch {}'.format(best_mrr, best_epoch))


def test(args, device, dataset, model, epoch, is_test=True):
    if is_test:
        data_loader = DataLoader(KGDataset(dataset.test_data), batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=KGDataset.collate_fn)
        hr_vocab = dataset.test_hr_vocab
    else:
        data_loader = DataLoader(KGDataset(dataset.valid_data), batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=KGDataset.collate_fn)
        hr_vocab = dataset.valid_hr_vocab
    metrics = defaultdict(float)
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            heads = torch.LongTensor(data[0]).to(device)
            relations = torch.LongTensor(data[1]).to(device)
            tails = torch.LongTensor(data[2]).to(device)
            scores = model(heads, relations, tails)[0]
            scores = scores.detach().cpu().numpy()

            for i, score in enumerate(scores):
                target = score[data[2][i]]
                score[hr_vocab[(data[0][i], data[1][i])]] = -1e8
                rank = np.sum(score >= target) + 1
                metrics['mrr'] += (1.0 / rank)
                metrics['hit@1'] += (rank <= 1)
                metrics['hit@3'] += (rank <= 3)
                metrics['hit@10'] += (rank <= 10)
                metrics['number'] += 1

    for metric in ['mrr', 'hit@1', 'hit@3', 'hit@10']:
        metrics[metric] /= metrics['number']
    t1 = time.time()
    print('[test: epoch {}], mrr: {}, hit@1: {}, hit@3: {}, hit@10: {}, time: {}s'.format(epoch, metrics['mrr'], metrics['hit@1'], metrics['hit@3'], metrics['hit@10'], t1 - t0))
    return metrics


def save(save_path, epoch, model):
    state = {
        'epoch': epoch,
        'model': model.state_dict()
    }
    state_path = os.path.join(save_path, 'epoch_best.pth')
    torch.save(state, state_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knowledge Graph Completion')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='mode')
    parser.add_argument('--device', type=str, default='cuda:4', choices=['cuda:0', 'cpu'], help='device')
    parser.add_argument('--dataset', type=str, default='NELL995', choices=['FB15k', 'FB15k-237', 'NELL995'], help='dataset')
    parser.add_argument('--data_path', type=str, default='datasets', help='data path')
    parser.add_argument('--save_path', type=str, default=None, help='save path')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--save_interval', type=int, default=5, help='number of epochs to save')

    parser.add_argument('--dimension', type=int, default=2000, help='embedding dimension')
    parser.add_argument('--epochs', type=int, default=200, help='epochs')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lambda1', type=float, default=0.0625, help='weight of relation prediction')
    parser.add_argument('--lambda2', type=float, default=0.05, help='weight of regularization')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parse_args = parser.parse_args()
    random.seed(parse_args.seed)
    np.random.seed(parse_args.seed)
    torch.manual_seed(parse_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(parse_args.seed)

    print(parse_args.__dict__)
    main(parse_args)
    # hyper-parameters settings
    # FB15k: lr: 0.01, lambda1: 0.0625, lambda2: 0.005
    # FB15k-237: lr: 0.1, lambda1: 4, lambda2: 0.05
    # NELL995: lr: 0.1, lambda1: 0.0625, lambda2: 0.05
