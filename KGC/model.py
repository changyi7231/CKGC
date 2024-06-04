import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_entities, num_relations, dimension):
        super().__init__()
        self.entity = nn.Embedding(num_entities, dimension * 2)
        self.relation = nn.Embedding(num_relations, dimension * 2)

        bound = 1e-2
        nn.init.uniform_(self.entity.weight, -bound, bound)
        nn.init.uniform_(self.relation.weight, -bound, bound)

    def forward(self, heads, relations, tails):
        h = self.entity(heads)
        r = self.relation(relations)
        t = self.entity(tails)
        h1, h2 = torch.chunk(h, 2, dim=-1)
        r1, r2 = torch.chunk(r, 2, dim=-1)
        t1, t2 = torch.chunk(t, 2, dim=-1)
        x1 = torch.cat([h1 * r1 - h2 * r2, h2 * r1 + h1 * r2], dim=-1)
        x2 = torch.cat([h1 * t1 + h2 * t2, h1 * t2 - h2 * t1], dim=-1)
        t_scores = torch.matmul(x1, self.entity.weight.t())
        r_scores = torch.matmul(x2, self.relation.weight.t())
        factor = ((h1 ** 2 + h2 ** 2) ** (3 / 2)).sum(-1).mean() + ((r1 ** 2 + r2 ** 2) ** (3 / 2)).sum(-1).mean() + (
                (t1 ** 2 + t2 ** 2) ** (3 / 2)).sum(-1).mean()
        return t_scores, r_scores, factor
