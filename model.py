import torch
import torch.nn as nn
import torch.nn.functional as F


class DistMult(nn.Module):
    def __init__(self, num_entities, num_relations, dimension):
        super().__init__()
        self.entity = nn.Embedding(num_entities, dimension)
        self.relation = nn.Embedding(num_relations, dimension)

        bound = 1e-2
        nn.init.uniform_(self.entity.weight, -bound, bound)
        nn.init.uniform_(self.relation.weight, -bound, bound)

    def forward(self, heads, relations):
        h = self.entity(heads)
        r = self.relation(relations)
        scores = torch.matmul(h*r, self.entity.weight.t())
        return scores


class SimplE(nn.Module):
    def __init__(self, num_entities, num_relations, dimension):
        super().__init__()
        self.entity = nn.Embedding(num_entities, dimension * 2)
        self.relation = nn.Embedding(num_relations, dimension * 2)

        bound = 1e-2
        nn.init.uniform_(self.entity.weight, -bound, bound)
        nn.init.uniform_(self.relation.weight, -bound, bound)

    def forward(self, heads, relations):
        h = self.entity(heads)
        r = self.relation(relations)
        h1, h2 = torch.chunk(h, 2, dim=-1)
        r1, r2 = torch.chunk(r, 2, dim=-1)
        x = torch.cat([h2 * r2, h1 * r1], dim=-1)
        scores = torch.matmul(x, self.entity.weight.t())
        return scores


class ComplEx(nn.Module):
    def __init__(self, num_entities, num_relations, dimension):
        super().__init__()
        self.entity = nn.Embedding(num_entities, dimension * 2)
        self.relation = nn.Embedding(num_relations, dimension * 2)

        scale = 1e-2
        nn.init.uniform_(self.entity.weight, -scale, scale)
        nn.init.uniform_(self.relation.weight, -scale, scale)

    def forward(self, heads, relations):
        h = self.entity(heads)
        r = self.relation(relations)
        h1, h2 = torch.chunk(h, 2, dim=-1)
        r1, r2 = torch.chunk(r, 2, dim=-1)
        x = torch.cat([h1 * r1 - h2 * r2, h2 * r1 + h1 * r2], dim=-1)
        scores = torch.matmul(x, self.entity.weight.t())
        return scores


class Model(nn.Module):
    def __init__(self, num_entities, num_relations, dimension, degree, fractions):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.degree = degree
        self.fractions = fractions

        self.weight = nn.Parameter(torch.zeros(num_entities, num_relations))
        self.kgc_model = ComplEx(num_entities, num_relations, dimension)
        self.adj = list()
        self.recomputing = True

    def forward(self, queries_dict, indexes_dict):
        all_embeddings, all_indexes = [], []
        for query_structure in queries_dict.keys():
            embeddings = self.embed_query(queries_dict[query_structure], query_structure)
            all_embeddings.append(embeddings)
            all_indexes.append(indexes_dict[query_structure])
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_indexes = torch.argsort(torch.cat(all_indexes, dim=0))
        scores = all_embeddings[all_indexes]
        return scores

    def embed_query(self, queries, query_structure):
        # compute the postfix expressions with same structure
        query_structure = query_structure.strip().split(' ')
        stack = []
        for i in range(len(query_structure)):
            if query_structure[i] == 'e':
                stack.append(F.one_hot(queries[:, i], self.num_entities))
            elif query_structure[i] == 'r':
                stack.append(queries[:, i])
            else:
                if query_structure[i] == 'p':
                    operand1 = stack.pop()
                    operand2 = stack.pop()
                    result = self.projection(operand2, operand1)
                elif query_structure[i] == 'i':
                    operand1 = stack.pop()
                    operand2 = stack.pop()
                    result = self.intersection(operand2, operand1)
                elif query_structure[i] == 'u':
                    operand1 = stack.pop()
                    operand2 = stack.pop()
                    result = self.union(operand2, operand1)
                elif query_structure[i] == 'c':
                    operand1 = stack.pop()
                    result = self.complement(operand1)
                else:
                    raise ValueError('wrong operator')
                stack.append(result)
        final_result = stack.pop()
        return final_result

    def projection(self, operand1, operand2):
        if self.recomputing:
            # we only use the top-1 element in operand1 in training
            scores, operand1 = torch.max(operand1, dim=-1)
            scores = scores.unsqueeze(-1) * F.softmax(self.kgc_model(operand1, operand2), dim=-1)
            scores = scores * self.degree[operand1, operand2].unsqueeze(-1)
            scores = torch.clamp(scores, max=1.0)
            scores = scores * torch.clamp(1 + self.weight[operand1, operand2], min=1e-5).unsqueeze(-1)
            result = torch.clamp(scores, max=1.0)
        else:
            dim = self.num_entities // self.fractions
            rest = self.num_entities - self.fractions * dim
            result = torch.zeros_like(operand1)
            for i in range(self.fractions):
                s = i * dim
                t = (i + 1) * dim
                if i == (self.fractions - 1):
                    t += rest
                fraction_operand1 = operand1[:, s:t]
                if fraction_operand1.sum().item() == 0:
                    continue
                nonzero = torch.nonzero(fraction_operand1, as_tuple=True)[1]
                fraction_operand1 = fraction_operand1[:, nonzero]
                fraction_adj_list = [self.adj[self.fractions * r + i].to_dense()[nonzero, :] for r in operand2]
                fraction_adj = torch.stack(fraction_adj_list, dim=0)
                fraction_result = torch.max(fraction_operand1.unsqueeze(-1) * fraction_adj, dim=1)[0]
                result = torch.maximum(result, fraction_result)
        return result

    def intersection(self, *operands):
        result = torch.prod(torch.stack(operands, dim=0), dim=0)
        return result

    def union(self, *operands):
        operands = [self.complement(operand) for operand in operands]
        result = self.complement(self.intersection(*operands))
        return result

    def complement(self, operand):
        result = 1 - operand
        return result
