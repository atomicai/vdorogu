import annoy
import numpy as np
import torch
from pipeline.SOTA.web_rank.yandex_dssm.source.src.utils import NpCyclicBuffer, predict


class RandomNegativeSampler:
    def __init__(self, backlog_capacity, dim):
        self.backlog_capacity = backlog_capacity
        self.backlog_offset = 0
        self.backlog_size = 0
        self.backlog = np.zeros([backlog_capacity, dim], dtype=np.int64)

    def get(self, count, positives):
        chosen_positive = positives[np.random.randint(0, len(positives))]
        if self.backlog_size < self.backlog_capacity:
            self.backlog[self.backlog_size] = chosen_positive
            self.backlog_size += 1
        else:
            self.backlog[self.backlog_offset] = chosen_positive
            self.backlog_offset = (self.backlog_offset + 1) % self.backlog_capacity
        chosen_negatives = self.backlog[np.random.randint(0, self.backlog_size, count)]
        return chosen_negatives


class HardNegativeSampler:
    def __init__(self, backlog_capacity, dim, matcher):
        self.backlog_capacity = backlog_capacity
        self.dim = dim
        self.matcher = matcher

        self.backlog_offset = 0
        self.backlog_size = 0
        self.backlog = np.zeros([backlog_capacity, dim], dtype=np.int64)

    def add_one(self, vec):
        if self.backlog_size < self.backlog_capacity:
            self.backlog[self.backlog_size] = vec
            self.backlog_size += 1
        else:
            self.backlog[self.backlog_offset] = vec
            self.backlog_offset = (self.backlog_offset + 1) % self.backlog_capacity

    def register(self, titles):
        for tg in titles:
            self.add_one(tg[np.random.randint(0, len(tg))])

    def get_backlog(self):
        return self.backlog[: self.backlog_size]

    def get(self, count, queries_emb, backlog_emb):
        negatives = np.zeros([len(queries_emb), count, self.dim], dtype=np.int64)
        if self.backlog_size < count:
            return negatives.reshape(-1, self.dim)
        queries_emb = queries_emb.unsqueeze(1)
        backlog_emb = backlog_emb.unsqueeze(0)
        sim = self.matcher(queries_emb, backlog_emb).cpu().detach().numpy()
        for i, s in enumerate(sim):
            if np.random.rand() > 1:
                negatives[i] = self.backlog[np.argsort(-s)[:count]]
            else:
                negatives[i] = self.backlog[np.random.randint(0, self.backlog_size, count)]
        return negatives.reshape(-1, self.dim)


def match_batched(matcher, l, r, batch_size=8192 * 6):
    res = np.zeros((l.shape[0], r.shape[1]))
    for i in range(0, r.shape[1], batch_size):
        res[:, i : i + batch_size] = matcher(l, r[:, i : i + batch_size]).cpu().numpy()
    return res


class GaussianNegativeSampler:
    def __init__(self, backlog_capacity, dim, matcher, pred_function, device, model_device):
        self.backlog_capacity = backlog_capacity
        self.model = pred_function
        self.matcher = matcher
        self.backlog_offset = 0
        self.backlog_size = 0
        self.backlog = np.zeros([backlog_capacity, dim], dtype=np.int64)
        self.lmb = 0.00001
        self.dim = dim
        self.device = device
        self.model_device = model_device

    def add_one(self, vec):
        # print('negative_sampling: vec device', torch.cuda.current_device(),  vec.get_device())
        if self.backlog_size < self.backlog_capacity:
            self.backlog[self.backlog_size] = vec
            self.backlog_size += 1
        else:
            self.backlog[self.backlog_offset] = vec
            self.backlog_offset = (self.backlog_offset + 1) % self.backlog_capacity

    def register(self, titles):
        # print('negative_sampling: titles device', torch.cuda.current_device(),  titles.get_device())
        for tg in titles:
            self.add_one(tg[np.random.randint(0, len(tg))])

    def get_backlog(self):
        # print('negative_sampling: backlog device', torch.cuda.current_device(),  self.backlog.get_device())
        return self.backlog[: self.backlog_size]

    def get(self, count, queries_emb, titles_emb):
        negatives = np.zeros([len(queries_emb), count, self.dim], dtype=np.int64)
        if self.backlog_size < count:
            return negatives.reshape(-1, self.dim)
        queries_emb = queries_emb.unsqueeze(1)
        backlog_emb = self.model(torch.from_numpy(self.get_backlog()).to(self.model_device)).unsqueeze(0)
        with torch.no_grad():
            tsim = self.matcher(queries_emb, titles_emb).cpu().numpy()
            bsim = match_batched(self.matcher, queries_emb, backlog_emb)
        tsim = torch.from_numpy(tsim).to(self.device)
        bsim = torch.from_numpy(bsim).to(self.device)
        tmean = torch.mean(tsim, 1, keepdim=True)
        tvar = torch.var(tsim, 1, keepdim=True) + self.lmb
        bmean = torch.mean(bsim, 1, keepdim=True)
        bvar = torch.var(bsim, 1, keepdim=True) + self.lmb
        nvar = 1.0 / (self.lmb + torch.nn.functional.relu(1.0 / tvar - 1.0 / bvar))
        nmean = tmean + (nvar / bvar) * (tmean - bmean)
        bweights = -((bsim - nmean) ** 2) / (2 * nvar)
        bweights -= torch.max(bweights, 1, keepdim=True)[0]
        bweights = torch.exp(bweights)
        # self.valid_negatives = torch.sum(bsim > tmean).item() / bsim.numel()
        ids = torch.multinomial(bweights, count, replacement=True).cpu().numpy()
        negatives = self.backlog[ids.ravel()]
        return negatives.reshape(-1, self.dim)


class AnnGaussianNegativeSampler:
    def __init__(self, backlog_capacity, dim, cache_lifetime, num_nn, pred_function, matcher, device):
        self.backlog_capacity = backlog_capacity
        self.cache_lifetime = cache_lifetime
        self.num_nn = num_nn
        self.model = pred_function
        self.matcher = matcher
        self.iter = -1
        self.backlog = NpCyclicBuffer([backlog_capacity, dim], np.int64)
        self.cache = []
        self.lmb = 0.00001
        self.device = device
        self.dim = dim

    def register(self, titles):
        for tg in titles:
            self.cache.append(tg[np.random.randint(0, len(tg))])
        self.iter += 1
        if self.iter % self.cache_lifetime != 0:
            return
        self.backlog.push_many(self.cache)
        self.cache = []

        self.embeds = predict(self.model, self.backlog.get_data(), 1024, self.device, [64])
        self.index = annoy.AnnoyIndex(self.embeds.shape[1], "euclidean")
        for i, emb in enumerate(self.embeds):
            self.index.add_item(i, emb)
        self.index.build(10)

    def get(self, count, queries_emb, titles_emb):
        negatives = np.zeros([len(queries_emb), count, self.dim], dtype=np.int64)
        if self.backlog.size < count:
            return negatives.reshape(-1, self.dim)
        num_neg_cand = min(self.num_nn, self.backlog.size)
        neighbors = np.random.randint(0, self.backlog.size, [len(queries_emb), num_neg_cand])
        #         neighbors = np.zeros([len(queries_emb), num_neg_cand], dtype=np.int64)
        for i, emb in enumerate(queries_emb.data.cpu().numpy()):
            nb = self.index.get_nns_by_vector(emb, num_neg_cand, 1000)
            neighbors[i, : len(nb)] = nb

        queries_emb = queries_emb.unsqueeze(1)
        #         backlog_emb = self.model(torch.from_numpy(self.backlog.get_data()).to(self.model.device)).unsqueeze(0)
        self.embeds = predict(self.model, self.backlog.get_data(), 1024, self.device, [64])
        backlog_emb = torch.from_numpy(self.embeds[neighbors]).float().to(self.device)
        #         backlog_emb = torch.from_numpy(self.embeds).float().to(self.model.device).unsqueeze(0)
        with torch.no_grad():
            tsim = self.matcher(queries_emb, titles_emb).data.cpu().numpy()
            #             negative_candidates = self.backlog.get_data()[neighbors]
            #             backlog_emb = self.model(torch.from_numpy(negative_candidates).to(self.model.device))
            bsim = self.matcher(queries_emb, backlog_emb).data.cpu().numpy()
        tmean = np.mean(tsim, axis=1, keepdims=True)
        tvar = np.var(tsim, axis=1, keepdims=True) + self.lmb
        self.blogits = (bsim[0] - tmean[0]) / np.sqrt(tvar[0])
        bmean = np.mean(bsim, axis=1, keepdims=True)
        bvar = np.var(bsim, axis=1, keepdims=True) + self.lmb
        nvar = 1.0 / (self.lmb + np.clip(1.0 / tvar - 1.0 / bvar, 0, None))
        nmean = tmean + (nvar / bvar) * (tmean - bmean)
        bweights = -((bsim - nmean) ** 2) / (2 * nvar)
        bweights -= np.max(bweights, axis=1, keepdims=True)
        bweights = np.exp(bweights)
        bweights /= np.sum(bweights, axis=1, keepdims=True)
        self.bweights = bweights[0]
        for i, w in enumerate(bweights):
            pick = np.random.choice(len(w), size=count, replace=True, p=w)
            pick = neighbors[i][pick]
            negatives[i] = self.backlog.get_data()[pick]
        #             negatives[i] = self.backlog[np.argsort(-w)[:count]]
        return negatives.reshape(-1, self.dim)
