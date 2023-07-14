#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import sys
from itertools import groupby

import numpy as np
from scipy import stats
from sklearn.metrics import precision_recall_curve, roc_auc_score


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def calc_average_metric(queries_data, calc_metric, param):
    result_score = 0.0
    num_good_queries = 0
    num_bad_queries = 0
    for new_query in queries_data:
        new_query_score = calc_metric(queries_data, new_query, param)
        if new_query_score is None:
            num_bad_queries += 1
        else:
            num_good_queries += 1
            result_score += new_query_score
    if calc_metric == calc_defective_pairs_by_query:
        return result_score
    if num_good_queries == 0:
        return 0
    return result_score / num_good_queries


def calc_ndcg_by_query(queries_data, new_query, k):
    query_predicts = queries_data[new_query]["predictions"]
    query_relevance = queries_data[new_query]["relevance"]
    if all(query_relevance[0] == i for i in query_relevance):
        return None
    sort_query_relevance = [
        x for (y, x) in sorted(zip(query_predicts, query_relevance), reverse=True, key=lambda x: x[0])
    ]
    return calc_ndcg(sort_query_relevance, k)


def calc_dcg_by_query(queries_data, new_query, k):
    query_predicts = queries_data[new_query]["predictions"]
    query_relevance = queries_data[new_query]["relevance"]
    if all(query_relevance[0] == i for i in query_relevance):
        return None
    sort_query_relevance = [x for (y, x) in sorted(zip(query_predicts, query_relevance), reverse=True)]
    return calc_dcg(sort_query_relevance, k, verbose=False)


def calc_max_dcg_by_query(queries_data, new_query, k):
    r = queries_data[new_query]["relevance"]
    return calc_dcg(sorted(r, reverse=True), k, verbose=False)


def calc_ndcg(r, k, verbose=False):
    dcg_max = calc_dcg(sorted(r, reverse=True), k, verbose)
    # empty list of marks?
    if not dcg_max:
        return 0.0
    return calc_dcg(r, k, verbose) / dcg_max


def calc_dcg(r, k, verbose=False):
    if k is None:
        k = len(r)
    r = np.asfarray(r)[:k]
    if r.size:
        # local_dcg = np.sum( (r) / np.log2(np.arange(2, r.size + 2)))
        local_dcg = np.sum((np.power(2.0, r) - 1) / np.log2(np.arange(2, r.size + 2)))
        if verbose:
            print(local_dcg)
        return local_dcg
    return 0.0


def calc_precision_by_query(queries_data, new_query, p, threshold=3):
    query_predicts = queries_data[new_query]["predictions"]
    query_relevance = queries_data[new_query]["relevance"]
    if p is None:
        new_p = 10
    else:
        new_p = p
    if all(query_relevance[0] == i for i in query_relevance):
        return None
    sort_query_relevance = [x for (y, x) in sorted(zip(query_predicts, query_relevance), reverse=True)]
    top_rel = np.array(sort_query_relevance[0:new_p])
    relevant_docs_count = len(top_rel[top_rel >= threshold])
    if relevant_docs_count > 0:
        new_score = relevant_docs_count / float(new_p)
        return new_score
    return None


def calc_recall_by_query(queries_data, new_query, r, threshold=3):
    query_predicts = queries_data[new_query]["predictions"]
    query_relevance = queries_data[new_query]["relevance"]
    if all(query_relevance[0] == i for i in query_relevance):
        return None
    sort_query_relevance = [x for (y, x) in sorted(zip(query_predicts, query_relevance), reverse=True)]
    num_of_rel = sum(1 * (np.array(query_relevance) >= threshold))
    top_rel = np.array(sort_query_relevance[0:num_of_rel])
    relevant_docs_count = len(top_rel[top_rel >= threshold])
    if relevant_docs_count > 0:
        new_score = relevant_docs_count / float(num_of_rel)
        return new_score
    return None


def calc_f1_by_query(queries_data, new_query, p, threshold=3):
    if p is None:
        new_p = 10
    else:
        new_p = p
    prec_score = calc_precision_by_query(queries_data, new_query, new_p)
    rec_score = calc_recall_by_query(queries_data, new_query, None)
    if prec_score is None or rec_score is None:
        return None
    return (2 * prec_score * rec_score) / (prec_score + rec_score)


def calc_defective_pairs_by_query(queries_data, new_query, d):
    query_predicts = queries_data[new_query]["predictions"]
    query_relevance = queries_data[new_query]["relevance"]
    if all(query_relevance[0] == i for i in query_relevance):
        return None
    if d is None:
        new_d = len(query_relevance)
    else:
        new_d = d
    defect_pairs = 0
    sort_query_relevance = [x for (y, x) in sorted(zip(query_predicts, query_relevance), reverse=True)]
    sort_query_relevance = sort_query_relevance[0:new_d]
    for i in range(0, len(sort_query_relevance)):
        rels = np.array(sort_query_relevance[0:i])
        defect_pairs += len(rels[rels < sort_query_relevance[i]])
    return 1.0 * defect_pairs


def map_mark_to_prob(mark):
    if mark < 2:
        return 0.0
    if mark == 2:
        return 0.07
    if mark == 3:
        return 0.14
    if mark == 4:
        return 0.41
    if mark == 5:
        return 0.61


def calc_pfound_by_query(queries_data, new_query, f):
    p_out = 0.15
    query_predicts = queries_data[new_query]["predictions"]
    query_relevance = queries_data[new_query]["relevance"]
    if all(query_relevance[0] == i for i in query_relevance):
        return None
    sort_query_relevance = [x for (y, x) in sorted(zip(query_predicts, query_relevance), reverse=True)]
    if f is None:
        new_f = 10
    else:
        new_f = f
    p_i = 1.0
    pfound = 0.0
    for i in range(0, min(new_f, len(query_predicts))):
        new_prob = map_mark_to_prob(sort_query_relevance[i])
        pfound += new_prob * p_i
        p_i = p_i * (1 - new_prob) * (1 - p_out)
    return pfound


def calc_err_by_query(queries_data, new_query, e):
    query_predicts = queries_data[new_query]["predictions"]
    query_relevance = queries_data[new_query]["relevance"]
    if all(query_relevance[0] == i for i in query_relevance):
        return None
    sort_query_relevance = [x for (y, x) in sorted(zip(query_predicts, query_relevance), reverse=True)]
    if e is None:
        new_e = min(10, len(query_predicts))
    else:
        new_e = min(e, len(query_predicts))
    p_i = 1.0
    err = 0.0
    for i in range(1, new_e + 1):
        new_prob = map_mark_to_prob(sort_query_relevance[i - 1])
        err += 1.0 * new_prob * p_i / (1.0 * i)
        p_i = p_i * (1.0 - new_prob)
    return err


def calc_best_by_query(queries_data, new_query, args):
    query_predicts = queries_data[new_query]["predictions"]
    query_relevance = queries_data[new_query]["relevance"]
    if all(query_relevance[0] == i for i in query_relevance):
        return None
    sort_query_relevance = [x for (y, x) in sorted(zip(query_predicts, query_relevance), reverse=True)]
    if sort_query_relevance[0] == max(query_relevance):
        return 1
    return 0


def calc_good_by_query(queries_data, new_query, threshold):
    query_predicts = queries_data[new_query]["predictions"]
    query_relevance = queries_data[new_query]["relevance"]
    if all(query_relevance[0] == i or i < threshold for i in query_relevance):
        return None
    sort_query_relevance = [x for (y, x) in sorted(zip(query_predicts, query_relevance), reverse=True)]
    if sort_query_relevance[0] >= threshold:
        return 1
    return 0


def test_model(y_pred, y_true, groups, k=5):
    res = {}
    for key, group in groupby(zip(y_pred, y_true, groups), key=lambda x: x[2]):
        group = list(zip(*group))
        res[key] = {"relevance": group[1], "predictions": group[0]}
    return calc_average_metric(res, calc_ndcg_by_query, k)


def test_spearman(y_pred, y_true, groups, k=None):
    res = []
    for key, group in groupby(zip(y_pred, y_true, groups), key=lambda x: x[2]):
        group = [np.array(item).reshape(-1) for item in list(zip(*group))]
        indices = np.argsort(group[1])[::-1]
        if k:
            correlation = stats.spearmanr(group[1][indices][:k], group[0][indices][:k]).correlation
        else:
            correlation = stats.spearmanr(group[1], group[0]).correlation

        res.append(correlation if not np.isnan(correlation) else 0)

    return np.mean(res)


def test_auc(y_pred, y_true, groups):
    res = []
    for key, group in groupby(zip(y_pred, y_true, groups), key=lambda x: x[2]):
        group = list(zip(*group))
        group[1] = (np.array(group[1]) > 1).astype(int)
        if np.unique(group[1]).shape[0] < 2:
            continue
        res.append(roc_auc_score(group[1], group[0]))
    return np.mean(res)


def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def test_precision_for_recall(y_pred, y_true, recall_values=[0.85, 0.91]):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    return [(precision[find_nearest_index(recall, recall_value)], recall_value) for recall_value in recall_values]
