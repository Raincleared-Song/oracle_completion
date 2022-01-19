import json
import torch
import numpy as np
from sklearn import metrics


def eval_multi_label(predict_out, labels, label_mask, na_id, eval_res: dict = None):
    if eval_res is None:
        eval_res = {
            'correct_num': 0,  # correct count
            'predict_num': 0,  # predict not NA count
            'instance_num': 0,  # instance count
            'auc_item': []  # [(is_correct, score)]
        }
    assert len(labels[0]) == len(predict_out[0])
    score_list, predict_label = torch.max(predict_out, dim=1)
    for i in range(len(predict_label)):
        if int(label_mask[i]) == 0:  # null label
            continue
        if int(labels[i][na_id]) == 0:  # answer is not NA
            eval_res['instance_num'] += int(labels[i].sum())
        pre_lab = int(predict_label[i])
        if pre_lab == na_id:  # predict to be NA
            continue
        eval_res['predict_num'] += 1
        predict_right = int(labels[i][pre_lab]) == 1
        eval_res['auc_item'].append((int(predict_right), float(score_list[i])))
        if predict_right:
            eval_res['correct_num'] += 1
    return eval_res


def eval_softmax(predict_out, labels, eval_res):
    if eval_res is None:
        eval_res = {'correct_num': 0, 'instance_num': 0}
    predict_out = torch.max(predict_out, dim=1)[1]
    correct = int(torch.sum(torch.eq(predict_out, labels)))
    eval_res['correct_num'] += correct
    eval_res['instance_num'] += int(predict_out.shape[0]) - int(torch.sum(labels == -100))
    return eval_res


def f1_auc_metric(eval_res: dict, mode: str):
    """
    :param eval_res: dict{'correct_num': 0,  # correct count
                          'predict_num': 0,  # predict not NA count
                          'instance_num': 0,  # instance count
                          'auc_item': []}
    :param mode: train/valid/test
    :return: evaluate result string
    """
    if eval_res['predict_num'] != 0 and eval_res['instance_num'] != 0:
        precision = eval_res['correct_num'] / eval_res['predict_num']
        recall = eval_res['correct_num'] / eval_res['instance_num']
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
    else:
        precision = 0
        recall = 0
        f1 = 0
    result = {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
    }
    if mode != 'train' and eval_res['predict_num'] != 0 and eval_res['instance_num'] != 0:
        eval_res['auc_item'].sort(key=lambda x: x[1], reverse=True)
        auc_x = []
        auc_y = []
        correct = 0
        for i, item in enumerate(eval_res['auc_item']):
            correct += item[0]
            auc_y.append(float(correct) / (i + 1))
            auc_x.append(float(correct) / eval_res['predict_num'])
        auc_x = np.asarray(auc_x, dtype='float32')
        auc_y = np.asarray(auc_y, dtype='float32')
        try:
            if len(auc_x) > 1:
                auc = metrics.auc(x=auc_x, y=auc_y)
                result['auc'] = round(auc, 4)
            else:
                result['auc'] = 0.0
        except ValueError as err:
            print(auc_x.shape, auc_y.shape)
            del eval_res['auc_item']
            print(eval_res)
            raise err
    ret = {
        'result': result,
        'stat': {
            'correct_num': eval_res['correct_num'],
            'predict_num': eval_res['predict_num'],
            'instance_num': eval_res['instance_num']
        }
    }
    return json.dumps(ret, sort_keys=True)


def binary_metric(eval_res: dict, _):
    ret = {}
    for key in eval_res.keys():
        ret[key] = 0
        if eval_res[key]['instance_num'] != 0:
            acc = round(eval_res[key]['correct_num'] / eval_res[key]['instance_num'], 4)
            ret[key] = (acc, eval_res[key]['instance_num'], eval_res[key]['correct_num'])
    return json.dumps(ret)


def eval_loss(loss, eval_res: dict = None):
    if eval_res is None:
        eval_res = {
            'total_loss': 0,
            'total_batch': 0
        }
    eval_res['total_loss'] += float(loss.mean())
    eval_res['total_batch'] += 1
    return eval_res


def loss_metric(eval_res: dict, _):
    ret = eval_res.copy()
    ret['batch_avg_loss'] = ret['total_loss'] / ret['total_batch']
    return json.dumps(ret)
