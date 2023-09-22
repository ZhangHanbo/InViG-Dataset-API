import numpy as np
import os
from nlgeval import compute_metrics
import re


def eval_topk_rank(
        results,
        annotations,
):
    """
    this function is used to evaluate the topk-ranking performance for text generation.

    results List[List[int]]:
        Given N samples in the validation set, and M candidates for each text, the outer List
        should be with length of N and inter List should be with length of M.
        The inter List contains the ranking of each candidate from the model, ranging from 1 to M.
    annotations List[int]:
        Given N samples, the annotations should be with length of N. Each element represents the
        index of the ground truth in the corresponding candidate list.

    Example:
        For a validation dataset containing 2 samples, with 3 candidates for each sample:
        results:
            [
                [1, 3, 2],
                [3, 2, 1]
            ]
        annotations:
            [0, 1]

        In this case ,the model gives ranking results of [1, 3, 2] for the first sample, and [3, 2, 1]
        for the second sample.
        The annotations indicate that the 0th candidate is the ground truth for the first sample, and 1th
        candidate is the ground truth for the second sample.

        Therefore, the ranking of ground truth from the model for the first sample should be
            results[0][annotations[0]] = 1
        and the second sample:
            results[1][annotations[1]] = 2

        The mean ranking is: 1.5, and the top-1 recall is 50%

    """
    gt_ranks = []
    for res, ann in zip(results, annotations):
        gt_ranks.append(res[ann])
    gt_ranks = np.array(gt_ranks)

    metrics = {}
    metrics['top_1_recall'] = (gt_ranks == 1).sum() / gt_ranks.size
    metrics['top_5_recall'] = (gt_ranks <= 5).sum() / gt_ranks.size
    metrics['mean_rank'] = gt_ranks.mean()
    return metrics

def eval_text_similarity(
        results,
        annotations,
        cache_dir='.',
):
    """
    this function is used to evaluate the text-similarity metrics for text generation.

    results List[str]:
        A list of predictions
    annotations List[str]:
        A list of ground truth texts

    results and annotations should share the same length.
    """
    result_cache = os.path.join(cache_dir, 'pred.txt')
    gt_cache = os.path.join(cache_dir, 'gt.txt')
    result_file = open(result_cache, 'w')
    gt_file = open(gt_cache, 'w')

    for res, ann in zip(results, annotations):
        a_pred = res.lower().strip()
        a_gt = ann.lower().strip()
        result_file.write(a_pred + '\n')
        gt_file.write(a_gt + '\n')

    result_file.close()
    gt_file.close()

    return compute_metrics(hypothesis=result_cache, references=[gt_cache], no_skipthoughts=True, no_glove=True)

def iou(box_a, box_b):
    A = box_a.shape[0]
    B = box_b.shape[0]
    max_xy = np.minimum(
        box_a[:, 2:][:, None, :].repeat(B, axis=1),
        box_b[:, 2:][None, :, :].repeat(A, axis=0)
    )
    min_xy = np.maximum(
        box_a[:, :2][:, None, :].repeat(B, axis=1),
        box_b[:, :2][None, :, :].repeat(A, axis=0)
    )
    inter_size = np.maximum((max_xy - min_xy), 0)
    inter = inter_size[:, :, 0] * inter_size[:, :, 1]

    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))[:, None]  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1]))[None, :]
    union = area_a + area_b - inter

    return inter / union

def eval_grounding_acc(
        grounding_results,
        grounding_annotations,
):
    """
    grounding_results List[List[float]]:
        a list of bounding boxes, each is 4-d
    grounding_annotations List[List[float]]:
        a list of bounding boxes, each is 4-d
    """
    iou_thresh = np.arange(0.1, 1.0, 0.1)
    iou_thresh = np.append(iou_thresh, 0.99)
    total = 0
    tp = np.zeros(iou_thresh.shape)
    for res, ann in zip(grounding_results, grounding_annotations):
        pred_bbox = np.array(res)[None, :]
        gt_bbox = np.array(ann)
        _iou = iou(pred_bbox, gt_bbox)
        total += 1
        tp += (iou_thresh < _iou.max())
    results = dict(zip(
        [f'grounding_acc_iou_{float(i) / 10}' for i in range(1, 10)],
        np.round(tp[:-1] / total, 4)
    ))
    results['grounding_acc_iou_0.99'] = np.round(tp[-1] / total, 4)
    return results