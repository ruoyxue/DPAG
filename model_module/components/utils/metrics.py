import torchaudio


def compute_feat_cer(pred, gt, ignore_id):
    """ pred(B, dim1) gt(B, dim2) """
    total_dist, total_len = 0, 0
    for i in range(pred.shape[0]):
        pred_single = pred[i]
        gt_single = gt[i]
        pred_single = pred_single[pred_single!=ignore_id]
        gt_single = gt_single[gt_single!=ignore_id]
        total_dist += torchaudio.functional.edit_distance(
            pred_single, 
            gt_single
        )
        total_len += sum(gt_single!=ignore_id)
    return total_dist / total_len
