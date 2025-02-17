import numpy as np
from model_module.components.utils.beam_search import Hypothesis


def minmax_normalize(v):
    v = np.array(v)
    if len(v[v != -np.inf]) > 0:
        min_value = min(v[v != -np.inf])
        max_value = max(v[v != np.inf])
        v = (v - min_value) / max(max_value - min_value, 1e-9)
    return v


def apply_lm(nbest_hyps, scorer, lm_weight, length_bonus, transform):
    """ Apply language model to the Hypothesis and add the score to their original scores. """
    texts = [transform.token2word(hyps.yseq) for hyps in nbest_hyps]
    model_scores = [hyps.score for hyps in nbest_hyps]
    lm_scores = scorer(texts)

    new_hyps = []
    for i in range(len(nbest_hyps)):
        new_score = nbest_hyps[i].score + lm_weight * lm_scores[i] + length_bonus * len(nbest_hyps[i].yseq)
        new_hyps.append(
            Hypothesis(
                yseq=nbest_hyps[i].yseq,
                score=new_score
            )
        )

    return new_hyps
