"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/smeetrs/deep_avsr
"""

import torch
import torch.nn as nn
import numpy as np
from itertools import groupby
from data_module.data_module import pad
import warnings
np.seterr(divide="ignore")


def ctc_greedy_decode(outputBatch, inputLenBatch, eosIx, blank=0, visualization=False):

    """
    Greedy search technique for CTC decoding.
    This decoding method selects the most probable character at each time step. This is followed by the usual CTC decoding
    to get the predicted transcription.
    Note: The probability assigned to <EOS> token is added to the probability of the blank token before decoding
    to avoid <EOS> predictions in middle of transcriptions. Once decoded, <EOS> token is appended at last to the
    predictions for uniformity with targets.
    """

    outputBatch = outputBatch.cpu().to(torch.float32)
    inputLenBatch = inputLenBatch.cpu()
    outputBatch[:,:,blank] = torch.log(torch.exp(outputBatch[:,:,blank]) + torch.exp(outputBatch[:,:,eosIx]))
    reqIxs = np.arange(outputBatch.shape[2])
    reqIxs = reqIxs[reqIxs != eosIx]
    outputBatch = outputBatch[:,:,reqIxs]

    # 可视化
    if visualization:
        pass

    predCharIxs = torch.argmax(outputBatch, dim=2).T.numpy()
    inpLens = inputLenBatch.numpy()
    preds = list()
    predLens = list()
    for i in range(len(predCharIxs)):
        pred = predCharIxs[i]
        ilen = inpLens[i]
        pred = pred[:ilen]
        pred = np.array([x[0] for x in groupby(pred)])
        pred = pred[pred != blank]
        pred = list(pred)
        pred.append(eosIx)
        preds.append(torch.tensor(pred))
        predLens.append(len(pred))
    # predictionBatch = torch.tensor(preds).int()
    predictionBatch, _ = pad(preds, -1)
    predictionLenBatch = torch.tensor(predLens).int()
    return predictionBatch, predictionLenBatch


class BeamEntry:
    """
    Class for a single entry in the beam.
    """
    def __init__(self):
        self.logPrTotal = -np.inf
        self.logPrNonBlank = -np.inf
        self.logPrBlank = -np.inf
        self.logPrText = 0
        self.lmApplied = False
        self.lmState = None
        self.labeling = tuple()


class BeamState:

    """
    Class for the beam.
    """

    def __init__(self, alpha, beta, lm_scorer, tranform):
        self.entries = dict()
        self.lm_weight = alpha
        self.length_bonus = beta
        self.lm_scorer = lm_scorer
        self.transform = tranform
        assert 0 <= self.lm_weight <= 1, "wrong lm_weight"
        # if self.transform.mode not in ['word_chinese', 'char_chinese']:
        #     warnings.warn("现在是中文模式，需要根据英文修改")

    def minmax_normalize(self, v):
        v = np.array(v)
        if len(v[v != -np.inf]) > 0:
            min_value = min(v[v != -np.inf])
            max_value = max(v[v != np.inf])
            v = (v - min_value) / max(max_value - min_value, 1e-9)
       
        return v

    def final_score(self, beams):
        """
        final score will balance lm_score and beam_score
        """
        beam_scores = [entry.logPrTotal for entry in beams]
        
        texts = [self.transform.token2word(entry.labeling) for entry in beams]
        lm_scores = self.lm_scorer(texts) if self.lm_scorer is not None else np.zeros_like(texts, dtype=np.float32)
        
        normalised_beam_scores = self.minmax_normalize(beam_scores)
        
        normalised_lm_scores = self.minmax_normalize(lm_scores)
        
        # print(texts)
        # print(normalised_beam_scores)
        # print(normalised_lm_scores)
        # print()
        # print()

        scores = (1 - self.lm_weight) * normalised_beam_scores + self.lm_weight * normalised_lm_scores
    
        return scores

    def score(self, beams):
        return [entry.logPrTotal for entry in beams]

    def sort(self, final=False):
        """
        Function to sort all the beam entries in descending order depending on their scores.
        """
        beams = [entry for (key, entry) in self.entries.items()]
        if not final:
            scores = self.score(beams)
        else:
            scores = self.final_score(beams)
        
        beams_with_score = [(entry, score) for entry, score in zip(beams, scores)]

        beams_with_score.sort(reverse=True, key=lambda x: x[-1])

        return [entry.labeling for (entry, score) in beams_with_score]
    
    def sort_with_flip(self, flip_beams):
        beams = [entry for (key, entry) in self.entries.items()] + [entry for (key, entry) in flip_beams.entries.items()]
        scores = self.final_score(beams)
        beams_with_score = [(entry, score) for entry, score in zip(beams, scores)]
        beams_with_score.sort(reverse=True, key=lambda x: x[-1])
        return [entry.labeling for (entry, score) in beams_with_score]


def add_beam(beamState, labeling):
    """
    Function to add a new entry to the beam.
    """
    if labeling not in beamState.entries.keys():
        beamState.entries[labeling] = BeamEntry()


def log_add(a, b):
    """
    Addition of log probabilities.
    """
    result = np.log(np.exp(a) + np.exp(b))
    return result


def find_best_n_outputs(outputBatch, inputLenBatch, beamSearchParams, 
                      spaceIx, eosIx, transform, lm_scorer, blank=0):
    outputBatch = outputBatch.cpu().to(torch.float32)
    inputLenBatch = inputLenBatch.cpu()
    outputBatch[:,:,blank] = torch.log(torch.exp(outputBatch[:,:,blank]) + torch.exp(outputBatch[:,:,eosIx]))
    reqIxs = np.arange(outputBatch.shape[2])
    reqIxs = reqIxs[reqIxs != eosIx]
    outputBatch = outputBatch[:,:,reqIxs]

    beamWidth = beamSearchParams["beamWidth"]
    alpha = beamSearchParams["alpha"]
    beta = beamSearchParams["beta"]
    threshProb = beamSearchParams["threshProb"]

    outLogProbs = outputBatch.transpose(0, 1).numpy()  # (B, T, odim)
    inpLens = inputLenBatch.numpy()
    
    best_entries = []

    for n in range(len(outLogProbs)):
        mat = outLogProbs[n]
        ilen = inpLens[n]
        mat = mat[:ilen,:]
        maxT, maxC = mat.shape

        #initializing the main beam with a single entry having empty prediction
        last = BeamState(alpha, beta, lm_scorer, transform)
        labeling = tuple()
        last.entries[labeling] = BeamEntry()
        last.entries[labeling].logPrBlank = 0
        last.entries[labeling].logPrTotal = 0

        #going over all the time steps
        for t in range(maxT):

            #a temporary beam to store all possible predictions (which are extensions of predictions
            #in the main beam after time step t-1) after time step t
            curr = BeamState(alpha, beta, lm_scorer, transform)
            #considering only the characters with probability above a certain threshold to speeden up the algo
            prunedChars = np.where(mat[t,:] > np.log(threshProb))[0]
            # print(len(np.where(mat[t,:] > np.log(threshProb))[0]))

            # keeping only the best predictions in the main beam
            bestLabelings = last.sort()[:beamWidth]

            #going over all the best predictions
            for labeling in bestLabelings:

                #same prediction (either blank or last character repeated)
                if len(labeling) != 0:
                    logPrNonBlank = last.entries[labeling].logPrNonBlank + mat[t, labeling[-1]]
                else:
                    logPrNonBlank = -np.inf

                logPrBlank = last.entries[labeling].logPrTotal + mat[t, blank]

                add_beam(curr, labeling)
                curr.entries[labeling].labeling = labeling
                curr.entries[labeling].logPrNonBlank = log_add(curr.entries[labeling].logPrNonBlank, logPrNonBlank)
                curr.entries[labeling].logPrBlank = log_add(curr.entries[labeling].logPrBlank, logPrBlank)
                curr.entries[labeling].logPrTotal = log_add(curr.entries[labeling].logPrTotal, log_add(logPrBlank, logPrNonBlank))
                curr.entries[labeling].logPrText = last.entries[labeling].logPrText
                curr.entries[labeling].lmApplied = True
                curr.entries[labeling].lmState = last.entries[labeling].lmState

                #extending the best prediction with all characters in the pruned set
                for c in prunedChars:

                    if c == blank:
                        continue

                    #extended prediction
                    newLabeling = labeling + (c,)

                    if (len(labeling) != 0)  and (labeling[-1] == c):
                        logPrNonBlank = mat[t, c] + last.entries[labeling].logPrBlank
                    else:
                        logPrNonBlank = mat[t, c] + last.entries[labeling].logPrTotal

                    add_beam(curr, newLabeling)
                    curr.entries[newLabeling].labeling = newLabeling
                    curr.entries[newLabeling].logPrNonBlank = log_add(curr.entries[newLabeling].logPrNonBlank, logPrNonBlank)
                    curr.entries[newLabeling].logPrTotal = log_add(curr.entries[newLabeling].logPrTotal, logPrNonBlank)

                    #applying language model
                    # if lm is not None:
                    #     apply_lm(curr.entries[labeling], curr.entries[newLabeling], spaceIx, lm)

            #replacing the main beam with the temporary beam having extended predictions
            last = curr

        best_entries.append(last)

    return best_entries
 

def ctc_search_decode(outputBatch, flip_outputBatch, inputLenBatch, beamSearchParams, 
                      spaceIx, eosIx, transform, lm_scorer, blank=0):

    best_entries = find_best_n_outputs(
        outputBatch, inputLenBatch, beamSearchParams, spaceIx, eosIx, transform, lm_scorer, blank=0
    )
    
    if flip_outputBatch is not None:
        flip_best_entries = find_best_n_outputs(
            flip_outputBatch, inputLenBatch, beamSearchParams, spaceIx, eosIx, transform, lm_scorer, blank=0
        )
    
    preds, predLens = [], []
    for i in range(len(best_entries)):
        last = best_entries[i]
        #output the best prediciton
        if flip_outputBatch is not None:
            bestLabeling = last.sort_with_flip(flip_best_entries[i])[0]
        else:
            bestLabeling = last.sort(final=True)[0]
        # print(transform.token2word(bestLabeling), '\n\n\n')
        bestLabeling = list(bestLabeling)
        bestLabeling.append(eosIx)
        preds.append(torch.tensor(bestLabeling))
        predLens.append(len(bestLabeling))

    predictionBatch, _ = pad(preds, -1)
    predictionLenBatch = torch.tensor(predLens).int()
    return predictionBatch, predictionLenBatch
