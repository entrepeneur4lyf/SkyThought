import logging
import math
from collections import defaultdict
import numpy as np


def pass_at_k(N, temp_to_scores):
    # pass at k per temperature
    # scores = list(correct[temp].values())
    pass_values = {}  # temp -> value
    for temp in temp_to_scores:
        scores = temp_to_scores[temp]  # dict mapping idx -> list of scores
        final_passk_scores = {}
        k_to_passk_scores = defaultdict(list) # k -> list of scores
        for _, sample_scores in scores.items():
            k = N
            while k > 0:
                # calculate pass @ k
                num_correct = np.sum(sample_scores)
                pass_k = 1 - (math.comb(N - num_correct, k) / math.comb(N, k))
                k_to_passk_scores[k].append(pass_k)
                k = k // 2
        
        for k in k_to_passk_scores:
            final_passk_scores[f"{k=}"] = round(np.mean(k_to_passk_scores[k]) * 100, 3)

        # print("Final pass @ k:")
        for k, s in final_passk_scores.items():
            logging.info(f"temp: {temp}, k: {k}, pass @ k: {s}")
        pass_values[f"{temp=}"] = final_passk_scores
        # temp_correct = sum([any(x) for x in scores])
    return pass_values
