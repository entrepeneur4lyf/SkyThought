import logging
import math

import numpy as np


def pass_at_k(N, temp_to_scores):
    # pass at k per temperature
    # scores = list(correct[temp].values())
    pass_values = {}  # temp -> value
    for temp in temp_to_scores:
        scores = temp_to_scores[temp]  # dict mapping idx -> list of scores
        k = N
        final_passk_scores = {}
        while k > 0:
            new_scores = []
            for _, sample_scores in scores.items():
                # calculate pass @ k
                num_correct = np.sum(sample_scores)
                pass_k = 1 - (math.comb(N - num_correct, k) / math.comb(N, k))
                new_scores.append(pass_k)
            final_passk_scores[f"{k=}"] = round(np.mean(new_scores) * 100, 3)
            k = k // 2

        # print("Final pass @ k:")
        for k, s in final_passk_scores.items():
            logging.info(f"temp: {temp}, k: {k}, pass @ k: {s}")
        pass_values[f"{temp=}"] = final_passk_scores
        # temp_correct = sum([any(x) for x in scores])
    return pass_values
