# QBETS: queue bounds estimation from time series
# http://www.cs.huji.ac.il/~feit/parsched/jsspp07/p4-nurmi.pdf
import numpy as np

from sim import *


def comb(n, r):
    return np.math.factorial(n) / (np.math.factorial(r) * np.math.factorial(n - r))


class QuantileModel:
    def __init__(self, quantile, confidence):
        self.quantile = quantile
        self.confidence_bound = confidence

    @DeprecationWarning
    def predict_old(self, retired_jobs: [job.JobLog]) -> float:
        sorted_jobs = list(sorted(retired_jobs, key=lambda x: (x.start - x.job.submit).total_seconds()))
        index = 0
        for k in range(2, len(retired_jobs) + 2):
            prob_accu = 0
            for j in range(0, k):
                prob_accu += comb(len(retired_jobs), j) * ((1 - self.quantile) ** j) * (
                        self.quantile ** (len(retired_jobs) - j))
            prob_accu = 1 - prob_accu
            if prob_accu >= self.confidence_bound:
                index = k - 2
                break
        return (sorted_jobs[index].start - sorted_jobs[index].job.submit).total_seconds() / 3600.0

    def predict(self, retired_jobs: [job.JobLog]) -> float:
        queue_wait_time = list(map(lambda x: (x.start - x.job.submit).total_seconds(), retired_jobs))
        return np.percentile(queue_wait_time, self.quantile * 100) / 3600.0
