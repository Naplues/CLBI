# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as stats

if __name__ == '__main__':
    #
    np.random.seed(1)
    x = [round(1 + np.random.random(), 3) for i in range(10)]
    y = [round(3 + np.random.random(), 3) for i in range(10)]

    print(x)
    print(y)
    test = stats.wilcoxon(x, y, correction=True)

    print(test)

    pass
