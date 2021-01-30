import pandas as pd
import numpy as np
import itertools
import re
from matplotlib import pyplot as plt

file = "news_articles_small.csv"

data = pd.read_csv(file)

def get_shingle(k, a):
    for i in range(0,len(a)-2+1):
        yield '-'.join(tuple(a[i:i+2]))

def jaccard_similarity(a, b):
    intersection = len(list(set(a) & set(b)))
    union = (len(a) + len(b)) - intersection
    return float(intersection) / float(union)

# Create n-shingles for every article
n = 2
data['article'] = data['article'].map(lambda x: [i for i in get_shingle(n, re.sub('[^A-Za-z0-9 ]+', '', x.lower()).split(' '))])

pairs = filter(lambda x: x[0] != x[1], itertools.product(data['article'], data['article']))
# pairs = itertools.product(data['article'], data['article'])
sim = []
for a, b in pairs:
    sim.append(jaccard_similarity(a, b))

bins = np.arange(0, 1, 0.005)

plt.xlim([0, 1])

plt.hist(sim, bins=bins, alpha=0.5)
plt.title('Histogram of Jaccard Similarities')
plt.xlabel('Jaccard Similarity')
plt.ylabel('count')

plt.show()