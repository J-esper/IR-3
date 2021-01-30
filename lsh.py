# http://ekzhu.com/datasketch/documentation.html#datasketch.MinHashLSH
# http://ekzhu.com/datasketch/lsh.html

import pandas as pd
import numpy as np
import itertools
import re
from datasketch import MinHash, MinHashLSH
file = "news_articles_large.csv"

data = pd.read_csv(file)

def get_shingle(k, a):
    for i in range(0,len(a)-2+1):
        yield '-'.join(tuple(a[i:i+2]))

# Different shingle lengths
# n = 5
# data['article'] = data['article'].map(lambda x: [i for i in get_shingle(n, re.sub('[^A-Za-z0-9 ]+', '', x.lower()).split(' '))])
# Length 1
data['article'] = data['article'].map(lambda x: re.sub('[^A-Za-z0-9 ]+', '', x.lower()).split(' '))


mh_list = []
num_perm = 256
for i in range(0, len(data)):
    mh = MinHash(num_perm=num_perm)
    for d in data.iloc[i]['article']:
        mh.update(d.encode('utf8'))
    mh_list.append(mh)

# Default, use optimizer in constructor
threshold = 0.8
lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
# Params adjusts the number of bands and size of each band (has to be smaller than num_perms).
# Using this will ignore threshold.
# lsh = MinHashLSH(threshold=threshold, num_perm=num_perm, params=(1,8))

for i in range(0, len(mh_list)):
    lsh.insert('doc_'+str(i), mh_list[i])

df_results = pd.DataFrame(columns = ['query', 'duplicates'])
count = 0
for i in range(0, len(mh_list)):
    results = lsh.query(mh_list[i])
    if len(results) > 1:
        # print("Similarity > ", threshold, "for query doc", i, ": ", results)
        count += len(results) - 1
        temp = pd.DataFrame({'query': 'doc_'+str(i), 'duplicates': [results]})
        df_results = df_results.append(temp, ignore_index=True)

print(df_results)
df_results.to_csv("results.csv")
print("# duplicates found for all queries with (b,r) = ({},{}): {}".format(lsh.b, lsh.r, count))