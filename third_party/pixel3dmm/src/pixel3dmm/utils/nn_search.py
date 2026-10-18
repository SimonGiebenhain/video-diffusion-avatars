import numpy as np
import torch
import faiss.contrib.torch_utils
from time import time

res = faiss.StandardGpuResources()  # use a single GPU


def knn_faiss(queries, points):
    t0 = time()
    # assert that batch dimensions is 1, since faiss might not support batch of indices
    assert points.shape[0] == 1
    assert points.shape[-1] == 2
    assert queries.shape[0] == 1
    assert queries.shape[-1] == 2

    points = points[0].contiguous()
    queries = queries[0].contiguous()

    nlist = 1000
    k = 1
    d = 2
    quantizer = faiss.IndexFlatL2(d)  # the other index
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    index = faiss.index_cpu_to_gpu(res, 0, index)

    index.nprobe = 10  # default nprobe is 1, try a few more

    index.train(points)
    index.add(points)  # add may be a bit slower as well

    D, I = index.search(queries, k)

    print(f'Faiss KNN took {time()-t0} seconds')
    return D.unsqueeze(0), I.unsqueeze(0) # add back batch dimension

