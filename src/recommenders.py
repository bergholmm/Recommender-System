import numpy as np
import scipy.sparse as sps
from Similarity import Cosine #, Pearson, AdjustedCosine


class TopRecommender(object):
    def fit(self, ds):
        self.ds = ds
        ds = check_matrix(ds, 'csc', dtype=np.float32)
        itemPopularity = (ds > 0).sum(axis = 0)
        itemPopularity = np.asarray(itemPopularity).squeeze()
        self.model = np.argsort(itemPopularity)[::-1]

    def recommend(self, playlist_id, at = 5, remove_seen = False):
        if remove_seen:
            r = filter_seen(self.ds, playlist_id, self.model)
            return r[0:at]
        else:
            return self.model[0:at]


# ----------------------------------------------------------------


class BasicItemKNNRecommender(object):
    def __init__(self, URM, k=50, shrinkage=0, similarity='cosine'):
        self.dataset = URM
        self.k = k
        self.shrinkage = shrinkage
        self.similarity_name = similarity
        if similarity == 'cosine':
            self.distance = Cosine(shrinkage=self.shrinkage)
        # elif similarity == 'pearson':
        #     self.distance = Pearson(shrinkage=self.shrinkage)
        # elif similarity == 'adj-cosine':
        #     self.distance = AdjustedCosine(shrinkage=self.shrinkage)
        else:
            raise NotImplementedError('Distance {} not implemented'.format(similarity))

    def __str__(self):
        return "ItemKNN(similarity={},k={},shrinkage={})".format(
            self.similarity_name, self.k, self.shrinkage)

    def fit(self, X):
        item_weights = self.distance.compute(X)
        print('dist done')
        item_weights = check_matrix(item_weights, 'csr') # nearly 10 times faster
        print("Converted to csr")

        values, rows, cols = [], [], []
        nitems = self.dataset.shape[1]
        for i in range(nitems):
            if (i % 1000 == 0):
                print("Item %d of %d" % (i, nitems))

            this_item_weights = item_weights[i,:].toarray()[0]
            top_k_idx = np.argsort(this_item_weights) [-self.k:]

            values.extend(this_item_weights[top_k_idx])
            rows.extend(np.arange(nitems)[top_k_idx])
            cols.extend(np.ones(self.k) * i)
        self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(nitems, nitems), dtype=np.float32)

    def recommend(self, user_id, at=5, exclude_seen=True):
        user_profile = self.dataset[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()

        ranking = scores.argsort()[::-1]
        if exclude_seen:
            ranking = filter_seen(self.dataset, user_id, ranking)

        return ranking[:at]


# -------------------------------------------------------------

def filter_seen(dataset, user_id, ranking):
    user_profile = self.dataset[user_id]
    seen = user_profile.indices
    unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
    return ranking[unseen_mask]

def check_matrix(X, format='csc', dtype=np.float32):
    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)
    else:
        return X.astype(dtype)
