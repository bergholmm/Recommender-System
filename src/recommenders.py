import numpy as np

class TopRecommender(object):
    i = 0
    def fit(self, ds):
        self.ds = ds

        itemPopularity = (ds > 0).sum(axis = 0)
        itemPopularity = np.array(itemPopularity).squeeze()

        self.model = np.flip(np.argsort(itemPopularity), axis = 0)


    def recommend(self, playlist_id, at = 5, remove_seen = False):
        self.i += 1
        if self.i % 100 == 0:
            print(self.i)
        if remove_seen:
            mask = np.in1d(self.model, self.ds[playlist_id].indices, assume_unique=True, invert=True)
            unseen = self.model[mask]
            return unseen[0:at]
        else:
            return self.model[0:at]
