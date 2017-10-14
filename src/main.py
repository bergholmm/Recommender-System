import numpy as np
import scipy as sp
import scipy.sparse as sps
import matplotlib.pyplot as plt
import seaborn
import sys
import time

import readData as data
import recommenders
import evalRecommender as e


def predict(targets, recommender, les):
    recommendations = []
    for t in targets:
        rec = recommender.recommend(t)
        rec = les.inverse_transform(rec)
        recommendations.append(' '.join(str(p) for p in rec))

    result = list(zip(targets, recommendations))
    t = int(time.time())
    np.savetxt('../result/'+ t +'.csv', result, fmt="%s", delimiter=',', header='playlist_id,track_ids')
    return recommendations


def evalRecommender(recommender, ds, targets):
    e.evaluate(ds, targets, recommender)


def recommend(recommender, les):
    t = data.readTargetPlaylists()
    predict(t, recommender, les)


def main():
    print('Read dataset')
    trainds, testds, targets, ICM, les = data.readData()
    # recommender = recommenders.BasicItemKNNRecommender(trainds)
    # print('Fit model')
    # recommender.fit(ICM)

    # recommender = recommenders.TopRecommender()
    # print('Fit model')
    # recommender.fit(trainds)

    # if sys.argv[1] == 'True':
    print('Evaluate recommender')
    # evalRecommender(recommender, testds, targets)
    # # else:
    # print('Recommend')
    # recommend(recommender, les)
    print('Done')


if __name__ == "__main__":
    main()
