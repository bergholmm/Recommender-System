import numpy as np
import scipy as sp
import scipy.sparse as sps
import matplotlib.pyplot as plt
import seaborn
import sys

import readData as data
import recommenders
import evalRecommender as e

def predict(targets, recommender):
    recommendations = []
    for t in targets:
        rec = recommender.recommend(t)
        recommendations.append(' '.join(str(p) for p in rec))

    result = list(zip(targets, recommendations))
    np.savetxt('../result/submission.csv', result, fmt="%s", delimiter=',', header='playlist_id,track_ids')
    return recommendations


def saveModel(model):
    print(model)


def evalRecommender(recommender, ds, targets):
    print('Evaluate model')
    e.evaluate(ds, targets, recommender)


def recommend(recommender):
    t = data.readTargetPlaylists()
    print('Predict')
    predict(t, recommender)
    print('Done')


def main():
    print('Read dataset')
    trainds, testds, targets = data.readData()
    recommender = recommenders.TopRecommender()
    print('Fit model')
    recommender.fit(trainds)

    if sys.argv[1] == 'True':
        print('Evaluate recommender')
        evalRecommender(recommender, testds, targets)
    else:
        print('Recommend')
        recommend(recommender)


if __name__ == "__main__":
    main()
