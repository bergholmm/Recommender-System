import numpy as np
import scipy.sparse as sps

def readTrainingSet():
    file = open('../data/train_final.csv', 'r')
    ds = []
    next(file)

    for line in file:
        split = line.split("\t")
        split[1] = split[1].replace("\n","")
        split[0] = int(split[0])
        split[1] = int(split[1])
        ds.append(tuple(split))

    file.close()
    return ds


def readTargetPlaylists():
    return np.genfromtxt('../data/target_playlists.csv', delimiter='\t', skip_header=1, dtype=int)


def readData():
    ds = readTrainingSet()
    playlists, songs = zip(*ds)
    targets = list(set(playlists))

    ps = np.array(playlists)
    s = np.array(songs)
    rating = np.ones(s.size)


    train_split = 0.8
    num_interactions = len(ds)

    mask = np.random.choice([True, False], num_interactions, p=[train_split, 1-train_split])
    trainds = sps.coo_matrix((rating[mask], (ps[mask], s[mask]))).tocsr()

    mask = np.logical_not(mask)
    testds = sps.coo_matrix((rating[mask], (ps[mask], s[mask]))).tocsr()

    return trainds, testds, targets


