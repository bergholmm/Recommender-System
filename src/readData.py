import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import seaborn
from sklearn import preprocessing

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

def readItemInfo():
    file = open('../data/tracks_final.csv', 'r')
    ds = []
    missing =[]
    next(file)

    for line in file:
        split = line.split("\t")
        split[5] = split[5].replace("\n","")
        split[5] = split[5].replace("[","")
        split[5] = split[5].replace("]","")
        split[5] = split[5].replace(",","")
        tags = split[5].split()

        if len(tags) == 0:
            missing.append(int(split[0]))
        else:
            for tag in tags:
                ds.append((int(split[0]), int(tag)))

    file.close()

    songWithTags, tags = zip(*ds)
    songWithTags = list(songWithTags)
    tags = list(tags)

    # Preprocess songs
    les = preprocessing.LabelEncoder()
    allSongs = songWithTags + missing
    les.fit(allSongs)
    songWithTags = les.transform(songWithTags)

    # Preprocess tags
    le = preprocessing.LabelEncoder()
    le.fit(tags)
    tags = le.transform(tags)
    indices = np.ones(len(tags))

    print('creating ICM')
    ICM = sps.coo_matrix((indices, (songWithTags, tags))).tocsc()

    print(ICM.shape)
    missingItems = np.zeros((1, len(set(tags))))
    missingItems = sps.csc_matrix(missingItems)
    ICM = sps.vstack((ICM, missingItems))
    print(ICM.shape)

    # return ICM.tocsr()
    # return ICM.tocsc(), les
    return ICM, les


def readTargetPlaylists():
    return np.genfromtxt('../data/target_playlists.csv', delimiter='\t', skip_header=1, dtype=int)


def splitTestTrainDS(ds, les):
    playlists, songs = zip(*ds)
    ps = np.array(playlists)
    s = list(songs)
    s = np.array(les.transform(s))
    rating = np.ones(s.size)

    train_split = 0.8
    num_interactions = len(ds)

    mask = np.random.choice([True, False], num_interactions, p=[train_split, 1-train_split])
    trainds = sps.coo_matrix((rating[mask], (ps[mask], s[mask]))).tocsr()
    mask = np.logical_not(mask)
    testds = sps.coo_matrix((rating[mask], (ps[mask], s[mask]))).tocsr()

    return trainds, testds


def readData():
    ICM, les = readItemInfo()
    ds = readTrainingSet()

    trainingSet, testSet = splitTestTrainDS(ds, les)

    playlists, _ = zip(*ds)
    targets = list(set(playlists))

    return trainingSet, testSet, targets, ICM, les


