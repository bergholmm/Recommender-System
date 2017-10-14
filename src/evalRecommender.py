import numpy as np

def precision(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score


def recall(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]

    return recall_score


def MAP(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score


def evaluate(ds, targets, recommender):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0

    num_eval = 0

    numTargets = len(targets)

    for t in targets:
        if num_eval % 1000 == 0:
            print('targets done: ', num_eval, numTargets)

        rel_songs = ds[t].indices

        if len(rel_songs) > 0:
            rec_songs = recommender.recommend(t)
            num_eval += 1
            cumulative_precision += precision(rec_songs, rel_songs)
            cumulative_recall += recall(rec_songs, rel_songs)
            cumulative_MAP += MAP(rec_songs, rel_songs)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval

    print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
    cumulative_precision, cumulative_recall, cumulative_MAP))
