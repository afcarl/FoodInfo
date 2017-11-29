import numpy as np
import random
import pandas as pd

def load_data(train):
    cult2id = {}
    id2cult = []
    comp2id = {'Nan':0}
    id2comp = ['Nan']

    train_cult = []
    train_comp = []
    train_comp_len = []

    comp_thr = 5
    max_comp_cnt = 0
    filtred_comp = 0

    train_f = open("../dataset/kaggle_and_nature.csv", 'r')
    lines = train_f.readlines()[4:]
    random.shuffle(lines)
    train_thr = int(len(lines) * 0.8)

    print "Build composer dictionary..."
    for i, line in enumerate(lines):

        tokens = line.strip().split(',')
        culture = tokens[0]
        composers = tokens[1:]

        if cult2id.get(culture) is None:
            cult2id[culture] = len(cult2id)
            id2cult.append(culture)

        if comp_thr > len(composers):
            filtred_comp += 1
            continue

        if max_comp_cnt < len(composers):
            max_comp_cnt = len(composers)

        for composer in composers:
            if comp2id.get(composer) is None:
                comp2id[composer] = len(comp2id)
                id2comp.append(composer)

        train_cult.append(cult2id.get(culture))
        train_comp.append([comp2id.get(composer) for composer in composers])

    for comp in train_comp:
        train_comp_len.append(len(comp))
        if len(comp) < max_comp_cnt:
            comp += [0]*(max_comp_cnt - len(comp))

    print "filtered composer count is", filtred_comp

    return id2cult, id2comp, train_cult[:train_thr], train_comp[:train_thr], train_comp_len[:train_thr], train_cult[train_thr:], train_comp[train_thr:], train_comp_len[train_thr:], max_comp_cnt

def batch_iter(data, batch_size):
    #data = np.array(data)
    data_size = len(data)
    num_batches = int(len(data)/batch_size)
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index:end_index]
