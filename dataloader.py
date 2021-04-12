import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import nltk
import pickle
import keras

def save_pickle(Object, filename):
    with open(filename, "wb") as handler:
        pickle.dump(Object, handler)

def read_csv(filename):
    return pd.read_csv(filename, names=["Tweet", "Affect Dimension"]).dropna()

def get_dictionary(data):
    tokenized_word = nltk.word_tokenize("  ".join(list(map(str, data["Tweet"].values))))
    vocabulary =  set(tokenized_word)
    word_to_index = {}
    word_to_index["UNK"] = 1
    for key,value in enumerate(vocabulary,2):
        word_to_index[value] = key
    index_to_word = {value:key for key,value in word_to_index.items()}
    return word_to_index, index_to_word
class data_loader(torch.utils.data.Dataset):
    def __init__(self, dataX, dataY, index_to_word, word_to_index, train=True):
        self.train = train
        self.dataX = dataX
        self.dataY = dataY
        self.word_to_index =word_to_index
        self.index_to_word = index_to_word
        self.pad_size = 12

    def __texttonumb__(self, x_val):
        splitted_val  = nltk.word_tokenize(str(x_val))
        xArray = []
        for val in splitted_val:
            if self.word_to_index.get(val) != None:
                xArray.append(self.word_to_index[val])
            else:
                xArray.append(self.word_to_index["UNK"])
        paddedXArray = keras.preprocessing.sequence.pad_sequences([xArray], maxlen=self.pad_size, padding='post')
        return paddedXArray

    def __getitem__(self, idx):
        X_val = self.dataX.iloc[idx]
        Y_val = int(self.dataY.iloc[idx])
        X = self.__texttonumb__(X_val)
        return torch.tensor(X), torch.tensor(Y_val)
    def __len__(self):
        return len(self.dataX)

def get_data_loader(filename, batch_size, train=True):
    word_to_index= None
    index_to_word= None
    data =None
    if train:
        data = read_csv(filename)
        # let's divide the data
        word_to_index, index_to_word = get_dictionary(data)
        save_pickle(word_to_index,"word_to_index.pickle")
        save_pickle(index_to_word, "index_to_word.pickle")
    else:
        word_to_index = pickle.load(open("word_to_index.pickle","rb"))
        index_to_word = pickle.load(open("index_to_word.pickle","rb"))
        data = read_csv(filename)


    dataLoad = data_loader(data["Tweet"],data["Affect Dimension"], index_to_word, word_to_index)
    indexs = np.arange(1,len(data["Tweet"]))
    subsetSampler = torch.utils.data.SubsetRandomSampler(indexs)
    trainDataLoader = torch.utils.data.DataLoader(dataLoad, batch_sampler=torch.utils.data.BatchSampler(subsetSampler,batch_size=batch_size,drop_last=True))
    return trainDataLoader

