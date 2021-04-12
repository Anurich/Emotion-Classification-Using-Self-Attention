import os
import numpy as np
import pandas as pd
import texthero as hero
import pickle
import re
def save_pickle_file(Object, file ):
    #save dictionary as pickle
    with open(file, "wb") as handler:
        pickle.dump(Object,handler)

def preprocess_data(folder, name, train=True):
    dataframe = pd.DataFrame(columns=("Tweet","Affect Dimension"))
    list_of_folders = os.listdir(folder)
    map_emotion={}
    intensityClass = {}
    value = 0
    if train:
        for file in list_of_folders:
            print(file)
            with open(os.path.join(folder, file),'r') as fp:
                lines = fp.readlines()[1:]
                for line in lines:
                    _, tweet, affDim, IC = line.split("\t")
                    class_id, text= IC.split(":")
                    map_emotion[value] = affDim
                    tweet = re.sub(r"@\s+","",tweet)
                    tweet = re.sub(r"#\s+","",tweet)
                    #tweet = re.sub(r"\.+", "",tweet)
                    tweet = hero.remove_stopwords(pd.Series(tweet)).to_string(index=False)
                    tweet = re.sub(r'[^\w\s]','', tweet)
                    tweet = re.sub(r"\w+\d+","",tweet)
                    tweet = re.sub(r"\d+\w+","",tweet)
                    tweet = re.sub(r"\b\w{1,3}\b","",tweet)

                    #print(tweet)
                    dataframe = dataframe.append([{"Tweet":tweet.lower().strip(),"Affect Dimension":value}], ignore_index =False)
                    intensityClass[text] = class_id
                value+=1
    # save to csv file
    else:
        for file in list_of_folders:
            with open(os.path.join(folder, file),'r') as fp:
                lines = fp.readlines()[1:]
                for line in lines:
                    _, tweet, affDim, IS = line.split("\t")
                    map_emotion[value] = affDim
                    tweet = re.sub(r"@\s+","",tweet)
                    tweet = re.sub(r"#\s+","",tweet)
                    #tweet = re.sub(r"\.+", "",tweet)
                    tweet = hero.remove_stopwords(pd.Series(tweet)).to_string(index=False)
                    tweet = re.sub(r'[^\w\s]','', tweet)
                    tweet = re.sub(r"\w+\d+","",tweet)
                    tweet = re.sub(r"\d+\w+","",tweet)
                    tweet = re.sub(r"\b\w{1,3}\b","",tweet)
                    #print(tweet)
                    dataframe = dataframe.append([{"Tweet":tweet.lower().strip(),"Affect Dimension":value}], ignore_index =False)
                value+=1
    dataframe.to_csv("emotion_detector_"+name+".csv")
    save_pickle_file(intensityClass,"intesity_class.pickle")
    save_pickle_file(map_emotion, "Affect_dimension.pickle")
preprocess_data("EI-oc-En-train/", "train", True)
