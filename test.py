from model import *
import pickle
import torch


if __name__ == "__main__":
    index_to_word = pickle.load(open("index_to_word.pickle","rb"))
    aff_dim = pickle.load(open("Affect_dimension.pickle","rb"))
    #intensity_class = pickle.load(open("intensity_class.pickle","rb"))
    enc = Encoder(len(index_to_word),6)
    checkpoint = torch.load("model.pt")
    enc.load_state_dict(checkpoint["model_state"])
    enc.eval()
    batchLoader = get_data_loader("emotion_detector_test.csv",32, False)
    X, Y = next(iter(batchLoader))
    prediction = torch.argmax(enc(X.squeeze().long()), dim=-1)
    for i in range(X.shape[0]):
        sent = ""
        sent1 = X[i][0]
        for num in sent1:
            num = num.item()
            if num != 0:
                if index_to_word.get(num) != None:
                    sent += index_to_word[num]+" "
                else:
                    sent += index_to_word[1]+" "

        predicted_val = prediction[i].item()
        str_pred = aff_dim.get(predicted_val)

        print(sent+" : "+ str_pred)
