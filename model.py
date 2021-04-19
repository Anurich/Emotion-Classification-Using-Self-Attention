import torch
import torch.nn as nn
import sys
from dataloader import *
class MultiheadedAttention(nn.Module):
    def __init__(self, vocab_size):
        super(MultiheadedAttention, self).__init__()
        self.num_head = 8
        self.d_model = 512
        self.split_ = self.d_model//self.num_head
        self.key  = nn.Linear(self.d_model,self.d_model)
        self.query = nn.Linear(self.d_model, self.d_model)
        self.value = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(p=0.1)
        self.layernorm= nn.LayerNorm(512)
        # feed forward network

        self.linear_feed = nn.Linear(self.d_model,self.d_model )
        #self.output = nn.Linear(6144, vocab_size)


    def dot_product(self, query, key, value):
        dim = torch.tensor(query.size()[-1]).type(torch.FloatTensor)
        qk = torch.matmul(query/dim**0.5, torch.transpose(key,2,3))
        # perform softmax
        softmax = torch.softmax(qk, dim=-1)
        # multiply with value
        attention = torch.matmul(softmax, value)
        return attention

    def forward(self, embedding):
        QK = self.key(embedding)
        QV = self.value(embedding)
        Qq = self.query(embedding)
        # reshape all
        #print(QK.shape, QV.shape, Qq.shape)
        QK = QK.view(32,-1, self.num_head, self.split_)
        QV = QV.view(32,-1, self.num_head, self.split_)
        Qq = Qq.view(32,-1, self.num_head, self.split_)
        attention = self.dot_product(Qq, QK, QV)
        attention = attention.view(32, -1, self.d_model)
        attention = self.dropout(attention)
        # summantion and normalization
        attention = embedding + attention
        normalizedAttention = self.layernorm(attention + embedding)
        dense  = torch.relu(self.linear_feed(normalizedAttention))
        dense  = self.dropout(dense)

        dense_normalized = self.layernorm(normalizedAttention + dense)
        #output
        return dense_normalized


class Encoder(nn.Module):
    def __init__(self, vocab_size, numtimes=2):
        super(Encoder, self).__init__()
       # define embedding layer for input
        self.d_model= 512
        self.sentence_encode = nn.Embedding(num_embeddings=vocab_size+1, embedding_dim =self.d_model)
        # define the poistional encoding
        self.position_encode = nn.Embedding(num_embeddings=12, embedding_dim = self.d_model)
        self.attention_list = []
        for i in range(numtimes):
            self.attention_list.append(MultiheadedAttention(vocab_size))
        self.attention_list = nn.Sequential(*self.attention_list)
        self.dense_output = nn.Linear(6144, vocab_size)

    def forward(self, input_sent):
        batch_size, maxlen = input_sent.size()
        semantic_embed = self.sentence_encode(input_sent)
        position  = torch.arange(0, maxlen)
        pos_embed = self.position_encode(position)
        embed = semantic_embed + pos_embed
        output = self.attention_list(embed)
        output = torch.log_softmax(self.dense_output(output.view(batch_size, -1)),dim=-1)
        return output

def main():
    if __name__ == "__main__":
        batch_size=32
        batchLoader = get_data_loader("emotion_detector_train.csv", batch_size, train=True)
        vocab_size = len(batchLoader.dataset.index_to_word)
        episode = 100
        enc  = Encoder(vocab_size, 6)
        criteria = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(enc.parameters(),lr=0.001)
        for ep in range(episode):
            stats = None
            for X, Y in batchLoader:
                X = X.long().squeeze()
                Y = Y.long()
                optimizer.zero_grad()
                output = enc(X)
                loss = criteria(output, Y)
                loss.backward()
                optimizer.step()
                stats = "[%d] Loss: %.4f, Perplexity: %5.4f "%(ep, loss.item(), np.exp(loss.item()))
                print("\r" +stats, end="")
                sys.stdout.flush()
            print("\r"+stats)
            if ep%10 == 0 and ep != 0:
                torch.save({
                    "model_state": enc.state_dict(),
                },"model.pt")

main()
