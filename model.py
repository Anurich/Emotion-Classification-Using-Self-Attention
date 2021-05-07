import torch
import torch.nn as nn
import sys
from dataloader import *
from einops import rearrange, reduce
class MultiheadedAttention(nn.Module):
    def __init__(self):
        super(MultiheadedAttention, self).__init__()
        self.num_head = 8
        self.d_model = 512
        self.split_ = self.d_model//self.num_head
        self.key  = nn.Linear(self.d_model,self.d_model)
        self.query = nn.Linear(self.d_model, self.d_model)
        self.value = nn.Linear(self.d_model, self.d_model)
        # linear layer

        self.linear_feed = nn.Linear(self.d_model,self.d_model )
        #self.output = nn.Linear(6144, vocab_size)


    def dot_product(self, query, key, value):
        dim = torch.tensor(query.size()[-1]).type(torch.FloatTensor)
        # query -> batch_size, head, seqLength, embeddingSize
        # key   -> batch_size, head, seqLength, embeddingSize
        # query * key ->  batch_size, head, seqLength, seqLength
        qk = torch.einsum('bhqj, bhkj -> bhqk', query, key)/torch.sqrt(dim)#-> (b h n n) where n: seqlength
        # perform softmax
        softmax = torch.softmax(qk, dim=-1)
        # multiply with value
        # softmax ->  batch_size, head, seqLength, seqLength
        # value   ->  batch_size, head, seqLength, embeddingSize
        # output  ->  batch_size, head, seqLength, embeddingSize
        attention = torch.einsum('bhqj, bhjv -> bhqv', softmax, value) #-> (b h n e) where e: embeddingSize
        return attention

    def forward(self, embedding):
        QK = self.key(embedding)
        QV = self.value(embedding)
        Qq = self.query(embedding)
        # reshape all
        #print(QK.shape, QV.shape, Qq.shape)
        QK = rearrange(QK,'b n (h d) -> b h n d', h = self.num_head)
        QV = rearrange(QV, 'b n (h d) -> b h n d', h=self.num_head)
        Qq = rearrange(Qq, 'b n (h d) -> b h n d', h=self.num_head)
        attention = rearrange(self.dot_product(Qq, QK, QV), 'b h n d -> b n (h d)')
        attention= self.linear_feed(attention)
        #output
        return attention



class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # multiheaded attention is ready
        self.multiheaded=  MultiheadedAttention()
        self.d_model = 512
        self.layerNorm  = nn.LayerNorm(self.d_model)

        # feed forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(in_features = self.d_model, out_features = self.d_model),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features = self.d_model, out_features = self.d_model)
                      )
        self.layerNorm1 = nn.LayerNorm(self.d_model)
    def forward(self, x):
        mutihead  = self.multiheaded(x)
        norm1 = self.layerNorm(mutihead + x)
        feed_out = self.feed_forward(norm1)

        output = self.layerNorm1(feed_out + norm1)
        return output

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
            self.attention_list.append(EncoderLayer())
        self.attention_list = nn.Sequential(*self.attention_list)
        self.finalLayer = nn.Sequential(
            nn.Linear(self.d_model, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, vocab_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, input_sent):
        batch_size, maxlen = input_sent.size()
        semantic_embed = self.sentence_encode(input_sent)
        position  = torch.arange(0, maxlen)
        pos_embed = self.position_encode(position)
        embed = semantic_embed + pos_embed
        output = self.attention_list(embed)
        output = reduce(output, 'b n e -> b e', reduction='mean')
        output = self.finalLayer(output)
        return output

def main():
    if __name__ == "__main__":
        batch_size = 64
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
