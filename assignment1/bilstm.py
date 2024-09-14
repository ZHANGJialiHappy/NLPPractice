import argparse
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Classificator(nn.Module):
    def __init__(self, embed_dim, lstm_dim, vocab_dim, num_labels):
        super(Classificator, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_dim, embed_dim)
        self.lstm = torch.nn.LSTM(embed_dim, lstm_dim, bidirectional=True, batch_first=True)
        self.hidden_to_tag = torch.nn.Linear(lstm_dim * 2, num_labels)
        self.lstm_dim = lstm_dim
        
    
    def forward(self, inputs):
        word_vectors = self.word_embeddings(inputs)
        lstm_out, _ = self.lstm(word_vectors)
        backward_out = lstm_out[:,0,-self.lstm_dim:]
        forward_out = lstm_out[:,-1,:self.lstm_dim]
        combined = torch.zeros((len(lstm_out), self.lstm_dim*2), device=lstm_out.device)
        combined[:,:self.lstm_dim] = forward_out
        combined[:,-self.lstm_dim:] = backward_out

        y = self.hidden_to_tag(combined) 
        log_probs = F.softmax(y)
        return log_probs
    
class Vocabulary():
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        # add special unk token
        self.getIdx('[UNK]', True)

    def getWord(self, idx: int):
        return self.idx2word(idx)
    
    def getIdx(self, word: str, update: bool):
        if update and word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
            return len(self.idx2word)-1
        if word not in self.word2idx:
            return 0 # [UNK] is always in 0
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

def readFile(path: str, word_vocabulary: Vocabulary, label_vocabulary: Vocabulary, update: bool, max_len: 100):
    labels = []
    features = []
    for line in open(path):
        tok = line.strip().split('\t')
        label = label_vocabulary.getIdx(tok[0], update)
        labels.append(label) 

        text = tok[1]
        words = [word_vocabulary.getIdx(word, update) for word in text.split(' ')[:max_len]]
        features.append(words)

    # convert to torch
    labels_torch = torch.tensor(labels, dtype=torch.long)
    features_torch = torch.zeros((len(features), max_len), dtype=torch.long)
    for sentenceIdx, sentence in enumerate(features):
        for word_idx, word in enumerate(sentence):
            features_torch[sentenceIdx][word_idx] = features[sentenceIdx][word_idx]
    
    return features_torch, labels_torch




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--dev_data", type=str, default='')
    parser.add_argument("--lstm_dim", type=int, default=50)
    parser.add_argument("--embed_dim", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    max_len = 100 # hardcoded for now to limit memory usage 
    word_vocabulary = Vocabulary()
    label_vocabulary = Vocabulary()
    # train labels is a 1d tensor containing a numerical 
    # representation of all labels
    # train features is ...
    train_features, train_labels = readFile(args.train_data, word_vocabulary, label_vocabulary, True, max_len)

    # This is a simplistic implementation of batching, it removes any remainders.
    # So if you have a tensor of labels of size 100, and batch size == 32
    # it would just remove the last 4 (100-32*3), and put chunks of 32 in the new
    # dimension. The resulting tensor would be of size (32,3)
    num_batches = int(len(train_labels)/args.batch_size)
    train_feats_batches = train_features[:args.batch_size*num_batches].view(num_batches,args.batch_size, max_len)
    train_labels_batches = train_labels[:args.batch_size*num_batches].view(num_batches, args.batch_size)

    
    sentiment_model = Classificator(args.embed_dim, args.lstm_dim, len(word_vocabulary), len(label_vocabulary))
    loss_function = nn.CrossEntropyLoss()

    # compile and train the model
    optimizer = optim.Adam(sentiment_model.parameters(), lr=0.01)
    start = time.time()

    sentiment_model.train()
    print(sentiment_model)
    for epoch in range(5):
        epoch_loss = 0.0
        for feats, labels in zip(train_feats_batches, train_labels_batches):
            optimizer.zero_grad()
            y = sentiment_model.forward(feats)
            loss = loss_function(y,labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(epoch, epoch_loss/len(train_feats_batches), time.time() - start)


        if args.dev_data != '':
            dev_features, dev_labels = readFile(args.dev_data, word_vocabulary, label_vocabulary, False, max_len)
            num_batches = int(len(dev_labels)/args.batch_size)
            dev_feats_batches = dev_features[:args.batch_size*num_batches].view(num_batches,args.batch_size, max_len)
            dev_labels_batches = dev_labels[:args.batch_size*num_batches].view(num_batches, args.batch_size)
            
            sentiment_model.eval()
            cor = 0
            total = 0
            for feats, labels in zip(dev_feats_batches, dev_labels_batches):
                pred_labels = torch.argmax(sentiment_model.forward(feats), 1)
                cor += torch.sum(labels==pred_labels).item()
                total += len(labels)
            print(cor/total)
        print()


