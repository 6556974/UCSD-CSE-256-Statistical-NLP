import torch
from torch import nn
import torch.nn.functional as F
from sentiment_data import read_sentiment_examples, WordEmbeddings
from torch.utils.data import Dataset


class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, word_embeddings, max_len=64):
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)

        # Extract sentences and labels from the examples
        self.sentences = [ex.words for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]

        # Store the word embeddings
        self.word_embeddings = word_embeddings

        # Convert sentences to embeddings
        self.max_len = max_len
        self.embeddings = [self._sentence_to_embedding(sentence) for sentence in self.sentences]
        self.embeddings = torch.stack(self.embeddings)

        # Convert labels to PyTorch tensors
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def _sentence_to_embedding(self, sentence):
        embedding_length = self.word_embeddings.get_embedding_length()
        sentence_embedding = torch.zeros(self.max_len, embedding_length)
        for i, word in enumerate(sentence[:self.max_len]):
            sentence_embedding[i] = torch.tensor(self.word_embeddings.get_embedding(word), dtype=torch.float32)
        return sentence_embedding

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return the feature vector and label for the given index
        return self.embeddings[idx], self.labels[idx]


class DAN(nn.Module):
    def __init__(self, word_embeddings, hidden_size):
        super().__init__()
        self.embedding = word_embeddings.get_initialized_embedding_layer(frozen=True)
        #self.dropout_embedding = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(300, hidden_size)
        #self.dropout_fc1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        #self.dropout_fc2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Calculate the average embedding for the sentence
        x = torch.mean(x, dim=1)
        # Pass the average embedding through the network
        x = F.relu(self.fc1(x))
        #x = self.dropout_fc1(x)
        x = F.relu(self.fc2(x))
        #x = self.dropout_fc2(x)
        x = self.fc3(x)
        x = self.log_softmax(x)
        return x


class SentimentDatasetRANDOMDAN(Dataset):
    def __init__(self, infile, word_indexer):
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)

        # Extract sentences and labels from the examples
        self.labels = [ex.label for ex in self.examples]
        self.word_indexer = word_indexer
        self.embedding_indices = []
        for ex in self.examples:
            self.embedding_indices.append([self.word_indexer.index_of(word) for word in ex.words])
        #print(self.embedding_indices[:10])
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.embedding_indices[idx], self.labels[idx]


class RANDOMDAN(nn.Module):
    def __init__(self, word_indexer, hidden_size):
        #print(word_indexer.objs_to_ints)
        super().__init__()
        self.embedding = nn.Embedding(len(word_indexer.objs_to_ints), 300)
        self.fc1 = nn.Linear(300, hidden_size)
        self.dropout_fc1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout_fc2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        # Calculate the average embedding for the sentence
        x = torch.mean(x, dim=1)
        # Pass the average embedding through the network
        x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc2(x)
        x = self.fc3(x)
        x = self.log_softmax(x)
        return x


class SentimentDatasetSUBWORDDAN(Dataset):
    def __init__(self, infile, subword_indexer):
        self.examples = read_sentiment_examples(infile)

        self.labels = [ex.label for ex in self.examples]
        self.subword_indexer = subword_indexer
        self.embedding_indices = []
        for ex in self.examples:
            words_indices = []
            for word in ex.words:
                if word != '':
                    idx = int(len(self.subword_indexer) - 1)

                    while idx != 1 and word != '':
                        if subword_indexer.get_object(str(idx)) in word:
                            words_indices.append(idx)
                            word = word.replace(subword_indexer.get_object(str(idx)), '', 1)
                        idx -= 1
            self.embedding_indices.append(words_indices)
        #print(self.embedding_indices)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.embedding_indices[idx], self.labels[idx]

class SUBWORDDAN(nn.Module):
    def __init__(self, subword_indexer, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(len(subword_indexer), 300) # first param: vacab_size
        self.fc1 = nn.Linear(300, hidden_size)
        self.dropout_fc1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout_fc2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        # Calculate the average embedding for the sentence
        x = torch.mean(x, dim=1)
        # Pass the average embedding through the network
        x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc2(x)
        x = self.fc3(x)
        x = self.log_softmax(x)
        return x
