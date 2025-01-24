# models.py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer

from BPE import read_vocab_from_file, train_bpe
from sentiment_data import read_word_embeddings, WordEmbeddings, read_train
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import SentimentDatasetDAN, DAN, RANDOMDAN, SentimentDatasetSUBWORDDAN, SUBWORDDAN, SentimentDatasetRANDOMDAN
from utils import read_train_subword_indexer, Indexer, read_word_indexer


def collate_fn(batch):
    # Separate the sentences and labels
    embedding_indices, labels = zip(*batch)

    globalmax = max(len(embedding_index) for embedding_index in embedding_indices)

    # Pad the sentences so they are all of the same length
    padded_embedding_indices = []
    for embedding_index in embedding_indices:
        padded_embedding_index = embedding_index + [0] * (globalmax - len(embedding_index)) # <PAD>: 0
        padded_embedding_indices.append(torch.tensor(padded_embedding_index, dtype=torch.long))

    padded_embedding_indices = torch.stack(padded_embedding_indices, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_embedding_indices, labels

# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss

def train_epoch2(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss

# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss

def eval_epoch2(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4) #weight_decay: L2 regularization

    all_train_accuracy = []
    all_test_accuracy = []
    all_train_loss = []
    all_test_loss = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)
        all_train_loss.append(train_loss)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)
        all_test_loss.append(test_loss)
        if epoch % 10 == 9 or epoch < 10:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy, all_train_loss, all_test_loss


def experiment2(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-5) #weight_decay: L2 regularization

    all_train_accuracy = []
    all_test_accuracy = []
    all_train_loss = []
    all_test_loss = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch2(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)
        all_train_loss.append(train_loss)

        test_accuracy, test_loss = eval_epoch2(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)
        all_test_loss.append(test_loss)
        if epoch % 10 == 9 or epoch < 10:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')

    return all_train_accuracy, all_test_accuracy, all_train_loss, all_test_loss

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Check if the model type is "BOW"
    if args.model == "BOW":
        # Load dataset
        start_time = time.time()

        train_data = SentimentDatasetBOW("data/train.txt")
        dev_data = SentimentDatasetBOW("data/dev.txt")
        print(train_data)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")

        # Train and evaluate NN2
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy, nn2_train_loss, nn2_test_loss= experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy, nn3_train_loss, nn3_test_loss= experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model == "DAN":
        # Load dataset
        start_time = time.time()
        word_embeddings = read_word_embeddings("data/glove.6B.300d-relativized.txt")
        train_data = SentimentDatasetDAN("data/train.txt", word_embeddings)
        dev_data = SentimentDatasetDAN("data/dev.txt", word_embeddings)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")

        # Train and evaluate DAN
        start_time = time.time()
        print('\nDAN Model:')
        dan_train_accuracy, dan_test_accuracy, dan_train_loss, dan_test_loss= experiment(DAN(word_embeddings=word_embeddings,
                                                               hidden_size=100),
                                                           train_loader, test_loader)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"DAN trained in : {elapsed_time} seconds")

        # Plot the accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_accuracy, label='Training Accuracy')
        plt.plot(dan_test_accuracy, label='Dev Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Dev Accuracy for DAN')
        plt.legend()
        plt.grid()

        # Save the accuracy figure
        training_accuracy_file = 'DAN_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the loss
        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_loss, label='Training Loss')
        plt.plot(dan_test_loss, label='Dev Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Dev Loss for DAN')
        plt.legend()
        plt.grid()

        # Save the loss figure
        training_loss_file = 'DAN_loss.png'
        plt.savefig(training_loss_file)
        print(f"\n\nTraining and Dev loss plot saved as {training_loss_file}")

        # plt.show()
    elif args.model == "RANDOMDAN":
        # Load dataset
        start_time = time.time()
        word_indexer = read_word_indexer("data/glove.6B.300d-relativized.txt")
        #print(word_indexer)
        train_data = SentimentDatasetRANDOMDAN("data/train.txt", word_indexer)
        dev_data = SentimentDatasetRANDOMDAN("data/dev.txt", word_indexer)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False, collate_fn=collate_fn)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")

        # Train and evaluate RANDOMDAN
        start_time = time.time()
        print('\nRANDOMDAN Model:')
        dan_train_accuracy, dan_test_accuracy, dan_train_loss, dan_test_loss= experiment2(RANDOMDAN(word_indexer=word_indexer,
                                                                     hidden_size=200),
                                                           train_loader, test_loader)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"RANDOMDAN trained in : {elapsed_time} seconds")

        # Plot the accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_accuracy, label='Training Accuracy')
        plt.plot(dan_test_accuracy, label='Dev Accuracy')
        plt.yticks(np.arange(0.4, 1.0, 0.1))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Dev Accuracy for RANDOMDAN')
        plt.legend()
        plt.grid()

        # Save the accuracy figure
        accuracy_file = 'RANDOMDAN_accuracy.png'
        plt.savefig(accuracy_file)
        print(f"\n\nTraining and Dev accuracy plot saved as {accuracy_file}")

        # Plot the loss
        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_loss, label='Training Loss')
        plt.plot(dan_test_loss, label='Dev Loss')
        plt.yticks(np.arange(0.0, 2.0, 0.25))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Dev Loss for RANDOMDAN')
        plt.legend()
        plt.grid()

        # Save the loss figure
        training_loss_file = 'RANDOMDAN_loss.png'
        plt.savefig(training_loss_file)
        print(f"\n\nTraining and Dev loss plot saved as {training_loss_file}")

        # plt.show()
    elif args.model == "SUBWORDDAN":
        '''
        # Train BPE
        start_time = time.time()
        subword_indexer = read_train_subword_indexer("data/train.txt")
        #print(subword_indexer.objs_to_ints)
        vocab = read_vocab_from_file("data/train.txt")
        subword_vocab = train_bpe(subword_indexer, vocab, num_merges=15000)
        with open('data/bpe_train.txt', 'w') as file:
            for obj, idx in subword_indexer.objs_to_ints.items():
                file.write(f'{obj}: {idx}\n')
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"BPE trained in : {elapsed_time} seconds")
        '''

        # Load dataset
        start_time = time.time()
        objs_to_ints = {}
        with open('data/bpe_train.txt', 'r') as file:
            for line in file:
                obj, idx = line.strip().split(': ', 1)
                objs_to_ints[obj] = idx
        subword_indexer2 = Indexer()
        for obj, idx in objs_to_ints.items():
            subword_indexer2.objs_to_ints[obj] = idx
            subword_indexer2.ints_to_objs[idx] = obj

        train_data = SentimentDatasetSUBWORDDAN("data/train.txt", subword_indexer2)
        dev_data = SentimentDatasetSUBWORDDAN("data/dev.txt", subword_indexer2)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False, collate_fn=collate_fn)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")

        # Train and evaluate SUBWORDDAN
        start_time = time.time()
        print('\nSUBWORDDAN Model:')
        dan_train_accuracy, dan_test_accuracy, dan_train_loss, dan_test_loss= experiment2(SUBWORDDAN(subword_indexer=subword_indexer2,
                                                               hidden_size=200),
                                                           train_loader, test_loader)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"SUBWORDDAN trained in : {elapsed_time} seconds")

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_accuracy, label='Training Accuracy')
        plt.plot(dan_test_accuracy, label='Dev Accuracy')
        y_min, y_max = plt.ylim()
        #plt.yticks(np.arange(0.4, 1.0, 0.1))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Dev Accuracy for SUBWORDDAN')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        accuracy_file = 'SUBWORDDAN_accuracy.png'
        plt.savefig(accuracy_file)
        print(f"\n\nTraining and Dev accuracy plot saved as {accuracy_file}")

        # Plot the loss
        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_loss, label='Training Loss')
        plt.plot(dan_test_loss, label='Dev Loss')
        #plt.yticks(np.arange(0.0, 3.0, 0.5))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Dev Loss for SUBWORDDAN')
        plt.legend()
        plt.grid()

        # Save the loss figure
        training_loss_file = 'SUBWORDDAN_loss.png'
        plt.savefig(training_loss_file)
        print(f"\n\nTraining and Dev loss plot saved as {training_loss_file}")

        # plt.show()
if __name__ == "__main__":
    main()
