import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import TransformerEncoder, Classifier, TransformerDecoder, TransformerEncoder2, TransformerEncoder3
from utilities import Utilities
seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers

eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500  # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 500  # Number of iterations to evaluate perplexity on the test set

## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15  # epochs for classifier training
ff_dim = 100  # Feedforward hidden dimensionality

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data.
    """

    texts = []
    files = os.listdir(directory)
    for filename in files:
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts


def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])),
                                               "constant", 0)
    labels = torch.stack(labels)
    return padded_sequences, labels


def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses = []

    with torch.no_grad():  # Disable gradient computation during evaluation
        for i, (X, Y) in enumerate(data_loader):
            if i >= eval_iters:
                break

            X, Y = X.to(device), Y.to(device)
            loss = decoderLMmodel(X, Y)  # The decoder should return the scalar loss directly

            # Check if the output is a tuple, and if so, extract the loss
            if isinstance(loss, tuple):
                loss = loss[0]

            losses.append(loss.item())

    # Calculate mean loss and perplexity
    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()  # Set the model back to training mode
    return perplexity


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--part', type=str, required=True, help='Model type to train')  # 添加模型类型参数

    # 解析命令行参数
    args = parser.parse_args()

    if args.part == "part1":

        print("Loading data and creating tokenizer ...")
        texts = load_texts('speechesdataset')
        tokenizer = SimpleTokenizer(' '.join(texts))  # create a tokenizer from the data
        print("Vocabulary size is", tokenizer.vocab_size)

        train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
        train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch)

        # Model Setup
        encoder = TransformerEncoder(vocab_size=tokenizer.vocab_size, embed_dim=n_embd, num_layers=n_layer,
                                     num_heads=n_head, ff_dim=4 * n_embd, max_len=block_size).to(device)
        classifier = Classifier(embed_dim=n_embd, num_classes=n_output).to(device)

        # Report the number of parameters in the encoder
        num_params = count_parameters(encoder)
        print(f"Number of parameters in the encoder: {num_params}")

        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate)
        # for the classification  task, you will train for a fixed number of epochs like this:
        # Sanity check using utilities.py
        utilities = Utilities(tokenizer, encoder)
        utilities.sanity_check("This is a sample sentence for sanity check.", block_size)
        # Training Loop for Classification Task
        for epoch in range(epochs_CLS):
            encoder.train()
            classifier.train()
            total_loss = 0
            correct = 0
            total = 0
            for xb, yb in train_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)

                # Forward pass
                embeddings, _ = encoder(xb)
                #embeddings = embeddings.mean(dim=1)  # Apply mean pooling over the sequence length
                outputs = classifier(embeddings)
                loss = criterion(outputs, yb)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == yb).sum().item()
                total += yb.size(0)

            train_accuracy = correct / total
            print(f"Epoch [{epoch + 1}/{epochs_CLS}], Loss: {total_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            # Evaluation on Test Set after each epoch for Classification Task
            # List to store test accuracies for reporting
            test_accuracies = []
            encoder.eval()
            classifier.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in test_CLS_loader:
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    embeddings, _ = encoder(inputs)
                    #embeddings = embeddings.mean(dim=1)  # Apply mean pooling over the sequence length
                    outputs = classifier(embeddings)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            test_accuracy = correct / total
            test_accuracies.append(test_accuracy)
            print(f"Test Accuracy after Epoch {epoch + 1}: {test_accuracy:.4f}")

        # Print final accuracy
        final_accuracy = test_accuracies[-1]
        print(f"Final Test Accuracy after {epochs_CLS} epochs: {final_accuracy:.4f}")

    if args.part == "part2":

        print("Loading data and creating tokenizer ...")
        texts = load_texts('speechesdataset')
        tokenizer = SimpleTokenizer(' '.join(texts))  # create a tokenizer from the data
        print("Vocabulary size is", tokenizer.vocab_size)

        inputfile = "speechesdataset/train_LM.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()

        train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

        test_file = "speechesdataset/test_LM_hbush.txt"
        with open(test_file, 'r', encoding='utf-8') as f:
            test_text = f.read()
        test_LM_dataset = LanguageModelingDataset(tokenizer, test_text, block_size)
        test_LM_loader = DataLoader(test_LM_dataset, batch_size=batch_size, shuffle=False)


        # Set up the Transformer Decoder
        decoder = TransformerDecoder(vocab_size=tokenizer.vocab_size, embed_dim=n_embd, num_layers=n_layer,
                                     num_heads=n_head, ff_dim=ff_dim, max_len=block_size).to(device)

        # Report the number of parameters in the encoder
        num_params = count_parameters(decoder)
        print(f"Number of parameters in the decoder: {num_params}")

        # Sanity check using utilities.py
        utilities = Utilities(tokenizer, decoder)
        utilities.sanity_check("This is a sample sentence for sanity check.", block_size)

        # Loss and Optimizer Setup
        optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

        # Training Loop for Language Modeling
        # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)
            decoder.train()

            # Get the loss from the decoder
            loss = decoder(xb, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate and report perplexity every eval_interval iterations
            if i % eval_interval == 0 or i == max_iters - 1:
                perplexity = compute_perplexity(decoder, train_LM_dataset, eval_iters=eval_iters)
                print(f"Iteration {i}, Perplexity: {perplexity:.4f}")


        test_files = ["speechesdataset/test_LM_obama.txt", "speechesdataset/test_LM_wbush.txt",
                      "speechesdataset/test_LM_hbush.txt"]
        test_names = ["Obama", "W. Bush", "H. Bush"]

        for test_file, test_name in zip(test_files, test_names):
            with open(test_file, 'r', encoding='utf-8') as f:
                test_text = f.read()

            test_dataset = LanguageModelingDataset(tokenizer, test_text, block_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Compute and report perplexity for each politician
            perplexity = compute_perplexity(decoder, test_loader, eval_iters=eval_iters)
            print(f"Perplexity on {test_name} Test Set: {perplexity:.4f}")


    if args.part == "part3-alibi":

        print("Loading data and creating tokenizer ...")
        texts = load_texts('speechesdataset')
        tokenizer = SimpleTokenizer(' '.join(texts))  # create a tokenizer from the data
        print("Vocabulary size is", tokenizer.vocab_size)

        train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
        train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch)

        # Model Setup
        encoder = TransformerEncoder2(vocab_size=tokenizer.vocab_size, embed_dim=n_embd, num_layers=n_layer,
                                     num_heads=n_head, ff_dim=4 * n_embd, max_len=block_size).to(device)
        classifier = Classifier(embed_dim=n_embd, num_classes=n_output).to(device)

        # Report the number of parameters in the encoder
        num_params = count_parameters(encoder)
        print(f"Number of parameters in the encoder: {num_params}")

        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate)
        # for the classification  task, you will train for a fixed number of epochs like this:
        # Sanity check using utilities.py
        utilities = Utilities(tokenizer, encoder)
        utilities.sanity_check("This is a sample sentence for sanity check.", block_size)
        # Training Loop for Classification Task
        for epoch in range(epochs_CLS):
            encoder.train()
            classifier.train()
            total_loss = 0
            correct = 0
            total = 0
            for xb, yb in train_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)

                # Forward pass
                embeddings, _ = encoder(xb)
                #embeddings = embeddings.mean(dim=1)  # Apply mean pooling over the sequence length
                outputs = classifier(embeddings)
                loss = criterion(outputs, yb)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == yb).sum().item()
                total += yb.size(0)

            train_accuracy = correct / total
            print(f"Epoch [{epoch + 1}/{epochs_CLS}], Loss: {total_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            # Evaluation on Test Set after each epoch for Classification Task
            # List to store test accuracies for reporting
            test_accuracies = []
            encoder.eval()
            classifier.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in test_CLS_loader:
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    embeddings, _ = encoder(inputs)
                    #embeddings = embeddings.mean(dim=1)  # Apply mean pooling over the sequence length
                    outputs = classifier(embeddings)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            test_accuracy = correct / total
            test_accuracies.append(test_accuracy)
            print(f"Test Accuracy after Epoch {epoch + 1}: {test_accuracy:.4f}")

        # Print final accuracy
        final_accuracy = test_accuracies[-1]
        print(f"Final Test Accuracy after {epochs_CLS} epochs: {final_accuracy:.4f}")

    if args.part == "part3-disentangled":

        print("Loading data and creating tokenizer ...")
        texts = load_texts('speechesdataset')
        tokenizer = SimpleTokenizer(' '.join(texts))  # create a tokenizer from the data
        print("Vocabulary size is", tokenizer.vocab_size)

        train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
        train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch,
                                      shuffle=True)
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch)

        # Model Setup
        encoder = TransformerEncoder3(vocab_size=tokenizer.vocab_size, embed_dim=n_embd, num_layers=n_layer,
                                      num_heads=n_head, ff_dim=4 * n_embd, max_len=block_size).to(device)
        classifier = Classifier(embed_dim=n_embd, num_classes=n_output).to(device)

        # Report the number of parameters in the encoder
        num_params = count_parameters(encoder)
        print(f"Number of parameters in the encoder: {num_params}")

        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate)
        # for the classification  task, you will train for a fixed number of epochs like this:
        # Sanity check using utilities.py
        utilities = Utilities(tokenizer, encoder)
        utilities.sanity_check("This is a sample sentence for sanity check.", block_size)
        # Training Loop for Classification Task
        for epoch in range(epochs_CLS):
            encoder.train()
            classifier.train()
            total_loss = 0
            correct = 0
            total = 0
            for xb, yb in train_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)

                # Forward pass
                embeddings, _ = encoder(xb)
                # embeddings = embeddings.mean(dim=1)  # Apply mean pooling over the sequence length
                outputs = classifier(embeddings)
                loss = criterion(outputs, yb)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == yb).sum().item()
                total += yb.size(0)

            train_accuracy = correct / total
            print(f"Epoch [{epoch + 1}/{epochs_CLS}], Loss: {total_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            # Evaluation on Test Set after each epoch for Classification Task
            # List to store test accuracies for reporting
            test_accuracies = []
            encoder.eval()
            classifier.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in test_CLS_loader:
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    embeddings, _ = encoder(inputs)
                    # embeddings = embeddings.mean(dim=1)  # Apply mean pooling over the sequence length
                    outputs = classifier(embeddings)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            test_accuracy = correct / total
            test_accuracies.append(test_accuracy)
            print(f"Test Accuracy after Epoch {epoch + 1}: {test_accuracy:.4f}")

        # Print final accuracy
        final_accuracy = test_accuracies[-1]
        print(f"Final Test Accuracy after {epochs_CLS} epochs: {final_accuracy:.4f}")

if __name__ == "__main__":
    main()
