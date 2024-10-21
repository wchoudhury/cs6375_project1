import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
import random
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt

unk = '<UNK>'

class RNN(nn.Module):
    def __init__(self, input_dim, h):
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        # Bi-Directional RNN
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh', bidirectional=True)
        self.W = nn.Linear(h * 2, 5)  # Adjust output layer for bidirectional RNN (h*2)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()
        self.layer_norm = nn.LayerNorm(h * 2)
        self.dropout = nn.Dropout(p=0.5)  # Dropout to prevent overfitting
        self.init_weights()

    def init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs, input_lengths=None):
        if input_lengths is not None:
            packed_inputs = pack_padded_sequence(inputs, input_lengths, enforce_sorted=False)
            packed_output, hidden = self.rnn(packed_inputs)
            rnn_output, _ = pad_packed_sequence(packed_output)
        else:
            rnn_output, hidden = self.rnn(inputs)

        rnn_output = self.layer_norm(rnn_output)  # Apply layer normalization
        rnn_output = self.dropout(rnn_output)     # Apply dropout

        summed_output = torch.sum(rnn_output, dim=0)  # Summing over time steps as per project guidelines
        output = self.W(summed_output)
        predicted_vector = self.softmax(output)
        return predicted_vector


def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(), int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"]-1)))
    return tra, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, default=128, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default="to fill", help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    word_embedding = pickle.load(open('/Users/wchoudhury/cs6375_project1/Data_Embedding/word_embedding.pkl', 'rb'))

    stopping_condition = False
    epoch = 0
    last_train_accuracy = 0
    last_validation_accuracy = 0

    # Initialize lists to store losses and accuracies for plotting
    training_losses = []
    validation_accuracies = []

    while not stopping_condition and epoch < args.epochs:
        random.shuffle(train_data)
        model.train()
        print("Training started for epoch {}".format(epoch + 1))
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)

        loss_total = 0
        loss_count = 0
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)

                # Remove punctuation
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]
                vectors = torch.tensor(vectors, requires_grad=False).view(len(vectors), 1, -1)

                output = model(vectors)
                example_loss = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label]))

                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1

                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

        avg_training_loss = loss_total / loss_count
        training_losses.append(avg_training_loss.item())
        print(f"Avg training loss for epoch {epoch + 1}: {avg_training_loss}")
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        train_accuracy = correct / total

        model.eval()
        correct = 0
        total = 0
        random.shuffle(valid_data)
        print("Validation started for epoch {}".format(epoch + 1))

        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]

            vectors = torch.tensor(vectors, requires_grad=False).view(len(vectors), 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1

        validation_accuracy = correct / total
        validation_accuracies.append(validation_accuracy)
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, validation_accuracy))

        if validation_accuracy < last_validation_accuracy and train_accuracy > last_train_accuracy:
            stopping_condition = True
            print("Training done to avoid overfitting!")
            print("Best validation accuracy is:", last_validation_accuracy)
        else:
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = train_accuracy

        epoch += 1

    # Plot learning curves
    epochs_range = range(1, epoch + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, training_losses, label='Training Loss')
    plt.plot(epochs_range, validation_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
