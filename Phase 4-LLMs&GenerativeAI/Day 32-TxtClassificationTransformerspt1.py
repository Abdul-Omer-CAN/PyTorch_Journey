# Today we are building a Sentiment Classifier. Given a movie review, predict if it is positive or negative. #
# We will use Transformer Block from yesterday. IMDB Dataset consisting of 50k movie reviews. Train to classify positive from negative. #


import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, Dataset  # 'dataloader' feeds data into batches. 'dataset' base class for creating custom datasets.

## Create Self Attention class ##


class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_size)  # swapping (-2,-1) is the index position. swapping second last position with last.
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)
        return out


## Transformer Block ##

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, ff_size):
        super().__init__()
        self.attention = SelfAttention(embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_size),
            nn.ReLU(),
            nn.Linear(ff_size, embed_size)
        )

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(x + attended)
        fed = self.ff(x)
        x = self.norm2(x + fed)
        return x


## Text Classifier ##

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, ff_size, num_classes):  # 'vocab_size' how many unique words in our vocab. 'num_classes' how many categories(2 positive or negative)
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)  # converts word indices to vectors. cat might contain 423 embedding turns it into 64 no. Each word gets its own vector. Think of nn.Embedding as a dictionary/lookup table.
        self.transformer = TransformerBlock(embed_size, ff_size)  # Our transformer block. Processes sequences and finds relationship between words.
        self.fc = nn.Linear(embed_size, num_classes)  # Final Layer. 64 numbers -> gives us 2 outputs(positive or negative)

    def forward(self, x):
        x = self.embedding(x)  # Convert word indices to vectors. Input[batch, sequence] -> Output[batch, sequence, 64]. What is a vector? -> It is a list of numbers [0.2, 0.8, 0.1, 0.5...]. Each word gets represented as a vector of 64 numbers.
        x = self.transformer(x)  # Pass thru TransformerBlock. Shape stays the same [batch, sequence, 64]
        x = x.mean(dim=1)  # Avg all word vectors into one sentence vectors. We use dim=1 because dim=1 is our index 1 in [batch, sequence, embed]. Index 1 is sequence. We want avg of the 5 words. words is the sequence.
        x = self.fc(x)  # Final classification. [batch, 64] -> [batch, 2] (positive or negative)
        return x


## DataSet ##


class IMDBDataset(Dataset):  # Customer dataset from PyTorch's Dataset
    def __init__(self, num_samples=100, seq_len=20, vocab_size=1000):  # 'num_samples=100' -> 100 fake reviews. 'seq_len=20' -> each review has a length of 20 words. 'vocab_size=1000' -> 1000 unique words in our vocab.
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))  # Creates a 100 fake reviews. Each review has 20 random word indices between 0 and 1000. [100, 20] 100 reviews and 20 words each.
        self.labels = torch.randint(0, 2, (num_samples,))  # Creates a 100 random labels. 0 negative and 1 positive. Shape is [100]. # min value is 0 and max value is 2. num_samples is 100 labels.

    def __len__(self):  # Returns how many samples we have. PyTorch DataLoader needs this to know datasize.
        return len(self.data)

    def __getitem__(self, idx):  # Returns one sample at a time by index.
        return self.data[idx], self.labels[idx]  # idx = 0 gives us first review and its label. DataLoader calls this automatically when loading batches.


## DataLoader - Feeds data in batches ##

dataset = IMDBDataset()
loader = DataLoader(dataset, batch_size=16, shuffle=True)

## Model Setup ##

model = TextClassifier(vocab_size=1000, embed_size=64, ff_size=256, num_classes=2)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


epochs = 5  # 5 Phases of training.

for epoch in range(epochs):
    model.train()  # training mode ON
    for texts, labels in loader:  # Loop thru batches of 16 reviews at a time

        outputs = model(texts)  # Forward pass - feed texts thru TextClassifier.
        loss = loss_fn(outputs, labels)  # Compare predictions vs real labels.

        optimizer.zero_grad()  # Clear old gradients.
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights.

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")  # print loss per epoch.


## Accuracy Check ##

model.eval()  # switch to eval mode
correct = 0
total = 0

with torch.no_grad():  # no gradient tracking needed.
    for texts, labels in loader:  # loop thru all batches.
        outputs = model(texts)  # Get predictions.
        _, predicted = torch.max(outputs, 1)  # pick highest score. 0 -> negative or 1 -> positive. The (outputs, 1) 1 signifes dim=1. index 1 looks across the row. Tells us which class wins. Class 0 for Negative or Class 1 for positive.
        total += labels.size(0)  # add batch size to total.
        correct += (predicted == labels).sum().item()  # count correct predictions.

print(f"Accuracy: {100 * correct / total:.2f}%")  # correct/total x 100 = accuracy %
