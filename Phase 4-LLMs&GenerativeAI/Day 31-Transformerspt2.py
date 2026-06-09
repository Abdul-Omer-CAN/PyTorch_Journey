## Import Section ##

import torch  # Main PyTorch network
import torch.nn as nn  # Neural Network tools
import math  # Attention formula has a square root in it. Hence math is needed.


## Self Attention Class ##

# 3 ingredients of  "Attention Mechanism" -> 1. Q for Query(What am i looking for) | 2. K for Key(What do i contain) | 3. V for Value(What do i actually give)
# Think of it like a search engine. Query is your search term. Key is the title of each webpage. Value is the actualy content of each page.
# If Q and K are similiar of a high match. attention weight would be high and the word would get more influence.

# Formula is -> Attention(Q, K, V) = softmax(QK^T / √d_k) × V
# Breakdown is -> QK^T -> multiply query with key will give similiarity scores.
# Step 1: Q × K^T          → compare cat's query to every word's key → get scores
# Step 2: ÷ √d_k           → scale down so numbers don't explode
# Step 3: softmax           → turn scores into weights that add up to 1
# Step 4: × V              → mix the values using those weights → final output

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size  # Stores the embedding dimension (e.g 64)
        self.query = nn.Linear(embed_size, embed_size)  # Q Layer - What am i looking for?
        self.key = nn.Linear(embed_size, embed_size)  # K Layer - What do i contain?
        self.value = nn.Linear(embed_size, embed_size)  # V Layer - What do i give you?

    def forward(self, x):
        Q = self.query(x)  # transform input into Query
        K = self.key(x)  # transform input into Key
        V = self.value(x)  # transform input into Value

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_size)  # QK^T / √d_k also flips K so dimensions line up [4,64]x[64,4]=[4,4]
        weights = torch.softmax(scores, dim=-1)
        # convert scores to probablities. dim=-1 means apply softmax row by row until the last dimension. If we use +1 it could mean second dimension because first would be 0(going down).
        # softmax converts raw scores to probabilities that add up to 1.
        # dim=-1 means apply across the last dimension (going across each row).
        # Each word gets its own probabilities - not all words combined.
        # dim=0 = going down (rows). dim=-1 = going across (columns) same as dim=1.
        # We use -1 because it always means last dimension regardless of tensor shape. Safer than +1.

        out = torch.matmul(weights, V)  # multiply attention weights by values. High weight = take more from that word's value.
        return out

## Transformer Block ##


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, ff_size):
        super().__init__()
        self.attention = SelfAttention(embed_size)  # Runs Self-Attention -> finds relationship between all words.
        self.norm1 = nn.LayerNorm(embed_size)  # Normalizes After Attention -> Keeps Values stable.
        self.norm2 = nn.LayerNorm(embed_size)  # Normalizes After feed forward -> Keeps Values stable.
        self.ff = nn.Sequential(  # Feed forward network - 2 linear layers that process each word individually after attention.
            nn.Linear(embed_size, ff_size),  # This expands the dimensions from 64 to 256. 4x bigger gives more room to process.
            nn.ReLU(),  # Activation layer. If the number is negative makes it zero and keeps positive as it is. It flats the negative and zero and creates a corner.
            # Everything below zero → flat line at 0
            # Everything above zero → diagonal line going up
            # They meet at zero → that corner is the non-linearity
            nn.Linear(ff_size, embed_size)  # Second layer compresses back down to from 256 to 64.
        )

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(x + attended)  # attended is what attention found which is the new things it learnt. + x is our original word. This is called Residual Connection without this information could get lost as it passes through attention. Deep networks forget the original input.
        fed = self.ff(x)  # passes thru feed forward network. gets richer word representations.
        x = self.norm2(x + fed)  # Residual connection. adds original x back to fed. Normalizes to keep values stable.
        return x  # x is now the final enriched representation.


## TEST BLOCK ##

# Test SelfAttention


attention = SelfAttention(embed_size=64)  # creates an instance(object created from a blueprint or class) of our SelfAttention class with each word represented by 64 numbers.
x = torch.randn(1, 5, 64)  # 1 represents batch size which is one sentence. 5 is the sequence length e.g 5 words. 64 is the embedding size aka 64 numbers per word.
out = attention(x)  # Pass our fake 5 word sentence through the attention layer. It runs forward(x) automatically.
print("Input shape: ", x.shape)  # Should print 'torch.size([1, 5, 64]) -> 1 sentence containing 5 words and each word represented w/ 64 numbers.
print("Output shape: ", out.shape)  # Will print the same as above. Shape is the same. Attention doesnt change the shape. It enriches each word with context.

# Test TransformerBlock

block = TransformerBlock(embed_size=64, ff_size=256)  # create an instance of TransformerBlock and feedforward to 256 which is 4x bigger.
x = torch.randn(1, 5, 64)  # Fake input consisting of 1 sentence, 5 words and 64 numbers set which each word has.
out = block(x)  # pass input thru full TransformerBlock. Runs: Attention -> Norm1 -> FeedForward -> Norm2
print("TransformerBlock Input shape: ", x.shape)
print("TransformerBlock Output shape: ", out.shape)
