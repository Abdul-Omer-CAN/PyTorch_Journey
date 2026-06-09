# In Day 27 we used Long-Short-Term Memory to predict stock prices by the time it reached day 100 it forgot day 1. That is the vanishing gradient problem.
# Transformers solves both problems:
# 1- Speed-> processes all words at once in parallel, not one by one.
# 2- Memory-> every word can directly attend to every other word regardless of distance.
# The mechanism that makes this possible is called 'Attention' specifically 'Self-Attention'.
# An example that we can use is.. "The animal didn't cross the street because it was too tired".
# Over here it refers to the animal that crossed the street. Not the street itself. Every word attends to every other word and assigns a weight.
# That is the core of transformers.
