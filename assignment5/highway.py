#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f

    def __init__(self, word_embed_size):
        super(Highway, self).__init__()
        self.x_proj = nn.Linear(word_embed_size, word_embed_size, bias=True)
        self.x_gate = nn.Linear(word_embed_size, word_embed_size, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_convout):
        X_proj = self.relu(self.x_proj(X_convout))
        X_gate = self.sigmoid(self.x_gate(X_convout))
        X_highway = torch.mul(X_gate, X_proj) + torch.mul(1 - X_gate, X_convout)

        return X_highway

    ### END YOUR CODE

