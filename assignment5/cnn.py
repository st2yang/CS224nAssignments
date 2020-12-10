#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    pass
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g

    def __init__(self, in_dim, out_dim):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=5, padding=1)

    def forward(self, X_reshaped):
        X_conv = self.conv(X_reshaped)

        return X_conv

    ### END YOUR CODE

