#!/usr/bin/env python3
"""
A function that updates the learning rate using inverse time decay in numpy
"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    A function that updates the learning rate using inverse time decay in numpy
    """
    new_alpha = alpha / (1 + decay_rate * (global_step // decay_step))
    return new_alpha
