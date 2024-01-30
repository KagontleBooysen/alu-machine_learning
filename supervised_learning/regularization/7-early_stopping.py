#!/usr/bin/env python3
"""
A function that determines if you should stop gradient descent early
"""
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    A function that determines if you should stop
    gradient descent early
    """
    stop = False
    if opt_cost - cost > threshold:
        count = 0
    else:
        count = count + 1
    if count == patience:
        stop = True
    return (stop, count)
