import csv
import math
import numpy as np
import sys
import copy
import time
import pandas as pd

def ForwardSelection():
    return

def BackwardElimination():
    return

def main():
    print('Welcome to Group 61\'s Feature Selection Algorithm.')

    num_features = int(input('Please enter total number of features: '))

    algo = int(input('Type the number of algorithm you want to run. \n'
                     '\n1) Forward Selection'
                     '\n2) Backward Elimination\n\n'))
    
    if (algo == 1):
        ForwardSelection()
    elif (algo == 2):
        BackwardElimination()