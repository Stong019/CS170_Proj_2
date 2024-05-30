import csv
import math
import numpy as np
import sys
import copy
import time
import pandas as pd


def main():
    print('Welcom to Group 61\'s Feature Selection Algorithm.')

    #fn = #file name
    #fn = open(f, 'r') #opens file

    algo = int(input('Type the number of algorithm you want to run. \n'
                     '\n1) Forward Selection'
                     '\n2) Backward Elimination\n\n'))
