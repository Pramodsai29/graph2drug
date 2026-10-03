
import sys
import os

# Add Original DGCAN to Python path
sys.path.append("/content/D-GCAN/DGCAN")

import preprocess as pp
from DGCAN import MolecularGraphNeuralNetwork, Trainer, Tester

import numpy as np
import torch
import timeit

print("Successfully imported Original DGCAN!")
