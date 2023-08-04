from enum import Enum
import numpy as np

# Environment settings
NUM_ACTIONS = 11
NUM_MEASUREMENTS = 14
MAX_VEHICLES = 27
MAX_SPEED = 16.67
MAX_WAIT = 300  # Just an estimate of max waiting time to scale input
MIN_REW = 5

"""Parameters without command line input options"""

per = False
ddqn = False
"""DQN"""
PRETRAIN = False
FILENAME = 'traffic.net'
INITIAL_STATE = np.zeros((17, 5))
EXPLORE = 5
Q_LEARNING_RATE = 0.001
DISCOUNT = 0.8
UPDATE_RATE = 500
BATCH_SIZE = 32
REPLAY_CAPACITY = 100000
# Prioritized experience replay parameters
ALPHA = 0.2
EPSILON = 0.001
# Importance sampling - beta should reach 1.0 by convergence, increased by eta each transition recorded
BETA = 0.6
ETA = 0.00025

# ADJUST !!!!
WL_EL = ['s', 'r', 'r', 'r', 'r', 's', 'r', 'r', 'r', 'G', 'G', 's', 'r', 'r', 'r', 'r', 's', 'r', 'r', 'r', 'G', 'G']
E_W = ['s', 'r', 'r', 'r', 'r', 'G', 'G', 'G', 'G', 'r', 'r', 's', 'r', 'r', 'r', 'r', 'G', 'G', 'G', 'G', 'r', 'r']
SL_NL = ['s', 'r', 'r', 'r', 'G', 's', 'r', 'r', 'r', 'r', 'r', 's', 'r', 'r', 'r', 'G', 's', 'r', 'r', 'r', 'r', 'r']
NNL_SSL = ['G', 'G', 'G', 'G', 'g', 's', 'r', 'r', 'r', 'r', 'r', 'G', 'G', 'G', 'G', 'g', 's', 'r', 'r', 'r', 'r', 'r']


class Phase(Enum):
    E = [1]
    N_R = [2]
    SL = [3]
    WL = [4]
    S_R = [5]
    N_L = [6]
    R = [] # ???
    N_LN_R = N_L + N_R
    SLS_R = SL + S_R


pre = 'laneAreaDetector.'
class LAD(Enum):
    __order__ = 'E N_R SL WL S_R N_L N_LN_R SLS_R'     # Required to be compatible with python 2
    E = [pre + 'E0', pre + 'E1', pre + 'E2', pre + 'E3']
    N_R = [pre + 'N2', pre + 'N3']
    SL = [pre + 'S0', pre + 'S1']                                              
    WL = [pre + 'W0', pre + 'W1', pre + 'W2', pre + 'W3', pre + 'W4']
    S_R = [pre + 'S2', pre + 'S3']                          # Possible conflict in S1 (12, 13) different phase same line, for now we eliminate S1 from S_R
    N_L = [pre + 'N0', pre + 'N1']
    N_LN_R = N_L + N_R
    SLS_R = SL + S_R

ind_pre = 'inductionLoop.'
class IND(Enum):
    __order__ = 'E N_R SL WL S_R N_L N_LN_R SLS_R'     # Required to be compatible with python 2
    E = [ind_pre + 'E0', ind_pre + 'E1', ind_pre + 'E2', ind_pre + 'E3']
    N_R = [ind_pre + 'N2', ind_pre + 'N3']
    SL = [ind_pre + 'S0', ind_pre + 'S1']                                              
    WL = [ind_pre + 'W0', ind_pre + 'W1', ind_pre + 'W2', ind_pre + 'W3', ind_pre + 'W4']
    S_R = [ind_pre + 'S2', ind_pre + 'S3']                          # Possible conflict in S1 (12, 13) different phase same line, for now we eliminate S1 from S_R
    N_L = [ind_pre + 'N0', ind_pre + 'N1']
    N_LN_R = N_L + N_R
    SLS_R = SL + S_R
