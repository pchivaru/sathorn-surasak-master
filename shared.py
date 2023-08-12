import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "1"    # Disable GPU
#sys.path.append('baselines')
sys.path.append('traci_env')
import traci_env
import tensorflow as tf
#from baselines.common.models import register

OBS_SPACE = (17, 5, 1)
ACT_SPACE = 8

BATCH_SIZE = 32
DISCOUNT = 0.9
UPDATE_RATE = 500
EPSILON = 0.05
EXPLORE = 0.25
ALPHA = 0.6
EPS = 0.000001

demands = ['low_demand.csv']
rush = [33900]
dead = [0]


