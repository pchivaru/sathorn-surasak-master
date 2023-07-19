import numpy as np
from enum import Enum

#Bangkok
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

#Texas
'''class Phase(Enum):
    W_L = [1]
    E = [2]
    N_L = [3]
    S = [4]
    E_L = [5]
    W = [6]
    S_L = [7]
    N = [8]
    R = []
    WW_L = W + W_L
    EE_L = E + E_L
    NN_L = N + N_L
    SS_L = S + S_L'''

# 1 Duplicate action is created when iterating over permissive left phases, purge them
def check_unique_action(action_sets, new_action_set):
    for action_set in action_sets:
        if new_action_set == action_set:
            return False
    return True

def get_all_actions(ew_upper_ring, ew_lower_ring, ns_upper_ring, ns_lower_ring):
    actions = []
    action_sets = []
    for upper in ew_upper_ring:
        for lower in ew_lower_ring:
            set_value = set(upper.value + lower.value)
            if check_unique_action(action_sets, set_value):
                action_sets.append(set_value)
                actions.append([upper, lower])
    for upper in ns_upper_ring:
        for lower in ns_lower_ring:
            set_value = set(upper.value + lower.value)
            if check_unique_action(action_sets, set_value):
                action_sets.append(set_value)
                actions.append([upper, lower])
    print(actions)
    return np.asarray(actions)

#Bangkok
ew_upper_ring = [Phase.E]
ew_lower_ring = [Phase.WL]
ns_upper_ring = [Phase.N_LN_R, Phase.N_L, Phase.S_R]
ns_lower_ring = [Phase.SLS_R, Phase.SL, Phase.N_R]

#Texas
'''ew_upper_ring = [Phase.E, Phase.W_L]
ew_lower_ring = [Phase.W, Phase.E_L]
ns_upper_ring = [Phase.NN_L, Phase.N, Phase.S_L]
ns_lower_ring = [Phase.SS_L, Phase.S, Phase.N_L]'''

get_all_actions(ew_upper_ring, ew_lower_ring, ns_upper_ring, ns_lower_ring)