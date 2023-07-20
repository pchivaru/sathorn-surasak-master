#!/usr/bin/env python3
import gym
from distutils.dir_util import copy_tree
import xml.etree.ElementTree as ElementTree
import os
import sys
from gym import spaces
if 'SUMO_HOME' in os.environ:
    print(os.environ['SUMO_HOME'])
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci
from sumolib import checkBinary

from . import SimThread
from . import Constants as C
from .Constants import LAD
from .Constants import IND
from .Constants import Phase
from .Constants import WL_EL, E_W, SL_NL, NNL_SSL

import numpy as np
import pandas as pd

import time

OBS_SPACE = spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32, shape=(20, 5, 1))
ACT_SPACE = spaces.Discrete(C.NUM_ACTIONS)


class SimInternal(object):
    def __init__(self, demand):
        self.demand = demand
        self.vehNr = 0
        self.current_phase = [1, 4]
        self.yellow = False
        self.conflict_red = None
        self.barrier_red = False
        self.last_phase_change = 0
        self.acted = False
        self.conflict_red_count = 0
        self.barrier_red_count = 0


class TraciEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.trial_id = 0
        self.eps_id = 0
        self.demand_file = None
        self.rush_hour = None
        self.dead_hour = None
        self.state_mode = None
        self.ew_upper_ring = [Phase.E]
        self.ew_lower_ring = [Phase.WL]
        self.ns_upper_ring = [Phase.N_LN_R, Phase.N_L, Phase.S_R]
        self.ns_lower_ring = [Phase.SLS_R, Phase.SL, Phase.N_R]
        self.actions = self.get_all_actions()
        self.prev_score = None
        self.upper_phase = None
        self.lower_phase = None
        self.demand = None
        self.sim = None
        self.sim_intern = None
        self.env_id = None
        self.sumoBinary = checkBinary('sumo')  #change to sumo-gui to show interface
        self.observation_space = OBS_SPACE
        self.action_space = ACT_SPACE
        self.steps = 0
        self.agent_type = None
        self.colab = 0
        self.path = None

    def open(self):
        self.env_id = str(self.trial_id) + '-' + str(self.eps_id)
        copy_tree(self.path+'Data', 'Data'+self.env_id)

        demand = pd.read_csv(self.path + self.demand_file)

        self.sim = SimThread.create_sim(self.sumoBinary, self.env_id, '')
        self.sim_intern = SimInternal(demand)
        self.prev_score = 0
        self.upper_phase = Phase.E
        self.lower_phase = Phase.WL
        self.steps = 0

    def close(self):
        self.sim.close()
        loss, loss_sq, trips, rh_loss, dh_loss, max_delay = self.get_delay('Data'+self.env_id+os.sep)
        CO_abs, CO2_abs, HC_abs, PMx_abs, NOx_abs, fuel_abs, elec_abs = self.get_contamination('Data'+self.env_id+os.sep)
        missed_trips = self.sim_intern.vehNr - trips
        if missed_trips > 0:
            print('MISSED TRIPS')
        penalized_loss = loss + missed_trips * C.MAX_WAIT

        max_queue = self.get_max_queue('Data'+self.env_id+os.sep)

        with open("results.csv", "a+") as f:
            f.write(str(self.trial_id) + ', ' + str(self.eps_id) + ', ' + str(trips) + ', ' + str(missed_trips) + ', ' +
                    str(loss) + ', ' + str(self.steps) + ', ' + str(rh_loss) + ', ' + str(dh_loss) + ', ' +
                    str(max_delay) + ', ' + str(max_queue) + ', '+ str(CO_abs) + ', ' + str(CO2_abs) + ', '+
                    str(HC_abs)+', '+str(PMx_abs)+', '+str(NOx_abs)+', '+str(fuel_abs)+', ' + str(elec_abs) + ', ' +
                    str(self.agent_type) + '\n')
        print(self.env_id, 'Completed iteration trips', trips, 'loss', loss, 'steps', self.steps, 'rush', rh_loss, 'dead', dh_loss)
        return penalized_loss

    def step(self, action):
        self.steps += 1
        reset = False
        
        print(f'Current time: {self.sim.simulation.getTime()}, Trial: {self.trial_id}, Epoch: {self.eps_id}')

        if self.sim.simulation.getTime() >= 50000: #55000!!!
            reset = True

        actuation = False
        if action >= 100:   # Actuated action
            action -= 100
            phase = self.get_phase_from_action(action)
            print('Phase:', phase)
            actuation = True
        else:
            phase = self.get_phase_from_action(action)

        rewards = []
        self.sim_intern.acted = False
        # Collect rewards from intermediate no-op states
        while not self.sim_intern.acted:
            SimThread.run_sim(self.sim, self.sim_intern, phase, actuation)            

            rewards.append(self.get_waiting_reduction())

        self.upper_phase = self.actions[action][0]
        self.lower_phase = self.actions[action][1]
        return self.get_state(), rewards, reset, {}

    def reset(self):
        #print(self.demand_file)
        if self.sim is not None: self.close()
        self.open()
        return self.get_state()
    
    def set_params(self, state_mode, trial_id, eps_id, demand_file, rush_hour, dead_hour, render, agent_type, colab):
        self.state_mode = state_mode
        self.trial_id = trial_id
        self.eps_id = eps_id
        self.demand_file = demand_file
        self.rush_hour = rush_hour
        self.dead_hour = dead_hour
        self.agent_type = agent_type
        self.colab = colab

        if render: self.sumoBinary = checkBinary('sumo-gui')

        if colab:
            self.path='/content/sathorn-surasak-master/traci_env/traci_env/envs/'
        else:
            self.path = os.path.dirname(os.path.abspath(__file__)) + os.sep

       
    '''def render(self, mode='human'):
        self.sumoBinary = checkBinary('sumo-gui')
        return'''

    def get_state(self):
        if self.state_mode == 'poly':
            return [self.raw_state(), self.poly_state()]
        elif self.state_mode == 'ind':
            return [self.inductor_state(), self.raw_state(), self.poly_state()]
        else:
            return self.raw_state()

    def inductor_state(self): # ADJUST !!!!
        current_signals = list(self.sim.trafficlight.getRedYellowGreenState("J21"))
        if current_signals == WL_EL:
            loops = ['W3', 'W4', 'E3', 'E4']
        elif current_signals == E_W:
            loops = ['W0', 'W1', 'W2', 'E0', 'E1', 'E2']
        elif current_signals == SL_NL:
            loops = ['S4', 'N4']
        elif current_signals == NNL_SSL:
            loops = ['N0', 'N1', 'N2', 'N3', 'N4', 'S0', 'S1', 'S2', 'S3', 'S4']
        else:
            return [-1, -1]     # Not configured

        lastDetected = []
        for loop in loops:
            lastDetected.append(self.sim.inductionloop.getTimeSinceDetection('inductionLoop.'+loop))
        return [min(lastDetected), self.sim.simulation.getTime()]

    def raw_state(self):
        current_signals = list(self.sim.trafficlight.getRedYellowGreenState("J21"))
        
        phase_light_idx = [16, 17, 18, 19, 2, 3, 9, 12, 4, 5, 6, 7, 8, 14, 15, 0, 1]
        state = np.zeros_like(C.INITIAL_STATE)
        lane_ids = []
        for i, phase in enumerate(LAD, 0):
            if i == 6: break  # Don't use NS or EW aggregate LADs
            for lane in phase.value:
                lane_ids.append(lane)

        for i, lane in enumerate(lane_ids, 0):
            if current_signals[phase_light_idx[i]] == 'r':
                state[i][0] = 0
            elif current_signals[phase_light_idx[i]] == 'y':
                state[i][0] = 0.3
            elif current_signals[phase_light_idx[i]] == 's':
                state[i][0] = 0.6
            elif current_signals[phase_light_idx[i]] == 'g':
                state[i][0] = 0.9
            elif current_signals[phase_light_idx[i]] == 'G':
                state[i][0] = 1

            approaching_vehicles, waiting, queue_length, speed = 0, 0, 0, 0
            for vehicle in self.sim.lanearea.getLastStepVehicleIDs(lane):
                veh_wait = self.sim.vehicle.getAccumulatedWaitingTime(vehicle)
                speed += self.sim.vehicle.getSpeed(vehicle)
                waiting += veh_wait
                if veh_wait > 0:
                    queue_length += 1
                else:
                    approaching_vehicles += 1
            total = queue_length + approaching_vehicles
            if total != 0:
                speed /= total

            state[i][1] = queue_length / C.MAX_VEHICLES
            state[i][2] = approaching_vehicles / C.MAX_VEHICLES
            state[i][3] = waiting / C.MAX_VEHICLES / C.MAX_WAIT
            state[i][4] = speed / C.MAX_SPEED
        state = np.expand_dims(state, axis=2)
        return state

    def poly_state(self):
        action_values = []
        for action in self.actions:
            upper, lower = action[0], action[1]
            action_inputs = []
            action_inputs += self.get_demand(LAD[upper.name])
            action_inputs += self.get_demand(LAD[lower.name])
            # Additional 3 variables to express added delays from signal changes
            if self.need_barrier_red(upper):
                # Need an all red to switch across barrier (e.g. NS -> EW)
                action_inputs += [1, 0, 0, 0]
            else:
                if self.need_conflict_red(upper, lower):
                    # Need red on conflicting lane (for protected rights)
                    action_inputs += [0, 1, 0, 0]
                else:
                    if self.upper_phase == upper and self.lower_phase == lower:
                        # No red light required
                        action_inputs += [0, 0, 1, 0]
                    else:
                        # Red light required
                        action_inputs += [0, 0, 0, 1]
            action_values.append(action_inputs)

        print(np.array(action_values))
        return np.array(action_values)

    def need_conflict_red(self, next_upper, next_lower):
        rights = [Phase.N_R, Phase.S_R]
        throughs = [Phase.E, Phase.WL, Phase.SL]
        if self.upper_phase in rights:
            if next_upper in throughs:
                return True
        if self.upper_phase in throughs:
            if next_upper in rights:
                return True
        if self.lower_phase in rights:
            if next_lower in throughs:
                return True
        if self.lower_phase in throughs:
            if next_lower in rights:
                return True
        return False

    def need_barrier_red(self, next_upper):
        if self.upper_phase in self.ns_upper_ring and next_upper in self.ew_upper_ring:
            return True
        elif self.upper_phase in self.ew_upper_ring and next_upper in self.ns_upper_ring:
            return True
        else:
            return False

    # Values should be 0-1, only if average waiting time over all lanes exceeds C.MAX_WAIT, then waiting is >1
    def get_demand(self, lad):
        approaching, waiting, queued, speed, avg_wait = 0, 0, 0, 0, 0
        for lane in lad.value:
            for vehicle in self.sim.lanearea.getLastStepVehicleIDs(lane):
                veh_wait = self.sim.vehicle.getAccumulatedWaitingTime(vehicle)
                speed += self.sim.vehicle.getSpeed(vehicle)
                waiting += veh_wait
                if veh_wait > 0:
                    queued += 1
                else:
                    approaching += 1
        queue_length = queued / len(lad.value) / C.MAX_VEHICLES
        total = queued + approaching
        if total != 0:
            speed = speed / total / C.MAX_SPEED

        queued = queued / C.MAX_VEHICLES
        approaching = approaching / C.MAX_VEHICLES
        waiting = waiting / C.MAX_VEHICLES / C.MAX_WAIT

        if queued != 0:
            avg_wait = waiting/queued

        return [queued, approaching, waiting, speed, queue_length, avg_wait]

    def get_phase_from_action(self, action_index):
        if action_index == -1: return -1    # Native actuation
        return self.actions[action_index][0].value + self.actions[action_index][1].value

    def get_all_actions(self):
        actions = []
        action_sets = []
        for upper in self.ew_upper_ring:
            for lower in self.ew_lower_ring:
                set_value = set(upper.value + lower.value)
                if self.check_unique_action(action_sets, set_value):
                    action_sets.append(set_value)
                    actions.append([upper, lower])
        for upper in self.ns_upper_ring:
            for lower in self.ns_lower_ring:
                set_value = set(upper.value + lower.value)
                if self.check_unique_action(action_sets, set_value):
                    action_sets.append(set_value)
                    actions.append([upper, lower])
        print(actions)
        return np.asarray(actions)

    # 1 Duplicate action is created when iterating over permissive left phases, purge them
    def check_unique_action(self, action_sets, new_action_set):
        for action_set in action_sets:
            if new_action_set == action_set:
                return False
        return True

    def get_waiting_reduction(self):
        cutoff, curr_score = 0, 0
        for phase in LAD:
            if cutoff == 6: break  # Use basic 6 detectors

            phase_score = 0
            for lane in phase.value:
                lane_score = 0
                for vehicle in self.sim.lanearea.getLastStepVehicleIDs(lane):
                    lane_score += self.sim.vehicle.getAccumulatedWaitingTime(vehicle)
                phase_score += (lane_score / C.MAX_VEHICLES)
            curr_score += phase_score
            cutoff += 1
        diff = self.prev_score - curr_score
        self.prev_score = curr_score
        return diff
    
    def get_contamination(self, data_path):
        tripinfos = ElementTree.parse(data_path+'tripinfo.xml').getroot().findall('tripinfo')
        trips = 0

        CO_abs, CO2_abs, HC_abs, PMx_abs, NOx_abs, fuel_abs, elec_abs = 0,0,0,0,0,0,0


        for tripinfo in tripinfos:
            emission = tripinfo.findall('emissions')[0]
            CO_abs += float(emission.get('CO_abs'))
            CO2_abs += float(emission.get('CO2_abs'))
            HC_abs += float(emission.get('HC_abs'))
            PMx_abs += float(emission.get('PMx_abs'))
            NOx_abs += float(emission.get('NOx_abs'))
            fuel_abs += float(emission.get('fuel_abs'))
            elec_abs += float(emission.get('electricity_abs'))
            trips+=1

        return CO_abs/trips, CO2_abs/trips, HC_abs/trips, PMx_abs/trips, NOx_abs/trips, fuel_abs/trips, elec_abs/trips

    def get_delay(self, data_path):
        loss = 0.0
        loss_sq = 0.0
        tripinfos = ElementTree.parse(data_path+'tripinfo.xml').getroot().findall('tripinfo')
        trips = len(tripinfos)
        rh_trips, rh_loss, dh_trips, dh_loss = 0, 0, 0, 0
        max_delay = 0
        for tripinfo in tripinfos:
            trip_loss = float(tripinfo.get('timeLoss'))
            if trip_loss > max_delay:
                max_delay = trip_loss
            loss += trip_loss
            loss_sq += trip_loss ** 2

            departed = float(tripinfo.get('depart'))
            if departed >= self.rush_hour and departed <= self.rush_hour+3600:
                rh_loss += trip_loss
                rh_trips += 1
            if departed >= self.dead_hour and departed <= self.dead_hour+3600:
                dh_loss += trip_loss
                dh_trips += 1
        if trips == 0: return 0, 0, 0, 0, 0, 0
        if rh_trips == 0 or dh_trips == 0:
            return loss / trips, loss_sq / trips, trips, 0, 0, max_delay
        return loss / trips, loss_sq / trips, trips, rh_loss / rh_trips, dh_loss / rh_trips, max_delay

    def get_max_queue(self, data_path):
        max_queue = 0
        cutoff = 0
        for phase in LAD:
            if cutoff == 6: break  # Use basic 6 detectors
            for lane in phase.value:
                lad_file = data_path + 'F' + lane + '.xml'
                intervals = ElementTree.parse(lad_file).getroot().findall('interval')
                for interval in intervals:
                    maxJamLengthInVehicles = float(interval.get('maxJamLengthInVehicles'))
                    if maxJamLengthInVehicles > max_queue:
                        max_queue = maxJamLengthInVehicles
            cutoff += 1
        return max_queue
