#!/usr/bin/env python

import random
import copy
from .Constants import Phase
import traci
import os

import time

data_interval = 300 # The time interval (seconds) for each line of data in the demand file. 5 minutes in UTDOT logs (https://udottraffic.utah.gov/ATSPM)
left_on_red = True # Do we allow vehicles to turn right on red
min_phase_time = 3 # the minimal time interval between signal change (including yellow phase)


def create_sim(sumoBinary, id, path):
    traci.start([sumoBinary,
                 "-c", path+'Data'+id+os.sep+"intersection.sumocfg",
                 "--tripinfo-output", path+'Data'+id+os.sep+"tripinfo.xml",
                 "--device.emissions.probability", "1.0"
                 ], label='sim' + id)
    return traci.getConnection('sim'+id)


def run_sim(conn, intern, next_phase, actuation):
    if next_phase == -1:    # Native actuation
        intern.last_phase_change = conn.simulation.getTime()
        run_min_steps(conn, intern)
        intern.acted = True
        return
    if actuation and intern.current_phase == next_phase:
        intern.last_phase_change = conn.simulation.getTime()
        sim_time = conn.simulation.getTime()
        if sim_time % 5000 == 0:
            print(int(sim_time), 'seconds passed')

        vehicles = vehicle_generator(sim_time, intern.demand)

        for v in vehicles:
            conn.vehicle.add(str(intern.vehNr), v[0], typeID=v[1], departSpeed="max", departLane="best")
            intern.vehNr += 1

        conn.simulationStep()
        
        intern.acted = True
        return
    
    if intern.conflict_red is not None:
        set_phase(conn, intern.conflict_red)
        intern.conflict_red_count += 1
        if intern.conflict_red_count > 0:
            intern.conflict_red_count = 0
            intern.conflict_red = None
        return run_min_steps(conn, intern)
    elif intern.barrier_red:
        set_phase(conn, [])
        intern.barrier_red_count += 1
        if intern.barrier_red_count > 1:
            intern.barrier_red_count = 0
            intern.barrier_red = False
        return run_min_steps(conn, intern)

    if set_phase(conn, next_phase):  # If the chosen phases are applicable (no yellow transition is required)
        intern.yellow = False
        intern.acted = True
        intern.current_phase = next_phase  # chosen phases index
    else:
        intern.acted = False
        intern.barrier_red = all_red_required(intern.current_phase, next_phase)  # For an all red when crossing barrier
        intern.yellow = True
        if intern.yellow and not intern.barrier_red:
            intern.conflict_red = red_on_conflict(intern.current_phase, next_phase)  # Deal with conflicting signals

    return run_min_steps(conn, intern)


def run_min_steps(conn, intern):
    intern.last_phase_change = conn.simulation.getTime()
    while True:
        if conn.simulation.getTime() - intern.last_phase_change >= min_phase_time:
            break
        time = conn.simulation.getTime()
        if time % 5000 == 0:
            print(int(time), 'seconds passed')
        vehicles = vehicle_generator(time, intern.demand)
        #print(vehicles)
        for v in vehicles:
            conn.vehicle.add(str(intern.vehNr), v[0], typeID=v[1], departSpeed="max", departLane="best")
            intern.vehNr += 1
        conn.simulationStep()


def vehicle_generator(time, demand):  # time in seconds and demand as a list of entries from the demand file
    interval = int(time / data_interval)
    if interval > 288:  # Out of demand data
        return []


    sensor_routes = {'E1': ["Eastbound.L", "Eastbound.T"],
                    'S1': ["Southbound.L", "Southbound.R"],
                    'W1': ["Westbound.T"],
                    'N1':['Northbound.TR', 'Northbound.TL', "Northbound.R"]
                    }
    
    weights = {'E1': [1, 4],
                'S1': [2, 2],
                'W1': [1],
                'N1':[2, 1, 2]
                }

    data = demand.iloc[interval] # The data entry for the current time interval is retrieved
    vehicles = []
    for sensor in sensor_routes.keys():
        p = float(data[sensor]) / data_interval  # The probability of generating a vehicle per time step for each route in the current time interval
        if random.uniform(0, 1) < p:
            route = random.choices(sensor_routes[sensor], weights=weights[sensor])[0]
            vehicle = [route, "passenger"]
            vehicles.append(vehicle)
    return vehicles


def set_phase(conn, indexes):
    phases = [None] * 7 # why not 6??

    # Specify the green lanes in each phase
    phases[1] = [16, 17, 18, 19]  # phase Through. Westbound
    phases[2] = [2, 3]  # phase Right. Southbound
    phases[3] = [9, 10, 11, 12]  # phase Through, Left. Northbound  
    phases[4] = [4, 5, 6, 7, 8]  # phase Through, Left. Eastbound
    phases[5] = [13, 14, 15]  # phase Right. Northbound
    phases[6] = [0, 1]  # phase Left. Southbound

    left_turns = [0, 9] 
    prioritized_left = [4]
    next_signals = list("rrrrrrrrrrrrrrrrrrrr")  # init signals

    if left_on_red:
        for x in left_turns:
            next_signals[x] = 's'  # init right turns to green light "lower case 'g' mean that the stream has to decelerate"

    for x in prioritized_left:
        next_signals[x] = 'G'

    if protected(indexes):
        for i in indexes:
            for x in phases[i]:
                next_signals[x] = 'G'
    else:
        for i in indexes:
            if i in [2, 5]:
                for x in phases[i]:
                    next_signals[x] = 'g'
            else:
                for x in phases[i]:
                    next_signals[x] = 'G'

    current_signals = list(conn.trafficlight.getRedYellowGreenState("J21"))  # get the currently assigned lights
    yellow_phase = False
    for x in range(len(current_signals)):
        if current_signals[x] in ['G', 'g'] and next_signals[x] in ['r', 's']:  # Check if a yellow phase is needed
            yellow_phase = True

    if yellow_phase:
        for x in range(len(current_signals)):
            if current_signals[x] in ['G', 'g'] and next_signals[x] in ['r', 's']:  # If a yellow phase is needed then find which lanes
                # should be assigned yellow
                current_signals[x] = 'y'
        conn.trafficlight.setRedYellowGreenState("J21", ''.join(current_signals))
        return False
    else:
        conn.trafficlight.setRedYellowGreenState("J21", ''.join(next_signals))
        return True


protected_pairs = [[2, 5], [2, 6], [5, 2], [3, 5]]   # [1] & [4] the function of this is to use a G when posible for 2 and 5
def protected(current_phase):
    if current_phase in protected_pairs:
        return True


def red_on_conflict(current_phase, next_phase):  # Function
    red_conflict = False
    cphase = set(current_phase)
    nphase = set(next_phase)
    # Enumerate protected turn phase values
    N_R = [{Phase.N_R.value[0], Phase.S_R.value[0]}, {Phase.N_R.value[0], Phase.N_L.value[0]}]
    S_R = [{Phase.S_R.value[0], Phase.N_R.value[0]}, {Phase.S_R.value[0], Phase.SL.value[0]}]

    '''# If current phase is protected left
    if cphase in W_L:
        # and next is opposing through
        if Phase.E.value[0] in nphase:
            # insert a red phase for the left before switching
            red_conflict = True
    # If next phase is protected left
    if nphase in W_L:
        # and current phase is opposing through
        if Phase.E.value[0] in cphase:
            # insert a red phase for the through before switching
            red_conflict = True'''

    if cphase in N_R:
        if Phase.SL.value[0] in nphase:
            red_conflict = True
    if nphase in N_R:
        if Phase.SL.value[0] in cphase:
            red_conflict = True

    if red_conflict:
        return cphase.intersection(nphase)
    else:
        return None


def all_red_required(current_phase, next_phase):  # Function??
    NS = [2, 3, 5, 6]
    curr_ring = 0
    next_ring = 0
    if any(currp in NS for currp in current_phase):
        curr_ring = 1
    if any(nextp in NS for nextp in next_phase):
        next_ring = 1
    if curr_ring != next_ring:
        return True
    else:
        return False
