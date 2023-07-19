import shared
import numpy as np
import optparse
import gym
import multiprocessing as mp
from actuated import Agent as ActuatedAgent
from dqn import Agent as DQNAgent

import time

def main():
    optParser = optparse.OptionParser()
    optParser.add_option("--agent", action="store", default='dqn', help="actu, dqn, dqnpq, drsq, drhq, rppo")
    optParser.add_option("--file", action="store", default='0', type='int', help="demand file index (0-2)")
    optParser.add_option("--trials", action="store", default='1', type='int', help="number of trials") #30
    optParser.add_option("--eps", action="store", default='1', type='int', help="number of episodes per trial") #40
    optParser.add_option("--procs", action="store", default='1', type='int', help="number of processors to use")
    options, args = optParser.parse_args()
    print(options)

    if options.agent == 'actu':
        num_eps = 1
        num_trials = 1
    else:
        num_eps = options.eps
        num_trials = options.trials

    if options.procs == 1:
        run_trial(options.agent, options.file, num_eps, 0, render=True)
    else:
        pool = mp.Pool(processes=options.procs)
        for trial in range(num_trials):
            pool.apply_async(run_trial, args=(options.agent, options.file, num_eps, trial))
        pool.close()
        pool.join()


def run_trial(agent_type, file, num_eps, trial, render=False):
    mode = 'raw'
    if agent_type == 'actu':
        agent = ActuatedAgent(shared.OBS_SPACE, 3)
    elif agent_type == 'dqn':
        agent = DQNAgent(shared.OBS_SPACE, shared.ACT_SPACE)
    else:
        raise ValueError('Invalid agent type')

    for ep_cnt in range(num_eps):
        env = gym.make('traci-v0') 
        
        env.set_params(mode, trial, ep_cnt, shared.demands[file], shared.rush[file], shared.dead[file], render, agent_type)
        
        #print(env.demand_file)
        state, rew, reset = env.reset(), 0, False
       
        #if render: env.render()

        
        step = 0
        while not reset:
            step += 1
            #print('Trial', trial, 'Epoch:', ep_cnt, 'Step:', step)
            state, rew, reset, _ = env.step(action=agent.act(state, rew, reset))
            
        
        agent.episode_end(env.env_id)                                                                                               
        env.close()


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()
