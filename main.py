import shared
import numpy as np
import optparse
import gym
import multiprocessing as mp
from actuated import Agent as ActuatedAgent
from static import Agent as StaticAgent
from dqn import Agent as DQNAgent

#import time
import re
#import tracemalloc

def main():
    optParser = optparse.OptionParser()
    optParser.add_option("--agent", action="store", default='dqn', help="actu, static, dqn")
    optParser.add_option("--file", action="store", default='0', type='int', help="demand file index (0-2)")
    optParser.add_option("--trials", action="store", default='1', type='int', help="number of trials") #30
    optParser.add_option("--eps", action="store", default='40', type='int', help="number of episodes per trial") #40
    optParser.add_option("--procs", action="store", default='1', type='int', help="number of processors to use")
    optParser.add_option("--colab", action="store", default='0', type='int', help="running in colab or not")
    optParser.add_option("--envid", action="store", default='0-0', help="starting environment id")
    options, args = optParser.parse_args()
    print(options)

    if options.agent == 'actu':
        num_eps = 1
        num_trials = 30 #30
    else:
        num_eps = options.eps
        num_trials = options.trials

    results = re.search(r'(\d+)-(\d+)', options.envid)

    trial_0 = int(results[1])
    eps_0 = int(results[2])

    if options.procs == 1:
        run_trial(options.agent, options.file, num_eps, trial_0, options.colab, eps_0, render=False)
    else:
        pool = mp.Pool(processes=options.procs)
        for trial in range(trial_0, num_trials):
            pool.apply_async(run_trial, args=(options.agent, options.file, num_eps, trial, options.colab))
        pool.close()
        pool.join()


def run_trial(agent_type, file, num_eps, trial, colab, eps_0=0, render=False):
    mode = 'raw'
    if agent_type == 'actu':
        agent = ActuatedAgent()
    elif agent_type == 'static':
        agent = StaticAgent()
    elif agent_type == 'dqn':
        agent = DQNAgent(shared.OBS_SPACE, shared.ACT_SPACE)
        if eps_0 != 0:
            #load the previous model weights and replay buffer!
            agent.load(str(trial)+'-'+str(eps_0-1))  
    else:
        raise ValueError('Invalid agent type')

    #tracemalloc.start()
    for ep_cnt in range(eps_0, num_eps):
        env = gym.make('traci-v0') 
        
        env.set_params(mode, trial, ep_cnt, shared.demands[file],
                        shared.rush[file], shared.dead[file], render,
                        agent_type, colab)
        
        #print(env.demand_file)
        state, rew, reset = env.reset(), 0, False
       
        #if render: env.render()

        
        step = 0
        while not reset:
            step += 1
            
            '''if step % 60 == 0:
                #Tracking memory allocation
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                print('Top 10 files that allocate the most memory')
                for stat in top_stats[:10]:
                    print(stat)

                #Print replay buffer
                print(agent.replay._storage[50])'''

            state, rew, reset, _ = env.step(action=agent.act(state, rew, reset))
            #print('Epsilon:', agent.explore)
            
        
        agent.episode_end(env.env_id)                                                                                               
        env.close()


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()
