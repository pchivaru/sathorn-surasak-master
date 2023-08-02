
class Agent(object):
    MIN_DUR = 5
    MAX_DUR = 300
    MAX_GAP = 3

    def __init__(self):
        self.curr_act = 100  # Dummy action        
        '''self.num_actions = num_actions
        
        self.last_change = 0'''

    def episode_end(self, env_id):
        pass
        '''
        self.curr_act = 100
        self.last_change = 0'''

    def act(self, state, reward, done):
        '''#return -1  # turns on SUMO automatic control
        if reward != -1 and self.num_actions != 3:
            print('Auto change')
            self.curr_act = reward
            if done == False:
                self.last_change = state[0][1] - 5 # -5

        if state[0][1] == -1:
            raise Exception('Actuated can not handle all phases currently.')
        elapsed = state[0][1] - self.last_change
        #print(elapsed, state[0][0])
        if elapsed >= self.MIN_DUR and (state[0][0] >= self.MAX_GAP or elapsed > self.MAX_DUR):
            print('Change')
            self.last_change = state[0][1]
            if self.curr_act == 100:
                self.curr_act = 107
            elif self.curr_act == 107:
                self.curr_act = 101
            elif self.curr_act == 101:
                self.curr_act = 100'''
        return self.curr_act
