class Agent(object):

    def __init__(self):
        self.curr_act = 200  # Dummy action        

    def episode_end(self, env_id):
        pass

    def act(self, state, reward, done):
        return self.curr_act
