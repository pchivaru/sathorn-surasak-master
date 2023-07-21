import pickle

with open('Tests/replay.buffer', 'rb') as replay_buffer_file:  
    storage = pickle.load(replay_buffer_file)

print(storage[50])