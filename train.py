import trainer

train = trainer.Trainer([4,4,2])

'''train.train(
    replay_capacity = 100
    episodes = 50
    max_timesteps = 400
    epsilon = 0.1
    batch_size = 50
    gamma = 0.8
)
'''

train.train(replay_capacity = 100,
episodes = 50,
max_timesteps = 50,
epsilon = 0.1,
batch_size = 3,
gamma = 0.8)

