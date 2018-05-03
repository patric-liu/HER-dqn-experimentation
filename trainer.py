import numpy as np 
import gym
import network
import random

class Trainer(object):

    def __init__(self, network_shape):
        self.replay_memory = []
        self.env = gym.make('CartPole-v1')
        self.net = network.Network( network_shape )

    def train(self, replay_capacity, episodes, max_timesteps, epsilon,
        batch_size, gamma):
        for episode in range(episodes):
            #print('beginning episode ',episode)
            state = self.env.reset()

            reward_tracker = 0

            for t in range(max_timesteps):
                print('episode: ',episode,'  timestep: ',t, '  reward: ',reward_tracker)
                if episode%100 == 0:
                    self.env.render()

                 # decide on an action
                if np.random.binomial(1,epsilon) == 1:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.net.feedforward(self.format_lists(state)))

                # take the action
                n_state, reward, done, info = self.env.step(action)
                reward = reward/10

                # record observations
                self.replay_memory.append([state,action,reward,n_state])
                if len(self.replay_memory) > replay_capacity:
                    del self.replay_memory[0]

                state = n_state

                reward_tracker += reward

                self.update_network(batch_size, gamma)

                if done:
                    break

                    
    def update_network(self, batch_size, gamma):

        training_data = []

        try:
            replays = random.sample(self.replay_memory, batch_size)
        except ValueError:
            replays = random.sample(self.replay_memory, len(self.replay_memory))

        for replay in replays:
            # input is state sj
            #print(replay[1],'action')
            #print(replay[2],'reward')
            state_j = self.format_lists(replay[0])
            # initialize target distribution to be equal to output given sj
            target = self.net.feedforward(state_j)
            #print(target,'Q(s)')
            # state sj+1
            state_jp1 = self.format_lists(replay[3])
            # expected value of each action at state sj+1
            q_jplus1 = self.net.feedforward(state_jp1)
            # target value based on bellman equation
            target_value = replay[2] + gamma * q_jplus1[np.argmax(q_jplus1)]
            # set target distribution to be same as output given sj except
            # with a different expected value for action aj
            target[replay[1]] = target_value
            # append training data as tuple for 'supervised' training
            
            tuple_ = (state_j,self.format_lists(target))
            
            #print(target,'r + Q(s+1)')
            training_data.append(tuple_)
            

        self.net.SGD(training_data, # Data to train on
        1, 			        # epochs
        len(training_data), # mini_batch_size
        0.1, 			   # eta
        10) 			   # lmbda

    def format_lists(self,input_list):
        # turns a list into an np.array with shape (len,1)
        input_array = np.zeros( (len(input_list),1) )
        for idx, val in enumerate(input_list):
            input_array[idx] = val

        return input_array
