import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from osim.env import L2M2019Env

def dic2array(dic):
    arr = []
    for key, val in dic.items():
        if type(val) == float:
            arr.append(val)
        elif type(val) == list:
            for e in val:
                arr.append(e)
        elif type(val) == dict:
            for k, v in val.items():
                arr.append(v)
    arr = np.array(arr)
    return arr

def obs_preprocessor(observation):
    v_tgt_field = observation['v_tgt_field'].flatten()
    pelvis = dic2array(observation['pelvis'])
    l_leg = dic2array(observation['l_leg'])
    r_leg = dic2array(observation['r_leg'])
    
    state = np.concatenate([v_tgt_field, pelvis, l_leg, r_leg], axis=0)    
    return state

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class Agent:
    def __init__(self):
        # for local test
        self.actor = Actor(339, 22, torch.ones(22))
        self.actor.load_state_dict(torch.load('112065802_hw4_data'))

    def act(self, observation):
        state = obs_preprocessor(observation)
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).cpu().data.numpy().flatten().tolist()
        for i in range(len(action)):
            if action[i] > 1:
                action[i] = 1
            elif action[i] < 0: 
                action[i] = 0
        return action

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(4e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.not_done[ind]).to(device)
        )

if __name__ == '__main__':
    env = L2M2019Env(difficulty=2, visualize=False)
    observation, done = env.reset(), False

    state = obs_preprocessor(observation)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # max_action = float(env.action_space.high[0])
    
    agent = Agent()
    episodes = 50
    total_reward = 0

    for e in range(episodes):
        observation = env.reset()
        episode_reward = 0
        print(f'Episode {e}')
        while True:
            # env.render()
            action = agent.act(observation)
            
            next_obs, reward, done, _ = env.step(action)
            next_state = obs_preprocessor(next_obs)
            
            observation = next_obs
            state = obs_preprocessor(observation)
            episode_reward += reward
            
            if done:
                break
        
        print(f'Episode reward in episode {e}: {episode_reward}')
        total_reward += episode_reward
    
    avg_reward = total_reward/50
    print(f'Average reward: {avg_reward}')