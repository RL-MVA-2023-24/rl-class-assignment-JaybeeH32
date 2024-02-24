from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import pickle
import torch
import random
import torch.nn.functional as F
from copy import deepcopy
import torch.nn as nn

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity)
        self.data = []
        self.index = 0
        self.device = device
        
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    
    def __len__(self):
        return len(self.data)

class QNetwork(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims, nb_layers):
        super(QNetwork, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dims, hidden_dims)] +
                                    [nn.Linear(hidden_dims, hidden_dims) for _ in range(nb_layers-1)])
        self.output_layer = nn.Linear(hidden_dims, output_dims)
    
    def forward(self, state):
        x = state
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x

class ProjectAgent:
    def __init__(self):
        self.input_dims = 6
        self.output_dims = 4
        self.hidden_dims = 256
        self.nb_layers = 3
        self.learning_rate = 0.0001
        self.gamma = 0.99
        
        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        
        self.buffer_capacity = 100_000
        self.batch_size = 32
        self.target_update_freq = 4
        
        self.max_steps = 200
        self.num_episodes = 200
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = QNetwork(self.input_dims, self.output_dims, self.hidden_dims, self.nb_layers).to(self.device)
        self.target_q_network = deepcopy(self.q_network).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(self.buffer_capacity, self.device)
    
    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.output_dims)
        state = torch.Tensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()
    
    def train(self):
        cum_reward = []
        qnetwork_loss = []
        for episode in tqdm(range(self.num_episodes)):
            state, _ = env.reset()
            episode_cum_reward = 0
            episode_qnetwork_loss = 0
            for step in range(self.max_steps):
                action = self.select_action(state, self.epsilon)
                next_state, reward, done, _, _ = env.step(action)
                episode_cum_reward += reward
                self.replay_buffer.append(state, action, reward, next_state, done)
                
                if len(self.replay_buffer) > self.batch_size:
                    loss = self.update_q_network()
                    episode_qnetwork_loss += loss
                
                if done:
                    break
                
                state = next_state
            
            cum_reward.append(episode_cum_reward / step)
            qnetwork_loss.append(episode_qnetwork_loss/ step)
            
            if episode % self.target_update_freq == 0:
                self.update_target_network()
            if self.epsilon > self.epsilon_end:
                self.epsilon *= self.epsilon_decay
            
            print(f"Episode {episode + 1}, Cumulative Reward: {episode_cum_reward}")
        
        return cum_reward, qnetwork_loss
    
    def update_q_network(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.long().to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        
        q_values = self.q_network(states)
        next_q_values = self.target_q_network(next_states)
        
        target_q_values = rewards + (1 - dones) * self.gamma * torch.max(next_q_values, dim=1)[0]
        
        target_q_values = target_q_values.unsqueeze(1)
        q_values = q_values.gather(1, actions.unsqueeze(1))
        
        loss = self.loss_fn(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())        

    def act(self, observation, use_random=False):
        if use_random:
            return np.random.randint(0, 4)
        else:
            state = torch.Tensor(observation).to(self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state)
            return q_values.argmax().item()

    def save(self, path):
        with open(path, 'wb') as file:
            torch.save(self.q_network, file)

    def load(self):
        checkpoint = torch.load('agent3.pt')
        self.q_network = QNetwork(self.input_dims, self.output_dims, self.hidden_dims, self.nb_layers).to(self.device)
        self.q_network.load_state_dict(checkpoint.state_dict())
            
        

if __name__ == "__main__":
    
    agent = ProjectAgent()
        
    cum_reward, qnetwork_loss = agent.train()
    
    agent.save("agent3.pt")
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    ax[0].plot(np.arange(agent.num_episodes), cum_reward)
    ax[0].set_title("Cumulative Reward")
    ax[1].plot(np.arange(agent.num_episodes), qnetwork_loss)
    ax[1].set_title("Actor Loss")
    plt.savefig("training3.png")