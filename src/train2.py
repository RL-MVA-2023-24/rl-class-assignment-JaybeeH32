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

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
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
    
class ActorNetwork(torch.nn.Module):
    def __init__(self, hidden_dims, nb_layers) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(*[torch.nn.Linear(hidden_dims, hidden_dims) for i in range(nb_layers)])
        self.input_layer = torch.nn.Linear(6, hidden_dims)
        self.output_layer = torch.nn.Linear(hidden_dims, 4)
        
        self.initialize_weights()
    
    def forward(self, state):
        x = self.input_layer(state)
        x = F.relu(x)
        x = self.layers(x)
        x = F.relu(x)
        x = self.output_layer(x)
        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

class CriticNetwork(torch.nn.Module):
    def __init__(self, hidden_dims, nb_layers) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(*[torch.nn.Linear(hidden_dims, hidden_dims) for i in range(nb_layers)])
        self.input_layer = torch.nn.Linear(6 + 4, hidden_dims)
        self.output_layer = torch.nn.Linear(hidden_dims, 1)
        
        self.initialize_weights()
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.input_layer(x)
        x = torch.nn.functional.relu(x)
        x = self.layers(x)
        x = torch.nn.functional.relu(x)
        x = self.output_layer(x)
        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)       

class ProjectAgent:
    def __init__(self) -> None:
        
        self.capacity = 100000
        self.batch_size = 32
        
        self.gamma = 0.99
        self.tau = 0.005
        
        self.actor_lr = 0.0001
        self.critic_lr = 0.0001
        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.num_episodes = 80
        self.max_steps = 200
        self.actor_update_freq = 2
        
        self.actor_hidden_dims = 64
        self.critic_hidden_dims = 64
        self.actor_nb_layers = 3
        self.critic_nb_layers = 3
        

    def act(self, observation, use_random=False):
        if use_random:
            return np.random.randint(0, 4)
        else:
            observation = torch.Tensor(observation).to(self.device).unsqueeze(0)
            return np.argmax(self.actor_network(observation).cpu().detach().numpy())

    def save(self, path):
        with open(path, 'wb') as file:
            torch.save(self.actor_network, file)

    def load(self):
        checkpoint = torch.load('agent.pt')
        self.actor_network = ActorNetwork(self.actor_hidden_dims, self.actor_nb_layers).to(self.device)
        self.actor_network.load_state_dict(checkpoint.state_dict())
            
        
    def initialize_networks(self):
        actor_network = ActorNetwork(self.actor_hidden_dims, self.actor_nb_layers).to(self.device)
        critic1_network = CriticNetwork(self.critic_hidden_dims, self.critic_nb_layers).to(self.device)
        critic2_network = CriticNetwork(self.critic_hidden_dims, self.critic_nb_layers).to(self.device)
        return actor_network, critic1_network, critic2_network
    
    def initialize_target_networks(self, actor_network, critic1_network, critic2_network):
        target_actor_network = deepcopy(actor_network).to(self.device)
        target_critic1_network = deepcopy(critic1_network).to(self.device)
        target_critic2_network = deepcopy(critic2_network).to(self.device)
        return target_actor_network, target_critic1_network, target_critic2_network
    
    def update_target_networks(self, network, target_network, tau):
        for param, target_param in zip(network.parameters(), target_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
        
    def get_action(self, state, actor_network, epsilon):
        if random.random() < epsilon:
            return np.random.randint(0, 4)
        else:
            state = torch.Tensor(state).to(self.device).unsqueeze(0)
            action = actor_network(state)
            return torch.argmax(action).item()

    def train(self, disable_tqdm=False):
                # Initialize parameters
        actor_network, critic1_network, critic2_network = self.initialize_networks()
        target_actor_network, target_critic1_network, target_critic2_network = self.initialize_target_networks(actor_network, critic1_network, critic2_network)
        replay_buffer = ReplayBuffer(self.capacity, self.device)
        actor_optimizer = torch.optim.Adam(actor_network.parameters(), lr=self.actor_lr)
        critic1_optimizer = torch.optim.Adam(critic1_network.parameters(), lr=self.critic_lr)
        critic2_optimizer = torch.optim.Adam(critic2_network.parameters(), lr=self.critic_lr)
        
        cum_reward = []
        actor_loss = []
        critic1_loss = []
        critic2_loss = []

        # Training loop
        for episode in tqdm(range(self.num_episodes), disable=disable_tqdm):
            # Initialize state
            state, _ = env.reset()

            # Explore and take action
            action = self.get_action(state, actor_network, self.epsilon)
                        
            episode_cum_reward = 0
            actor_loss_episode = 0
            critic1_loss_episode = 0
            critic2_loss_episode = 0

            for step in range(self.max_steps):
                # Take action and observe next state, reward, etc.
                next_state, reward, done, _, _ = env.step(action)
                
                episode_cum_reward += reward

                # Store transition in replay buffer
                replay_buffer.append(state, action, reward, next_state, done)
                
                if len(replay_buffer) < self.batch_size:
                    continue

                # Sample a batch from replay buffer
                states, actions, rewards, next_states, dones = replay_buffer.sample(self.batch_size)
                
                actions = F.one_hot(actions.long(), num_classes=4).float()
                

                # Update critic networks
                with torch.no_grad():
                    # Target actions for stability
                    target_actions = target_actor_network(next_states)
                    target_Q1 = target_critic1_network(next_states, target_actions)
                    target_Q2 = target_critic2_network(next_states, target_actions)
                    target_Q = torch.min(target_Q1, target_Q2)
                    target_value = reward + (self.gamma * (1 - done) * target_Q)

                Q1 = critic1_network(states, actions)
                loss_Q1 = F.mse_loss(Q1, target_value)

                Q2 = critic2_network(states, actions)
                loss_Q2 = F.mse_loss(Q2, target_value)

                critic1_optimizer.zero_grad()
                loss_Q1.backward()
                critic1_optimizer.step()

                critic2_optimizer.zero_grad()
                loss_Q2.backward()
                critic2_optimizer.step()
                
                critic1_loss_episode += loss_Q1.item()
                critic2_loss_episode += loss_Q2.item()                

                # Update actor network (less frequently)
                if step % self.actor_update_freq == 0:
                    with torch.no_grad():
                        # Deterministic action for stability
                        policy_actions = actor_network(states)

                    Q = critic1_network(states, policy_actions)
                    loss_policy = -torch.mean(Q)

                    actor_optimizer.zero_grad()
                    loss_policy.backward()
                    actor_optimizer.step()
                    
                    actor_loss_episode += loss_policy.item()

                    # Update target networks (soft update)
                    self.update_target_networks(actor_network, target_actor_network, self.tau)
                    self.update_target_networks(critic1_network, target_critic1_network, self.tau)
                    self.update_target_networks(critic2_network, target_critic2_network, self.tau)

                # if done, print episode info
                if done or step == self.max_steps - 1:
                    cum_reward.append(episode_cum_reward)
                    actor_loss.append(actor_loss_episode / step)
                    critic1_loss.append(critic1_loss_episode / step)
                    critic2_loss.append(critic2_loss_episode / step)
                    
                    print("Episode ", '{:2d}'.format(episode), 
                        ", buffer size ", '{:4d}'.format(len(replay_buffer)), 
                        ", episode return ", '{:4.1f}'.format(episode_cum_reward), 
                        ", episode length ", '{:3d}'.format(step),
                        sep='')
                    break
                else:
                    action = self.get_action(state, actor_network, self.epsilon)
            
            if self.epsilon > self.epsilon_end:
                self.epsilon *= self.epsilon_decay
        self.actor_network = actor_network
        return cum_reward, actor_loss, critic1_loss, critic2_loss


if __name__ == "__main__":
    
    agent = ProjectAgent()
        
    cum_reward, actor_loss, critic1_loss, critic2_loss = agent.train()
    
    agent.save("agent.pt")
    
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    ax[0, 0].plot(np.arange(agent.num_episodes), cum_reward)
    ax[0, 0].set_title("Cumulative Reward")
    ax[0, 1].plot(np.arange(agent.num_episodes), actor_loss)
    ax[0, 1].set_title("Actor Loss")
    ax[1, 0].plot(np.arange(agent.num_episodes), critic1_loss)
    ax[1, 0].set_title("Critic 1 Loss")
    ax[1, 1].plot(np.arange(agent.num_episodes), critic2_loss)
    ax[1, 1].set_title("Critic 2 Loss")
    plt.savefig("training.png")