from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import pickle

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self, config) -> None:
        
        self.nb_actions = 4
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.99
        self.iterations = config['iterations'] if 'iterations' in config.keys() else 200
        self.regressor = config['regressor'] if 'regressor' in config.keys() else 'RF'
        
        self.nb_envs = config['collecting_ens'] if 'collecting_ens' in config.keys() else 20
        self.horizon = config['collecting_horizon'] if 'collecting_horizon' in config.keys() else 200
    
    def act(self, observation, use_random=False):
        if use_random:
            print('playing random !')
            return np.random.choice(np.arange(self.nb_actions))
        Q2 = np.zeros((1, self.nb_actions))
        for a2 in range(self.nb_actions):
            A2 = a2 
            S2A2 = np.append(observation, A2, axis=1)
            Q2[:, a2] = self.Qfunctions[-1].predict(S2A2)
        max_Q2 = np.max(Q2, axis=1)
        return max_Q2

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self.Qfunctions[-1], file)

    def load(self):
        with open('agent.pkl', 'rb') as file:
            Qfunction = pickle.load(file)
        self.Qfunctions = [Qfunction]
        
    def get_regressor(self):
        if self.regressor == 'RF':
            return RandomForestRegressor()
        elif self.regressor == 'GB':
            return GradientBoostingRegressor()
        else:
            raise ValueError(f"Type of regressor {self.regressor} is not recognized.")
    
    def train(self, disable_tqdm=False):
        S, A, R, S2, D = self.train_dataset
        nb_samples = S.shape[0]
        Qfunctions = []
        SA = np.append(S,A,axis=1)
        for iter in tqdm(range(self.iterations), disable=disable_tqdm):
            if iter==0:
                value=R.copy()
            else:
                Q2 = np.zeros((nb_samples, self.nb_actions))
                for a2 in range(self.nb_actions):
                    A2 = a2 * np.ones((S.shape[0], 1))
                    S2A2 = np.append(S2, A2, axis=1)
                    Q2[:, a2] = Qfunctions[-1].predict(S2A2)
                max_Q2 = np.max(Q2, axis=1)
                value = R + self.gamma * (1 - D) * max_Q2
            Q = self.get_regressor()
            Q.fit(SA, value)
            print(f"Iteration {iter} - MSE loss: {self.evaluate_model(Q)}")
            Qfunctions.append(Q)
        self.Qfunctions = Qfunctions
    
    def evaluate_model(self, Qfunction):
        S, A, R, S2, D = self.test_dataset
        nb_samples = S.shape[0]
        SA = np.append(S,A,axis=1)
        Q2 = np.zeros((nb_samples, self.nb_actions))
        for a2 in range(self.nb_actions):
            A2 = a2 * np.ones((S.shape[0], 1))
            S2A2 = np.append(S2, A2, axis=1)
            Q2[:, a2] = Qfunction[-1].predict(S2A2)
        max_Q2 = np.max(Q2, axis=1)
        value = R + self.gamma * (1 - D) * max_Q2
        
        predict_value = Qfunction.predict(SA)
        return mean_squared_error(value, predict_value)
        
    
    def collect_samples(self, disable_tqdm=False, print_done_states=True):
        print("Starting to collect samples.")
        for iter in tqdm(range(self.nb_envs), disable=disable_tqdm):
            if iter ==0:
                env = HIVPatient()
            else:
                env = HIVPatient(domain_randomization=True)
            s, _ = env.reset()
            S = []
            A = []
            R = []
            S2 = []
            D = []
            for _ in range(self.horizon):
                a = env.action_space.sample()
                s2, r, done, trunc, _ = env.step(a)
                S.append(s)
                A.append(a)
                R.append(r)
                S2.append(s2)
                D.append(done)
                if done or trunc:
                    s, _ = env.reset()
                    if done and print_done_states:
                        print(f"env {iter} done!")
                else:
                    s = s2
        S = np.array(S)
        A = np.array(A).reshape((-1,1))
        R = np.array(R)
        S2= np.array(S2)
        D = np.array(D)
        n = len(S)
        self.train_dataset = (S[:int(n * 0.8)], A[:int(n * 0.8), :], R[:int(n * 0.8)], S2[:int(n * 0.8)], D[:int(n * 0.8)])
        self.test_dataset = (S[int(n * 0.8):], A[int(n * 0.8):, :], R[int(n * 0.8):], S2[int(n * 0.8):], D[int(n * 0.8):])

        print("Done collecting samples.")

if __name__ == "__main__":
    
    config = {
        'gamma': 0.99,
        'iterations': 200,
        'regressor': 'RF',
        
        'collecting_envs': 20,
        'collecting_horizon': 200,
        
    }
    agent = ProjectAgent(config)
    
    agent.collect_samples()
    
    agent.train()
    
    agent.save("agent.pkl")