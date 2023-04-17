import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv #not needed unless you use ReadCsv()
import time
import gym
import pygame
import warnings
from matplotlib.colors import LinearSegmentedColormap
from gym.envs.toy_text.frozen_lake import generate_random_map
import seaborn as sns
from decorators.decorators import print_runtime
import functools
import math
import random
from tqdm import tqdm
from itertools import chain
from IPython.display import clear_output
from time import sleep
from matplotlib import animation
from gym.envs.registration import register
from gym.envs.toy_text.taxi import TaxiEnv
from custom_env import CustomTaxiEnv
import pandas as pd
from itertools import zip_longest






def print_runtime(func):
    @functools.wraps(func)
    def wrapper_print_runtime(*args, **kwargs):
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        running_time = end - start
        print("runtime = %.2f seconds" % running_time)
        return value
    return wrapper_print_runtime


# from https://wiki.python.org/moin/PythonDecoratorLibrary
def add_to(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        setattr(func, args[0].__name__, args[0])
        return func
    return decorator


# from https://realpython.com/
def debug(func):
    """Print the function signature and return value"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]                      # 1
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
        signature = ", ".join(args_repr + kwargs_repr)           # 3
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")           # 4
        return value
    return wrapper_debug


def plot_learning_curve(lrn_crv,algo_name,dataset=1):
    crv = np.array(lrn_crv)
    x_data = crv[:,0] #train_percentages
    y_data = crv[:,1:] # train and test scores
    figure_name = algo_name + "Dataset-" + str(dataset) + ".jpg"
    
    fig, ax = plt.subplots()
    title = algo_name + " : Learning Curve - "+ "Dataset-" + str(dataset)
    plt.title(title)
    plt.xlabel("Training Data Percentage (%) --> ")
    plt.ylabel("Score (%) --> ")
    plt.plot(x_data, y_data)
    plt.legend(['train score', 'test score'])
    plt.savefig(figure_name)
    #plt.show()










def GetData(dataset = 1):
    """
    with open("./fetal_health.csv", 'r') as file:
      csvreader = csv.reader(file)
      for row in csvreader:
        print(row)
    """
    if dataset == 1:
        name = "fetal_health"
    else:
        name = "mobile_price"
    name = name + ".csv"
    df = pd.read_csv(name)
    data = df.to_numpy() # rows and columns just like .csv
    X = data[:,0:-1]
    y = np.transpose(data)[-1]
    #print("data",data)
    #print("X",X)
    #print("y",y)
    return (X,y,name)





def ReadCsv():    
    with open("./fetal_health.csv", 'r') as file:
      csvreader = csv.reader(file)
      for row in csvreader:
        print(row)
    


class Plots:
    @staticmethod
    def grid_world_policy_plot(data, label):

        print("shape",data.shape)
        print("shape[0]",data.shape[0])
        s = math.sqrt(data.shape[0])
        
        if not math.modf(math.sqrt(len(data)))[0] == 0.0:
            warnings.warn("Grid map expected.  Check data length")
        else:
            data = np.around(np.array(data).reshape((int(s), int(s))), 2) #change shape
            df = pd.DataFrame(data=data)
            my_colors = ((0.0, 0.0, 0.0, 1.0), (0.8, 0.0, 0.0, 1.0), (0.0, 0.8, 0.0, 1.0), (0.0, 0.0, 0.8, 1.0))
            cmap = LinearSegmentedColormap.from_list('Custom', my_colors, len(my_colors))
            ax = sns.heatmap(df, cmap=cmap, linewidths=1.0, annot=False) 
            colorbar = ax.collections[0].colorbar
            colorbar.set_ticks([.4, 1.1, 1.9, 2.6])
            colorbar.set_ticklabels(['Left', 'Down', 'Right', 'Up'])
            plt.title(label)
            
            name="policy_plot_"+label+".png"
            plt.savefig(name)
            plt.show()

    @staticmethod
    def taxi_policy_plot(data, label):

        print("shape",data.shape)
        print("shape[0]",data.shape[0])
        s = math.sqrt(data.shape[0])
        
        if not math.modf(math.sqrt(len(data)))[0] == 0.0:
            warnings.warn("Grid map expected.  Check data length")
        else:
            data = np.around(np.array(data).reshape((int(s), int(s))), 2) #change shape
            df = pd.DataFrame(data=data)
            my_colors = ((0.0, 0.0, 0.0, 1.0), (0.8, 0.0, 0.0, 1.0), (0.0, 0.8, 0.0, 1.0), (0.0, 0.0, 0.8, 1.0))
            cmap = LinearSegmentedColormap.from_list('Custom', my_colors, len(my_colors))
            ax = sns.heatmap(df, cmap=cmap, linewidths=1.0, annot=False) 
            colorbar = ax.collections[0].colorbar
            colorbar.set_ticks([.4, 1.1, 1.9, 2.6])
            colorbar.set_ticklabels(['Left', 'Down', 'Right', 'Up'])
            plt.title(label)
            
            name="policy_plot_"+label+".png"
            plt.savefig(name)
            plt.show()

    @staticmethod
    def grid_values_heat_map(data, label):
        s = math.sqrt(data.shape[0])
        if not math.modf(math.sqrt(len(data)))[0] == 0.0:
            warnings.warn("Grid map expected.  Check data length")
        else:
            data = np.around(np.array(data).reshape((int(s), int(s))), 2)
            df = pd.DataFrame(data=data)
            sns.heatmap(df, annot=True).set_title(label)
            
            name="heat_map_"+label+".png"
            plt.savefig(name)
            plt.show()

    @staticmethod
    def v_iters_plot(data, label, name=""):
        df = pd.DataFrame(data=data)
        df.columns = [label]
        sns.set_theme(style="whitegrid")
        title = label + " v Iterations"
        sns.lineplot(x=df.index, y=label, data=df).set_title(title)
        
        name="iters_plot_"+label+".png"
        plt.savefig(name)
        plt.show()


class TestEnv:
    def __init__(self):
        pass

    @staticmethod
    def test_env(env, render=True, n_iters=10, pi=None, user_input=False, convert_state_obs=lambda state, done: state):
        """
        Parameters
        ----------------------------
        env {OpenAI Gym Environment}:
            MDP problem

        render {Boolean}:
            openAI human render mode
        
        n_iters {int}, default = 10:
            Number of iterations to simulate the agent for
        
        pi {lambda}:
            Policy used to calculate action value at a given state
        
        user_input {Boolean}:
            Prompt for letting user decide which action to take at a given state
        
        convert_state_obs {lambda}:
            The state conversion utilized in BlackJack ToyText problem.
            Returns three state tuple as one of the 280 converted states.

        
        Returns
        ----------------------------
        test_scores {list}:
            Log of reward at the end of each iteration
        """
        if render:
            # unwrap env and and reinit in 'human' render_mode
            env_name = env.unwrapped.spec.id
            env = gym.make(env_name, render_mode='human')
        n_actions = env.action_space.n
        test_scores = np.full([n_iters], np.nan)
        for i in range(0, n_iters):
            state, info = env.reset()
            done = False
            state = convert_state_obs(state, done)
            total_reward = 0
            while not done:
                if user_input:
                    # get user input and suggest policy output
                    print("state is %i" % state)
                    print("policy output is %i" % pi(state))
                    while True:
                        action = input("Please select 0 - %i then hit enter:\n" % int(n_actions-1))
                        try:
                            action = int(action)
                        except ValueError:
                            print("Please enter a number")
                            continue
                        if 0 <= action < n_actions:
                            break
                        else:
                            print("please enter a valid action, 0 - %i \n" % int(n_actions - 1))
                else:
                    action = pi(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_state = convert_state_obs(next_state, done)
                state = next_state
                total_reward = reward + total_reward
            test_scores[i] = total_reward
        env.close()
        return test_scores


class Planner:
    def __init__(self, P):
        self.P = P

    @print_runtime
    def value_iteration(self, gamma=1.0, n_iters=1000, theta=1e-10):
        """
        PARAMETERS:

        gamma {float}:
            Discount factor

        n_iters {int}:
            Number of iterations

        theta {float}:
            Convergence criterion for value iteration.
            State values are considered to be converged when the maximum difference between new and previous state values is less than theta.
            Stops at n_iters or theta convergence - whichever comes first.


        RETURNS:

        V {numpy array}, shape(possible states):
            State values array

        V_track {numpy array}, shape(n_episodes, nS):
            Log of V(s) for each iteration

        pi {lambda}, input state value, output action value:
            Policy mapping states to actions.
        """
        V = np.zeros(len(self.P), dtype=np.float64)
        V_track = np.zeros((n_iters, len(self.P)), dtype=np.float64)
        i = 0
        converged = False
        deltas = []
        iter_conv = 0
        while i < n_iters-1 and not converged:
            i += 1
            Q = np.zeros((len(self.P), len(self.P[0])), dtype=np.float64)
            for s in range(len(self.P)):
                for a in range(len(self.P[s])):
                    for prob, next_state, reward, done in self.P[s][a]:
                        Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
            delta = np.max(np.abs(V - np.max(Q, axis=1)))
            deltas.append(delta)
            if delta < theta:
                converged = True
                iter_conv = i
            V = np.max(Q, axis=1)
            V_track[i] = V
        if not converged:
            warnings.warn("Max iterations reached before convergence.  Check theta and n_iters.  ")
        # Explanation of lambda:
        # def pi(s):
        #   policy = dict()
        #   for state, action in enumerate(np.argmax(Q, axis=1)):
        #       policy[state] = action
        #   return policy[s]
        pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        return V, V_track, pi, deltas, iter_conv

    @print_runtime
    def policy_iteration(self, gamma=1.0, n_iters=50, theta=1e-10):
        """
        PARAMETERS:

        gamma {float}:
            Discount factor

        n_iters {int}:
            Number of iterations

        theta {float}:
            Convergence criterion for policy evaluation.
            State values are considered to be converged when the maximum difference between new and previous state
            values is less than theta.


        RETURNS:

        V {numpy array}, shape(possible states):
            State values array

        V_track {numpy array}, shape(n_episodes, nS):
            Log of V(s) for each iteration

        pi {lambda}, input state value, output action value:
            Policy mapping states to actions.
        """
        random_actions = np.random.choice(tuple(self.P[0].keys()), len(self.P))
        # Explanation of lambda:
        # def pi(s):
        #   policy = dict()
        #   for state, action in enumerate(np.argmax(Q, axis=1)):
        #       policy[state] = action
        #   return policy[s]
        pi = lambda s: {s: a for s, a in enumerate(random_actions)}[s]
        # initial V to give to `policy_evaluation` for the first time
        V = np.zeros(len(self.P), dtype=np.float64)
        V_track = np.zeros((n_iters, len(self.P)), dtype=np.float64)
        i = 0
        converged = False
        deltas=[]
        iter_conv = 0
        while i < n_iters and not converged:
            print("running iteration ",i)
            i += 1
            old_pi = {s: pi(s) for s in range(len(self.P))}
            V = self.policy_evaluation(pi, V, gamma, theta)
            V_track[i-1] = V
            if i>1:
                delta = np.max(np.abs(V_track[i-2] - V))
                deltas.append(delta)
            pi = self.policy_improvement(V, gamma)
            if old_pi == {s: pi(s) for s in range(len(self.P))}:
                converged = True
                iter_conv = i

        if not converged:
            warnings.warn("Max iterations reached before convergence.  Check n_iters.")
        """
        pi_track=[]
        i=0
        n_states = 400
        for v in V_track:
            interim_pi = self.policy_improvement(v, gamma)
            new_pi = list(map(lambda x: interim_pi(x), range(n_states)))
            s = int(math.sqrt(n_states))
            print("PI grid world policy plot", np.array(new_pi))  
            title = str(i)+"_FL_PI_"+"_Grid World Policy"
            Plots.grid_world_policy_plot(np.array(new_pi), title) 
            i+=1
        """

        return V, V_track, pi, deltas, iter_conv

    def policy_evaluation(self, pi, prev_V, gamma=1.0, theta=1e-10):
        deltas = []
        i=0
        while True:
            V = np.zeros(len(self.P), dtype=np.float64)            
            for s in range(len(self.P)):
                for prob, next_state, reward, done in self.P[s][pi(s)]:
                    #print("s,prob,next_state,reward,done", s,prob,next_state,reward,done)
                    V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
                    
                    #print("i=",i)
                #print("finished s= ",s)
            #print("not stuck any more")
            #print("prev",prev_V)
            #print("V",V)
            delta = np.max(np.abs(prev_V - V))
            #print("-------------- delta is ", delta)
            #print("-------------- theta is ", theta)
            #print("i",i)
            i+=1
            #if delta < theta or i>=5 :
            if delta < theta:
                print("breaking")
                break
            prev_V = V.copy()
        print("policy eval done")
        
        return V

    def policy_improvement(self, V, gamma=1.0):
        Q = np.zeros((len(self.P), len(self.P[0])), dtype=np.float64)
        for s in range(len(self.P)):
            for a in range(len(self.P[s])):
                for prob, next_state, reward, done in self.P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
        # Explanation of lambda:
        # def new_pi(s):
        #   policy = dict()
        #   for state, action in enumerate(np.argmax(Q, axis=1)):
        #       policy[state] = action
        #   return policy[s]
        new_pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        return new_pi




class RL:
    def __init__(self, env):
        self.env = env
        self.callbacks = MyCallbacks()
        self.render = False

    @staticmethod
    def decay_schedule(init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10):
        """
        Parameters
        ----------------------------
        init_value {float}:
            Initial value of the quantity being decayed
        
        min_value {float}:
            Minimum value init_value is allowed to decay to
            
        decay_ratio {float}:
            The exponential factor exp(decay_ratio).
            Updated decayed value is calculated as 
        
        max_steps {int}:
            Max iteration steps for decaying init_value
        
        log_start {array-like}, default = -2:
            Starting value of the decay sequence.
            Default value starts it at 0.01
        
        log_base {array-like}, default = 10:
            Base of the log space.
        
        
        Returns
        ----------------------------
        values {array-like}, shape(max_steps):
            Decay values where values[i] is the value used at i-th step
        """
        decay_steps = int(max_steps * decay_ratio)
        rem_steps = max_steps - decay_steps
        values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
        values = (values - values.min()) / (values.max() - values.min())
        values = (init_value - min_value) * values + min_value
        values = np.pad(values, (0, rem_steps), 'edge')
        return values

    @print_runtime
    def q_learning(self,
                   nS=None,
                   nA=None,
                   convert_state_obs=lambda state, done: state,
                   gamma=.99,
                   init_alpha=0.5,
                   min_alpha=0.01,
                   alpha_decay_ratio=0.5,
                   init_epsilon=0.8,
                   min_epsilon=0.01,
                   epsilon_decay_ratio=0.9,
                   n_episodes=100000):
        """
        Parameters
        ----------------------------
        nS {int}:
            Number of states
        
        nA {int}:
            Number of available actions
            
        convert_state_obs {lambda}:
            The state conversion utilized in BlackJack ToyText problem.
            Returns three state tuple as one of the 280 converted states.
        
        gamma {float}, default = 0.99:
            Discount factor
        
        init_alpha {float}, default = 0.5:
            Learning rate
        
        min_alpha {float}, default = 0.01:
            Minimum learning rate
        
        alpha_decay_ratio {float}, default = 0.5:
            Decay schedule of learing rate for future iterations
        
        init_epsilon {float}, default = 0.1:
            Initial epsilon value for epsilon greedy strategy.
            Chooses max(Q) over available actions with probability 1-epsilon.
        
        min_epsilon {float}, default = 0.1:
            Minimum epsilon. Used to balance exploration in later stages.
        
        epsilon_decay_ratio {float}, default = 0.9:
            Decay schedule of epsilon for future iterations
            
        n_episodes {int}, default = 10000:
            Number of episodes for the agent


        Returns
        ----------------------------
        Q {numpy array}, shape(nS, nA):
            Final action-value function Q(s,a)

        pi {lambda}, input state value, output action value:
            Policy mapping states to actions.

        V {numpy array}, shape(nS):
            State values array

        Q_track {numpy array}, shape(n_episodes, nS, nA):
            Log of Q(s,a) for each episode

        pi_track {list}, len(n_episodes):
            Log of complete policy for each episode
        """
        rewards_all_episodes = []
        if nS is None:
            nS=self.env.observation_space.n
        if nA is None:
            nA=self.env.action_space.n
        pi_track = []
        Q = np.zeros((nS, nA), dtype=np.float64)
        Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
        # Explanation of lambda:
        # def select_action(state, Q, epsilon):
        #   if np.random.random() > epsilon:
        #       return np.argmax(Q[state])
        #   else:
        #       return np.random.randint(len(Q[state]))
        select_action = lambda state, Q, epsilon: np.argmax(Q[state]) \
            if np.random.random() > epsilon \
            else np.random.randint(len(Q[state]))
        alphas = RL.decay_schedule(init_alpha,
                                min_alpha,
                                alpha_decay_ratio,
                                n_episodes)
        epsilons = RL.decay_schedule(init_epsilon,
                                  min_epsilon,
                                  epsilon_decay_ratio,
                                  n_episodes)
        deltas = []
        for e in tqdm(range(n_episodes), leave=False):
            reward_current_episode = 0
            self.callbacks.on_episode_begin(self)
            self.callbacks.on_episode(self, episode=e)
            state, info = self.env.reset()
            done = False
            state = convert_state_obs(state, done)
            while not done:
                if self.render:
                    warnings.warn("Occasional render has been deprecated by openAI.  Use test_env.py to render.")
                action = select_action(state, Q, epsilons[e])
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                if truncated:
                    warnings.warn("Episode was truncated.  Bootstrapping 0 reward.")
                done = terminated or truncated
                self.callbacks.on_env_step(self)
                next_state = convert_state_obs(next_state,done)
                td_target = reward + gamma * Q[next_state].max() * (not done)
                td_error = td_target - Q[state][action]
                Q[state][action] = Q[state][action] + alphas[e] * td_error
                state = next_state
                reward_current_episode += reward
            Q_track[e] = Q
            
            if e>=1:
                delta = np.max(np.abs(Q_track[e-1] - Q)) 
                deltas.append(delta)
            
            pi_track.append(np.argmax(Q, axis=1))
            self.render = False
            self.callbacks.on_episode_end(self)
            rewards_all_episodes.append(reward_current_episode)

        V = np.max(Q, axis=1)
        # Explanation of lambda:
        # def pi(s):
        #   policy = dict()
        #   for state, action in enumerate(np.argmax(Q, axis=1)):
        #       policy[state] = action
        #   return policy[s]
        pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        return Q, V, pi, Q_track, pi_track, deltas, rewards_all_episodes

    @print_runtime
    def sarsa(self,
              nS=None,
              nA=None,
              convert_state_obs=lambda state, done: state,
              gamma=.99,
              init_alpha=0.5,
              min_alpha=0.000001,
              alpha_decay_ratio=0.5,
              init_epsilon=1.0,
              min_epsilon=0.4,
              epsilon_decay_ratio=0.05,
              n_episodes=10000):
        """
        Parameters
        ----------------------------
        nS {int}:
            Number of states
        
        nA {int}:
            Number of available actions
            
        convert_state_obs {lambda}:
            The state conversion utilized in BlackJack ToyText problem.
            Returns three state tuple as one of the 280 converted states.
        
        gamma {float}, default = 0.99:
            Discount factor
        
        init_alpha {float}, default = 0.5:
            Learning rate
        
        min_alpha {float}, default = 0.01:
            Minimum learning rate
        
        alpha_decay_ratio {float}, default = 0.5:
            Decay schedule of learing rate for future iterations
        
        init_epsilon {float}, default = 0.1:
            Initial epsilon value for epsilon greedy strategy.
            Chooses max(Q) over available actions with probability 1-epsilon.
        
        min_epsilon {float}, default = 0.1:
            Minimum epsilon. Used to balance exploration in later stages.
        
        epsilon_decay_ratio {float}, default = 0.9:
            Decay schedule of epsilon for future iterations
            
        n_episodes {int}, default = 10000:
            Number of episodes for the agent


        Returns
        ----------------------------
        Q {numpy array}, shape(nS, nA):
            Final action-value function Q(s,a)

        pi {lambda}, input state value, output action value:
            Policy mapping states to actions.

        V {numpy array}, shape(nS):
            State values array

        Q_track {numpy array}, shape(n_episodes, nS, nA):
            Log of Q(s,a) for each episode

        pi_track {list}, len(n_episodes):
            Log of complete policy for each episode
        """
        if nS is None:
            nS = self.env.observation_space.n
        if nA is None:
            nA = self.env.action_space.n
        pi_track = []
        Q = np.zeros((nS, nA), dtype=np.float64)
        visits = np.zeros((nS, nA), dtype=np.float64)
        Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
        # Explanation of lambda:
        # def select_action(state, Q, epsilon):
        #   if np.random.random() > epsilon:
        #       return np.argmax(Q[state])
        #   else:
        #       return np.random.randint(len(Q[state]))
        select_action = lambda state, Q, epsilon: np.argmax(Q[state]) \
            if np.random.random() > epsilon \
            else np.random.randint(len(Q[state]))
        alphas = RL.decay_schedule(init_alpha,
                                min_alpha,
                                alpha_decay_ratio,
                                n_episodes)
        epsilons = RL.decay_schedule(init_epsilon,
                                  min_epsilon,
                                  epsilon_decay_ratio,
                                  n_episodes)

        for e in tqdm(range(n_episodes), leave=False):
            self.callbacks.on_episode_begin(self)
            self.callbacks.on_episode(self, episode=e)
            state, info = self.env.reset()
            done = False
            state = convert_state_obs(state, done)
            action = select_action(state, Q, epsilons[e])
            while not done:
                if self.render:
                    warnings.warn("Occasional render has been deprecated by openAI.  Use test_env.py to render.")
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                if truncated:
                    warnings.warn("Episode was truncated.  Bootstrapping 0 reward.")
                done = terminated or truncated
                self.callbacks.on_env_step(self)
                next_state = convert_state_obs(next_state, done)
                next_action = select_action(next_state, Q, epsilons[e])
                td_target = reward + gamma * Q[next_state][next_action] * (not done)
                td_error = td_target - Q[state][action]
                Q[state][action] = Q[state][action] + alphas[e] * td_error
                state, action = next_state, next_action
            Q_track[e] = Q
            pi_track.append(np.argmax(Q, axis=1))
            self.render = False
            self.callbacks.on_episode_end(self)

        V = np.max(Q, axis=1)
        # Explanation of lambda:
        # def pi(s):
        #   policy = dict()
        #   for state, action in enumerate(np.argmax(Q, axis=1)):
        #       policy[state] = action
        #   return policy[s]
        pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        return Q, V, pi, Q_track, pi_track

class Callbacks:
    """
    Base class.
    """
    def __init__(self):
        pass

    def on_episode_begin(self, caller):
        pass

    def on_episode_end(self, caller):
        pass

    def on_episode(self, caller, episode):
        pass

    def on_env_step(self, caller):
        pass


class MyCallbacks(Callbacks):
    """
    To create a callback, override one of the callback functions in the child class MyCallbacks.
    """
    def __init__(self):
        pass

    def on_episode(self, caller, episode):
        """
        Parameters
        ----------------------------
        caller (RL type): Calling object

        episode {int}: Current episode from caller
        """
        # do things on specific episodes
        pass

    def on_episode_begin(self, caller):
        """
        Parameters
        ----------------------------
        caller (RL type): Calling object
        """
        # do things on episode begin
        pass

    def on_episode_end(self, caller):
        """
        Parameters
        ----------------------------
        caller (RL type): Calling object
        """
        # do things on episode end
        pass

    def on_env_step(self, caller):
        """
        Parameters
        ----------------------------
        caller (RL type): Calling object
        """
        # do things on env. step
        pass


def plot_deltas(deltas, algo="",problem=""):
    title = problem + "_" + algo +"_" + "Convergence - Deltas from one iteration to another"
    plt.title(title)
    plt.xlabel("Iterations --> ")
    plt.ylabel("Delta --> ")
    plt.legend(problem)
    plt.plot(deltas)
    print("deltas",deltas)

    figname = problem +"_" + algo + "_delta_plot.png"
    plt.savefig(figname)
    plt.show()


def small_taxi_vi(any_env, gamma=1, n_iters=1000, theta=1e-10, problem=""):
    np.random.seed(42)

    #Run VI
    start = time.time()
    V, V_track, pi,deltas,iters = Planner(any_env.P).value_iteration(gamma, n_iters, theta)
    end= time.time()
    vitime = end-start

    #delta plot
    plot_deltas(deltas, "vi",problem)

    #policy plot
    n_states = any_env.observation_space.n
    new_pi = list(map(lambda x: pi(x), range(n_states)))
    s = int(math.sqrt(n_states))
    print("VI grid world policy plot", np.array(new_pi))

    if problem != "Taxi" or problem != "SmallTaxi":
        title = problem + "_VI_" +"_Grid World Policy"
        Plots.grid_world_policy_plot(np.array(new_pi), title)

    if problem == "Taxi" or problem=="SmallTaxi":
        # do this
        print("taxi rendering in progress")
    else:
        any_env.reset(seed=42)
        img = any_env.render()    
        plt.imshow(img)
        title = problem + "_VI_.png"
        plt.savefig(title)
        plt.show()

    
    # VI/PI v_iters_plot
    """
    max_value_per_iter = np.amax(V_track, axis=1)
    name = problem + "_VI_" + "_Value"
    Plots.v_iters_plot(max_value_per_iter, name)
    """
    return V, V_track, pi,deltas,iters, vitime


def small_taxi_pi(any_env, gamma=1, n_iters=50, theta=1e-10, problem = ""):

    #Run PI
    start = time.time()
    V, V_track, pi, deltas,iters = Planner(any_env.P).policy_iteration(gamma, n_iters, theta)
    end= time.time()
    pitime = end-start

    #delta plot
    plot_deltas(deltas, "pi",problem)

    #policy plot
    n_states = any_env.observation_space.n
    new_pi = list(map(lambda x: pi(x), range(n_states)))
    s = int(math.sqrt(n_states))
    print("PI grid world policy plot", np.array(new_pi))

    if problem != "Taxi" and problem != "SmallTaxi":
        title = problem + "_PI_"+"_Grid World Policy"
        Plots.grid_world_policy_plot(np.array(new_pi), title)

        #render
        any_env.reset()
        img = any_env.render()
        print("img",img)
        plt.imshow(img)
        title = problem + "_PI_.png"
        plt.savefig(title)
        plt.show()

    # v max iterations plot
    
    max_value_per_iter = np.amax(V_track, axis=1)
    name = problem + "_PI_" + "_Value"
    Plots.v_iters_plot(max_value_per_iter, name)
    
    return V, V_track, pi,deltas, iters, pitime
    

def vi(any_env, gamma=1, n_iters=1000, theta=1e-10, problem="", plot=True):
    np.random.seed(42)

    start = time.time()
    #Run VI
    V, V_track, pi,deltas, iters = Planner(any_env.env.P).value_iteration(gamma, n_iters, theta)
    end= time.time()
    vitime = end-start

    #delta plot
    #plot_deltas(deltas, "vi",problem)

    #policy plot
    n_states = any_env.env.observation_space.n
    new_pi = list(map(lambda x: pi(x), range(n_states)))
    s = int(math.sqrt(n_states))
    print("VI grid world policy plot", np.array(new_pi))

    if problem != "Taxi":
        title = problem + "_VI_" +"_Grid World Policy"
        if plot==True:
            Plots.grid_world_policy_plot(np.array(new_pi), title)

    if problem == "Taxi":
        # do this
        print("taxi rendering in progress")
    else:
        if plot==True:
            any_env.reset(seed=42)
            img = any_env.render()    
            plt.imshow(img)
            title = problem + "_VI_.png"
            plt.savefig(title)
            plt.show()
    """
    print("All the Vs: ")
    i=0
    for v in V_track:
        
        print(i,"\n",v)
        Plots.grid_values_heat_map(v, str(i)+" State Values")
        i+=1
    """
    
    
    # VI/PI v_iters_plot
    max_value_per_iter = np.amax(V_track, axis=1)
    name = problem + "_VI_" + "_Value"
    if plot==True:
        Plots.v_iters_plot(max_value_per_iter, name)
    return V, V_track, pi,deltas, iters, vitime

   



def pi(any_env, gamma=1, n_iters=50, theta=1e-10, problem = "", plot=True):

    #Run PI
    start = time.time()
    V, V_track, pi, deltas, iters = Planner(any_env.env.P).policy_iteration(gamma, n_iters, theta)
    end = time.time()
    pitime= (end-start)

    #delta plot
    #plot_deltas(deltas, "pi",problem)

    #policy plot
    n_states = any_env.env.observation_space.n
    new_pi = list(map(lambda x: pi(x), range(n_states)))
    s = int(math.sqrt(n_states))
    print("PI grid world policy plot", np.array(new_pi))

    if problem != "Taxi":
        title = problem + "_PI_"+"_Grid World Policy"
        if plot==True:
            Plots.grid_world_policy_plot(np.array(new_pi), title)

        #render
        if plot==True:
            any_env.reset()
            img = any_env.render()
            print("img",img)
            plt.imshow(img)
            title = problem + "_PI_.png"
            plt.savefig(title)
            plt.show()

    # v max iterations plot
    max_value_per_iter = np.amax(V_track, axis=1)
    name = problem + "_PI_" + "_Value"
    if plot==True:
        Plots.v_iters_plot(max_value_per_iter, name)

    if plot==True:
        print("All the Vs: ")
        i=0
        for v in V_track:
            
            print(i,"\n",v)
            Plots.grid_values_heat_map(v, str(i)+" State Values")
            i+=1
    
    return V, V_track, pi,deltas, iters, pitime

def q(any_env,problem=""):
    n_episodes = 10000
    nS=None,
    nA=None,
    convert_state_obs=lambda state, done: state,
    gamma=.99,
    init_alpha=0.5,
    min_alpha=0.01,
    alpha_decay_ratio=0.5,
    init_epsilon=1,
    min_epsilon=0.00001,
    epsilon_decay_ratio=0.9,
    n_episodes=100000
    start = time.time()
    Q, V, pi, Q_track, pi_track, deltas, rewards = RL(any_env.env).q_learning()
    end = time.time()
    qtime = end - start
    """
                   nS,
                   nA,
                   convert_state_obs,
                   gamma,
                   init_alpha,
                   min_alpha,
                   alpha_decay_ratio,
                   init_epsilon,
                   min_epsilon,
                   epsilon_decay_ratio,
                   n_episodes)
    """
    
    n_states = any_env.env.observation_space.n
    new_pi = list(map(lambda x: pi(x), range(n_states)))
    s = int(math.sqrt(n_states))
    print("Q- learning grid world policy plot", np.array(new_pi))
    title = problem + "_Q-learning_"+"_Grid World Policy"
    Plots.grid_world_policy_plot(np.array(new_pi), title)

    any_env.reset(seed=42)
    img = any_env.render()
    print("img",img)
    plt.imshow(img)
    title = problem + "problem_Q.png"
    plt.savefig(title)
    plt.show()


    #max_value_per_iter = np.amax(Q_track, axis=1)
    #Plots.v_iters_plot(max_value_per_iter, "Value")

    #delta plot
    plot_deltas(deltas, "Q-learning",problem)


    rewards_per_1000_epi = np.split(np.array(rewards),n_episodes/1000)
    title = problem + "_Q_Convergence - Rewards"
    plt.title(title)
    plt.xlabel("Episodes --> ")
    plt.ylabel("Rewards --> ")
    plt.plot(rewards)
    print("rewards",rewards)
    name=problem+"q_rewards.png"
    plt.savefig(name)
    plt.show()
    

    count=1000
    for r in rewards_per_1000_epi:
        print(count, ": ",str(sum(r/1000)))
        count += 1000

    Plots.grid_values_heat_map(V, "State Values")
    print("Q", Q)

    return qtime



def GetFrozenLakeProblem(map_no):
    if map_no == 1:
        mapx = ["SFFF", "FHFH", "FFFH", "HFFG"]
    elif map_no == 2:
        mapx = ["SFFF", "FFHF", "FFFF", "HFFG"]
    elif map_no == 3:
        mapx = ["SFFFFFFFFFFFFFFFHFFF","FHFFFFFFFFFFFFFFFFFF","FFFFFFFFFFFFFFFFFFFF","FFFFFFFFFFFFFFFFFFFF","FFFFFFFHFFFFFFFFFFFF","FFFFFFFFFFFFFFFFFFFF","FFFFFFFFFFFFFFFFFFFF","FFFFFFFFFFHFFFFFFFFF","FFFFFFFFFFFFFFFFFFFF","FFFFFFFFFFFFFFFFFHFF","FFFFFFFFFFFFFFFFFFFF","FFFFFFFFFFFFFFFFFFFF","FFFFFFFFFFFFFFFFFFFF","FFFFFFFFFFFFFFFFFFFF","FFFFFFFFFFFFFFFFFFFF","FFFFFFFFFFFFFFFFFFFF","FFHFFFFFFFFFFFFFFFFF","FFFFFFFFFFFFFFFFFFFF","FFFFFFFFFFFFFFFFFFFF","HFFFFFFFFFFFFFFFFFFG"]
    elif map_no == 4:
        mapx = ["SFFFFFF", "FHFHFFF", "FFFFFFF", "FFFFFFF", "FFFFFFF", "FFFHFFF", "HFFFFFG"]
    elif map_no == 5:
        mapx = ["SFFFFFFFFFH", "FHFHFFFFFFF", "FFFFFFFFFFF", "FFFFFFFFFFF", "FFFFFFFHHHH","FFFFFFFFFFF","FFFFFFFFFFF","FFFFFFFFFFF","FFFFFFFFFFF", "FFFHFFFFFFF", "HFFFFFFFFFG"]
    frozen_lake = gym.make('FrozenLake-v1', desc=mapx,render_mode="rgb_array")
    cells = list(chain.from_iterable([[cell for cell in row] for row in mapx])) #get all cells as a 1D list
    holes = [i for i, element in enumerate(cells) if element=='H']

    goal_state = len(cells)-1

    return (frozen_lake,holes,goal_state)


def GetFrozenLakeWithRewards(frozen_lake, holes, goal_state):
    hole_reward = -1
    goal_reward = 10   


    stateIndex=0
    for state,tp in frozen_lake.env.P.items():
        #print("state", state)
        #print("-----left down right up")
        directionIndex=0
        for direction,probs in tp.items():
            
            #print("---direction",direction)
            #print("---probs", probs)
            probIndex = 0
            for prob in probs:
                if prob[1] in holes: #prob[1] is new state
                    frozen_lake.env.P[stateIndex][directionIndex][probIndex] = (prob[0], prob[1], hole_reward, prob[3])
                if prob[1] == goal_state: #prob[1] is new state
                    frozen_lake.env.P[stateIndex][directionIndex][probIndex] = (prob[0], prob[1], goal_reward, prob[3])
                

                    
                #print("---------prob", frozen_lake.env.P[stateIndex][directionIndex][probIndex])
                probIndex += 1
            directionIndex += 1       
        #print("------------")
        stateIndex += 1

    return frozen_lake
    

def Display_VIPI_hyps(df, problem ="",labels=[]):
    plt.title("Iterations to converge vs Gamma - " + problem)
    plt.xlabel("Iterations -->")
    plt.ylabel("Deltas -->")
    plt.plot(df)
    plt.legend(labels)
    name = problem + "_deltas_vs_iterations.png"
    plt.savefig(name)
    plt.show()

    
def Frozen_Lake_Hyp():
    # Get the problem grid
    frozen_lake, holes, goal_state = GetFrozenLakeProblem(3) #1 is 4x4; #2 is another 4x4; #3 is 20x20

    # Modify the rewards in the problem grid
    frozen_lake = GetFrozenLakeWithRewards(frozen_lake, holes, goal_state)
    print("This is the P of the environment", frozen_lake.env.P)
    
    # Value Iteration
    gammas=[0.2, 0.4, 0.6, 0.8, 1.0]
    #gammas=[0.2, 0.4, 0.6, 0.8]
    
    n_iters=1000
    theta=1e-2
    iter_vis=[]
    delta_list=[]
    for gamma in gammas:
        Vv, Vv_track, vpi,deltasv, iter_vi = vi(frozen_lake, gamma, n_iters, theta, "FL_VI_gamma_" + str(gamma), False)
        iter_vis.append(iter_vi)
        delta_list.append(np.asarray(deltasv))
    deltas = np.array(list(zip_longest(*delta_list, fillvalue=0))).T 
    #print(deltas)
    print("shape", deltas.shape)

    df = pd.DataFrame()
    for i in range(len(deltas)):
        colname = "g=" + str(gammas[i])
        df[colname]=deltas[i]

    df.head()
    print(df)
    labels=["g=0.2","g=0.4","g=0.6","g=0.8","g=1.0"]
    #labels=["g=0.2","g=0.4","g=0.6","g=0.8"]
    Display_VIPI_hyps(df, "FL_VI",labels)

    print("iters to converge", iter_vis)
    
        
    """
    # Policy Iteration
    n_iters=50
    theta=1e-2
    iter_pis=[]
    delta_list=[]
    for gamma in gammas:
        Vp, Vp_track, ppi,deltasp,iter_pi = pi(frozen_lake, gamma, n_iters, theta, "FL_PI_gamma_" + str(gamma), False)
        iter_pis.append(iter_pi)
        delta_list.append(np.asarray(deltasp))
        #print("gamma,iters", gamma, iter_pi)

    deltas = np.array(list(zip_longest(*delta_list, fillvalue=0))).T 
    #print(deltas)
    print("shape", deltas.shape)

    df = pd.DataFrame()
    for i in range(len(deltas)):
        colname = "g=" + str(gammas[i])
        df[colname]=deltas[i]

    df.head()
    print(df)
    #labels=["g=0.2","g=0.4","g=0.6","g=0.8","g=1.0"]
    labels=["g=0.2","g=0.4","g=0.6","g=0.8"]
    Display_VIPI_hyps(df, "FL_PI",labels)

    print("iters to converge", iter_pis)
    """

    
    # Q Learning
    #q(frozen_lake, "Frozen_Lake")



    
    

    

def Frozen_Lake():

    # Get the problem grid
    frozen_lake, holes, goal_state = GetFrozenLakeProblem(3) #1 is 4x4; #2 is another 4x4; #3 is 20x20

    # Modify the rewards in the problem grid
    frozen_lake = GetFrozenLakeWithRewards(frozen_lake, holes, goal_state)
    #print("This is the P of the environment", frozen_lake.env.P)
    
    # Value Iteration
    gamma=1.0
    n_iters=1000
    theta=1e-2
    #vi(frozen_lake, gamma, n_iters, theta, "Frozen_Lake")
        
        
    

    # Policy Iteration
    gamma=1.0
    n_iters=50
    theta=1e-2
    pi(frozen_lake,gamma, n_iters, theta, "Frozen_Lake")

    
    # Q Learning
    #qtime = q(frozen_lake, "Frozen_Lake")
    print("time ", q)


def DisplayTaxi(taxi):
    
    stateIndex=0
    for state,tp in taxi.env.P.items():
        print("state", state)
        print("-----0-south; 1-north; 2-east")
        directionIndex=0
        for direction,probs in tp.items():
            
            print("---direction",direction)
            #print("---probs", probs)
            probIndex = 0
            for prob in probs:
                print("---------prob", prob)
                probIndex += 1
            directionIndex += 1       
        print("------------")
        stateIndex += 1

def Taxi():
    
    
    seed=42
    
    # create Taxi environment
    taxi = gym.make('Taxi-v3', render_mode="rgb_array")
    #print("number of states: ", len(taxi.env.P))
    #DisplayTaxi(taxi)

    
    taxi.reset(seed=seed)

    """
    # Set the environment to a specific state
    # For example, you can set the taxi at a specific location by modifying the observation
    # The observation consists of (taxi_row, taxi_col, passenger_location, destination)
    # You can set these values to the desired state
    taxi_row = 0
    taxi_col = 0
    passenger_location = 3
    destination = 2
    observation = taxi.encode(taxi_row, taxi_col, passenger_location, destination)
    # Set the environment to the desired state
    taxi.s = observation
    """

    print("taxi obs", taxi.s)
    

   
    #Value Iteration
    gamma=1.0
    n_iters=1000
    theta=1e-2
    #V, V_track, vpi, deltas, iters, vtime = vi(taxi, gamma, n_iters, theta, "Taxi")
    #Test_taxi(taxi, vpi, "VI",seed)


    """
    taxi.reset()
    # Set the environment to a specific state
    # For example, you can set the taxi at a specific location by modifying the observation
    # The observation consists of (taxi_row, taxi_col, passenger_location, destination)
    # You can set these values to the desired state
    taxi_row = 0
    taxi_col = 0
    passenger_location = 3
    destination = 2
    observation = taxi.encode(taxi_row, taxi_col, passenger_location, destination)
    # Set the environment to the desired state
    taxi.s = observation
    """
        
    #Policy Iteration
    gamma=1.0
    n_iters=50
    theta=11
    #Vp, Vp_track, ppi, pdeltas, iters, ptime = pi(taxi,gamma, n_iters, theta, "Taxi")
    #Test_taxi(taxi, ppi, "PI")

    print("taxi obs", taxi.s)

    #print("times",vtime,ptime)
    #Q Learning
    #q(taxi, "Taxi")

    trained_env, Q, Q_track, pi = Q_train_for_taxi(taxi,"Q_Taxi")
    #Test_taxi(taxi, pi,"Q_Taxi",seed)
    
    # end this instance of the taxi environment
    taxi.close()

"""
# Define custom TaxiEnv with 180 states
class CustomTaxiEnv(TaxiEnv):
    def __init__(self):
        super().__init__()
        self.max_row=3
        self.max_col=3
        self.num_states=180
        self.num_actions=6
"""

def SmallTaxi_hyp():

    """
    # Register custom TaxiEnv
    register(id='CustomTaxi-v0', entry_point='sroy335_code_5:CustomTaxiEnv')

    taxi = gym.make('CustomTaxi-v0')
    """
    seed = 42
    render_mode = "rgb_array"
    taxi = CustomTaxiEnv(render_mode)

    
    taxi.reset(seed=seed)
  
    #Value Iteration
    gamma=1.0
    n_iters=10000
    theta=1e-2
    #V, V_track, vpi, deltas, itersv, vtime = small_taxi_vi(taxi, gamma, n_iters, theta, "SmallTaxi")
    #Test_taxi(taxi, vpi, "VI_Small_Taxi_",seed)

        
    #Policy Iteration
    gamma=1.0
    n_iters=50
    theta=11
    #Vp, Vp_track, ppi, pdeltas, iters, ptime = small_taxi_pi(taxi,gamma, n_iters, theta, "SmallTaxi")
    #Test_taxi(taxi, ppi, "PI_Small_Taxi",seed)
    #print("times",vtime,ptime)

    #Q
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
    for alpha in alphas:
        trained_env, Q, Q_track, pi = Q_train_for_taxi(taxi,"Q_Small_Taxi",alpha)    
    Test_taxi(taxi, pi,"Q_Small_Taxi",seed)



def SmallTaxi():

    """
    # Register custom TaxiEnv
    register(id='CustomTaxi-v0', entry_point='sroy335_code_5:CustomTaxiEnv')

    taxi = gym.make('CustomTaxi-v0')
    """
    seed = 42
    render_mode = "rgb_array"
    taxi = CustomTaxiEnv(render_mode)

    
    taxi.reset(seed=seed)
  
    #Value Iteration
    gamma=1.0
    n_iters=10000
    theta=1e-2
    #V, V_track, vpi, deltas, itersv, vtime = small_taxi_vi(taxi, gamma, n_iters, theta, "SmallTaxi")
    #Test_taxi(taxi, vpi, "VI_Small_Taxi_",seed)

        
    #Policy Iteration
    gamma=1.0
    n_iters=50
    theta=11
    #Vp, Vp_track, ppi, pdeltas, iters, ptime = small_taxi_pi(taxi,gamma, n_iters, theta, "SmallTaxi")
    #Test_taxi(taxi, ppi, "PI_Small_Taxi",seed)
    #print("times",vtime,ptime)

    #Q
    trained_env, Q, Q_track, pi = Q_train_for_taxi(taxi,"Q_Small_Taxi")    
    Test_taxi(taxi, pi,"Q_Small_Taxi",seed)


   
   



def Taxi2():
    # create Taxi environment
    env = gym.make('Taxi-v3',render_mode = "rgb_array")

    # initialize q-table
    state_size = env.observation_space.n

    print("state_size", state_size)
    
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    # hyperparameters
    learning_rate = 0.9
    discount_rate = 0.8
    epsilon = 1.0
    decay_rate= 0.005

    # training variables
    num_episodes = 1000
    max_steps = 99 # per episode

    # training
    for episode in range(num_episodes):

        # reset the environment
        state_tuple = env.reset()
        state=state_tuple[0]
        done = False

        for s in range(max_steps):

            # exploration-exploitation tradeoff
            if random.uniform(0,1) < epsilon:
                # explore
                action = env.action_space.sample()
            else:
                # exploit
                action = np.argmax(qtable[state,:])

            # take action and observe reward
            
            new_state, reward, done, info, _ = env.step(action)

            # Q-learning algorithm
            print("state,action,new_state",state)
            qtable[state,action] = qtable[state,action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:])-qtable[state,action])

            # Update to our new state
            state = new_state

            # if done, finish episode
            if done == True:
                break

        # Decrease epsilon
        epsilon = np.exp(-decay_rate*episode)

    print(f"Training completed over {num_episodes} episodes")
    input("Press Enter to watch trained agent...")

    # watch trained agent
    state_tuple = env.reset()
    state = state_tuple[0]
    done = False
    rewards = 0

    for s in range(max_steps):

        print(f"TRAINED AGENT")
        print("Step {}".format(s+1))
        action = np.argmax(qtable[state,:])
        new_state, reward, done, info,_ = env.step(action)
        rewards += reward
        env.render()
        print(f"score: {rewards}")
        state = new_state

        if done == True:
            break

    env.close()

def Taxi3():
    env = gym.make("Taxi-v3", render_mode="rgb_array").env
    state, _ = env.reset()

    # Print dimensions of state and action space
    print("State space: {}".format(env.observation_space))
    print("Action space: {}".format(env.action_space))

    # Sample random action
    action = env.action_space.sample(env.action_mask(state))
    next_state, reward, done, _, _ = env.step(action)

    # Print output
    print("State: {}".format(state))
    print("Action: {}".format(action))
    print("Action mask: {}".format(env.action_mask(state)))
    print("Reward: {}".format(reward))

    # Render and plot an environment frame
    frame = env.render()
    plt.imshow(frame)
    plt.axis("off")
    plt.show()
    
    """
    #Simulation with random agent
    epoch = 0
    num_failed_dropoffs = 0
    experience_buffer = []
    cum_reward = 0

    done = False

    state, _ = env.reset()

    while not done:
        # Sample random action
        "Action selection without action mask"
        action = env.action_space.sample()

        "Action selection with action mask"
        #action = env.action_space.sample(env.action_mask(state))

        state, reward, done, _, _ = env.step(action)
        cum_reward += reward

        # Store experience in dictionary
        experience_buffer.append({
            "frame": env.render(),
            "episode": 1,
            "epoch": epoch,
            "state": state,
            "action": action,
            "reward": cum_reward,
            }
        )

        if reward == -10:
            num_failed_dropoffs += 1

        epoch += 1

    # Run animation and print console output
    run_animation(experience_buffer)

    print("# epochs: {}".format(epoch))
    print("# failed drop-offs: {}".format(num_failed_dropoffs))
    """

    trained_env, Q, Q_track, pi = Q_train_for_taxi(env)
    Test_taxi(env, pi,"Q")

    

def Q_train_for_taxi(env,problem="",alpha=0.1):
    start = time.time()

    """Training the agent"""
    nS = env.observation_space.n
    nA = env.action_space.n
    q_table = np.zeros([nS, nA])

    # Hyperparameters
    #alpha = 0.1  # Learning rate
    gamma = 1.0  # Discount rate
    epsilon = 0.1  # Exploration rate
    num_episodes = 10000  # Number of episodes

    # Output for plots
    cum_rewards = np.zeros([num_episodes])
    total_epochs = np.zeros([num_episodes])
    
    Q_track = np.zeros((num_episodes, nS, nA), dtype=np.float64)
    for episode in range(1, num_episodes+1):
        # Reset environment
        state, info = env.reset()
        epoch = 0 
        num_failed_dropoffs = 0
        done = False
        cum_reward = 0

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            next_state, reward, done, _ , info = env.step(action)
            cum_reward += reward
            old_q_value = q_table[state,action]
            next_max = np.max(q_table[next_state])
            new_q_value=(1 - alpha)*old_q_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_q_value
            if reward == -10:
                num_failed_dropoffs += 1
            state = next_state
            epoch += 1
            total_epochs[episode-1] = epoch
            cum_rewards[episode-1] = cum_reward

        if episode % 100 == 0:
            clear_output(wait=True)
            print(f"Episode #: {episode}")
        Q_track[episode-1] = q_table

    
    V = np.max(q_table, axis=1)
    # Explanation of lambda:
    # def pi(s):
    #   policy = dict()
    #   for state, action in enumerate(np.argmax(Q, axis=1)):
    #       policy[state] = action
    #   return policy[s]
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(q_table, axis=1))}[s]

    end = time.time()
    qtime= end - start
    print("\n")
    print("===Training completed.===\n")
    print("time = ", qtime)

    # Plot reward convergence
    plt.title("Cumulative reward per episode")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.plot(cum_rewards)
    pltname = problem + "_Cumulative_Reward.png"
    plt.savefig(pltname)
    plt.show()

    # Plot epoch convergence
    plt.title("# epochs per episode")
    plt.xlabel("Episode")
    plt.ylabel("# epochs")
    plt.plot(total_epochs)
    pltname = problem + "_Epochs_per_episode.png"
    plt.savefig(pltname)
    plt.show()


    return env, q_table, Q_track, pi

def Test_taxi(env, pi, name="",seed=42):
    np.random.seed(42)
    """Test policy performance after training"""

    num_epochs = 0
    total_failed_deliveries = 0
    num_episodes = 1
    experience_buffer = []
    store_gif = True
    nS = env.observation_space.n
    new_pi = list(map(lambda x: pi(x), range(nS)))


    for episode in range(1, num_episodes+1):
        # Initialize experience buffer

        my_env = env.reset(seed=seed)
        state = 29#my_env[0]
        epoch = 1 
        num_failed_deliveries =0
        cum_reward = 0
        done = False

        while not done:
            action = new_pi[state]
            state, reward, done, _, _ = env.step(action)
            cum_reward += reward

            if reward == -10:
                num_failed_deliveries += 1

            # Store rendered frame in animation dictionary
            experience_buffer.append({
                'frame': env.render(),
                'episode': episode,
                'epoch': epoch,
                'state': state,
                'action': action,
                'reward': cum_reward
                }
            )

            epoch += 1

        total_failed_deliveries += num_failed_deliveries
        num_epochs += epoch

        if store_gif:
            path='./'
            filename= name +'animation.gif'
            store_episode_as_gif(experience_buffer,path,filename,name)

    # Run animation and print output
    run_animation(experience_buffer,name)

    # Print final results
    print("\n") 
    print(f"Test results after {num_episodes} episodes:")
    print(f"Mean # epochs per episode: {num_epochs / num_episodes}")
    print(f"Mean # failed drop-offs per episode: {total_failed_deliveries / num_episodes}")
    


    



def run_animation(experience_buffer, name=""):
    """Function to run animation"""
    time_lag = 0.05  # Delay (in s) between frames
    for experience in experience_buffer:
        # Plot frame
        clear_output(wait=True)
        plt.imshow(experience['frame'])
        plt.axis('off')
        plt.show()

        # Print console output
        print(f"Episode: {experience['episode']}/{experience_buffer[-1]['episode']}")
        print(f"Epoch: {experience['epoch']}/{experience_buffer[-1]['epoch']}")
        print(f"State: {experience['state']}")
        print(f"Action: {experience['action']}")
        print(f"Reward: {experience['reward']}")
        # Pauze animation
        sleep(time_lag)

def store_episode_as_gif(experience_buffer, path='./', filename='animation.gif',name="algo"):
    """Store episode as gif animation"""
    fps = 5   # Set framew per seconds
    dpi = 300  # Set dots per inch
    interval = 50  # Interval between frames (in ms)

    # Retrieve frames from experience buffer
    frames = []
    for experience in experience_buffer:
        frames.append(experience['frame'])

    # Fix frame size
    plt.figure(figsize=(frames[0].shape[1] / dpi, frames[0].shape[0] / dpi), dpi=dpi)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    # Generate animation
    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=interval)

    # Save output as gif
    anim.save(path + filename, writer='imagemagick', fps=fps)

    # Save as images
    for i in range(len(frames)):
        img = plt.imshow(frames[i])
        title = name + "_" + str(i) + "_.png"
        plt.savefig(title)
        

if __name__ == "__main__":

    Frozen_Lake()
    #Taxi()
    #SmallTaxi()
    #Frozen_Lake_Hyp()


    

    

