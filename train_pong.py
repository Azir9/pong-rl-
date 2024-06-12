import torch
import gym
from policy_grad import REINFORCE
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def main(name= "Pong-v4",alg = "policy_grad"):

    learning_rate = 1e-3
    num_episodes = 10
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ## init the gym env
    env_name =name
    env = gym.make(env_name)



    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    
    if alg ==  "policy_grad":
        agent = REINFORCE(state_dim=state_dim,hidden_dim=128,action_dim=action_dim,learning_rate=learning_rate,gamma=gamma,device=device)
    
    return_list = []

    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                    env.render()
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE on {}'.format(env_name))
    plt.show()


if __name__== "__main__":
    main(name="Pong-v4")