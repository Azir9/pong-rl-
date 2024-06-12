import gym
#from envs import drone_env
from time import sleep

def train(env):
    env = gym.make(env)

    env.reset()
    
    for _ in range(1000):
        sleep(1/240)

        env.step(env.action_space.sample()) # take a random action
        env.render()
    env.close()
if __name__ == "__main__":
    train("Pong-v4")