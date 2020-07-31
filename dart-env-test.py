import gym
import my_pydart_envs
import pydart2
import numpy as np
import sys
import time

if __name__ == '__main__':
    if len(sys.argv) > 1:
        env = gym.make(sys.argv[1])
    else:
        env = gym.make('DartHopper-v1')

    env.env.disableViewer = False

    env.reset()

    for i in range(1000):
        env.render()

        env.step(env.action_space.sample())