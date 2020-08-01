#  Copyright 2020 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import pybullet_envs
import gym
import argparse
import pybullet as p

import my_pybullet_envs


def test(args):
    count = 0
    env = gym.make(args.env, render=True)
    env.seed(args.seed)
    # env.env.configure(args)
    # print("args.render=", args.render)
    # if (args.render == 1):
    #   env.render(mode="human")

    for _ in range(100):
        env.reset()
        input("press enter")
        if (args.resetbenchmark):
            while (1):
                env.reset()
                print("p.getNumBodies()=", p.getNumBodies())
                print("count=", count)
                count += 1
        print("action space:")

        sample = env.action_space.sample()
        action = sample
        print("action=")
        print(action)

        for i in range(args.steps):
            obs, rewards, done, _ = env.step(action)

            print("obs=")
            print(obs)
            print("rewards")
            print(rewards)
            print("done")
            print(done)

            if done:
                input("press enter")
                break

        input("press enter")

    # p.getCameraImage()


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='LaikagoBulletEnv-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=222)
    parser.add_argument('--render', help='OpenGL Visualizer', type=int, default=1)
    parser.add_argument('--resetbenchmark',
                        help='Repeat reset to show reset performance',
                        type=int,
                        default=0)
    parser.add_argument('--steps', help='Number of steps', type=int, default=500)  # large enough

    args = parser.parse_args()
    test(args)


if __name__ == '__main__':
    main()
