# DoF index, DoF (joint) Name, joint type (0 means hinge joint), joint lower and upper limits, child link of this joint
# (0, b'rootx', 1, 7, 6, 1, 0.0, 0.0, -200.0, 200.0, 10000.0, 100.0, b'link1_2', (1.0, 0.0, 0.0),
# (1, b'rootz', 1, 8, 7, 1, 0.0, 0.0, -200.0, 200.0, 10000.0, 100.0, b'link1_3', (0.0, 0.0, 1.0),
# (2, b'rooty', 0, 9, 8, 1, 0.0, 0.0, -200.0, 200.0, 10000.0, 100.0, b'torso', (0.0, 1.0, 0.0),
# (3, b'thigh_joint', 0, 10, 9, 1, 1.0, 0.0001, -2.61799, 0.5, 10000.0, 100.0, b'thigh', (0.0, -1.0, 0.0),
# (4, b'leg_joint', 0, 11, 10, 1, 1.0, 0.0001, -2.61799, 0.5, 10000.0, 100.0, b'leg', (0.0, -1.0, 0.0),
# (5, b'foot_joint', 0, 12, 11, 1, 1.0, 0.0001, -0.785398, 0.785398, 10000.0, 100.0, b'foot', (0.0, -1.0, 0.0),

import pybullet_utils.bullet_client as bc
import time
import gym, gym.utils.seeding
import numpy as np
import math

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


class HopperURDF:
    def __init__(self,
                 init_noise=True,
                 obs_noise=True,
                 act_noise=True,
                 time_step=1. / 500,
                 np_random=None,
                 box_shape=True             # TODO: capsule false
                 ):

        self.init_noise = init_noise
        self.obs_noise = obs_noise
        self.act_noise = act_noise

        self._ts = time_step
        self.np_random = np_random

        self.base_init_pos = np.array([0., 0, 1.5])         # starting position
        self.base_init_euler = np.array([0., 0, 0])         # starting orientation

        self.max_forces = [200.0] * 3       # joint torque limits
        self.ctrl_dofs = [3, 4, 5]
        self.root_dofs = [0, 1, 2]          # uncontrollable 2d xyr root
        self.n_total_dofs = len(self.ctrl_dofs) + len(self.root_dofs)
        assert len(self.max_forces) == len(self.ctrl_dofs)

        self._p = None  # bullet session to connect to
        self.hopper_id = -2      # bullet id for the loaded humanoid, to be overwritten
        self.torque = None      # if using torque control, the current torque vector to apply

        self.ll = None      # stores joint lower limits
        self.ul = None      # stores joint upper limits

    def reset(
             self,
             bullet_client
             ):
        self._p = bullet_client
        self.hopper_id = self._p.loadURDF(os.path.join(currentdir,
                                                       "assets/hopper_my_box.urdf"),
                                          list(self.base_init_pos),
                                          self._p.getQuaternionFromEuler(list(self.base_init_euler)),
                                          flags=self._p.URDF_USE_SELF_COLLISION,
                                          useFixedBase=1)

        # self.print_all_joints_info()

        if self.init_noise:
            init_q = self.perturb([0.0] * self.n_total_dofs, 0.01)
            init_dq = self.perturb([0.0] * self.n_total_dofs, 0.1)
        else:
            init_q = self.perturb([0.0] * self.n_total_dofs, 0.0)
            init_dq = self.perturb([0.0] * self.n_total_dofs, 0.0)

        for ind in range(self.n_total_dofs):
            self._p.resetJointState(self.hopper_id, ind, init_q[ind], init_dq[ind])

        # turn off root default control:
        self._p.setJointMotorControlArray(
            bodyIndex=self.hopper_id,
            jointIndices=self.root_dofs,
            controlMode=self._p.VELOCITY_CONTROL,
            forces=[0.0] * len(self.root_dofs))
        # use torque control
        self._p.setJointMotorControlArray(
            bodyIndex=self.hopper_id,
            jointIndices=self.ctrl_dofs,
            controlMode=self._p.VELOCITY_CONTROL,
            forces=[0.0] * len(self.ctrl_dofs))
        self.torque = [0.0] * len(self.ctrl_dofs)

        self.ll = np.array([self._p.getJointInfo(self.hopper_id, i)[8] for i in self.ctrl_dofs])
        self.ul = np.array([self._p.getJointInfo(self.hopper_id, i)[9] for i in self.ctrl_dofs])

    def perturb(self, arr, r=0.02):
        r = np.abs(r)
        return np.copy(np.array(arr) + self.np_random.uniform(low=-r, high=r, size=len(arr)))

    def print_all_joints_info(self):
        for i in range(self._p.getNumJoints(self.hopper_id)):
            print(self._p.getJointInfo(self.hopper_id, i)[0:3],
                  self._p.getJointInfo(self.hopper_id, i)[8], self._p.getJointInfo(self.hopper_id, i)[9],
                  self._p.getJointInfo(self.hopper_id, i)[12])

    def apply_action(self, a):
        act = np.clip(a, -1.0, 1.0)

        self.torque = act * self.max_forces

        if self.act_noise:
            self.torque = self.perturb(self.torque, 10.0)   # TODO

        self._p.setJointMotorControlArray(
            bodyIndex=self.hopper_id,
            jointIndices=self.ctrl_dofs,
            controlMode=self._p.TORQUE_CONTROL,
            forces=self.torque)

    def get_q_dq(self, dofs):
        joints_state = self._p.getJointStates(self.hopper_id, dofs)
        joints_q = np.array(joints_state)[:, [0]]
        joints_q = np.hstack(joints_q.flatten())
        joints_dq = np.array(joints_state)[:, [1]]
        joints_dq = np.hstack(joints_dq.flatten())
        return joints_q, joints_dq

    def get_robot_observation(self):
        obs = []
        a_q, a_dq = self.get_q_dq(self.root_dofs + self.ctrl_dofs)

        # TODO: seems a bug:
        # -1.5 for the a_q[1]
        a_q[1] = self._p.getLinkState(self.hopper_id, 2, computeForwardKinematics=1)[0][2]
        a_dq[1] = self._p.getLinkState(self.hopper_id, 2, computeForwardKinematics=1, computeLinkVelocity=1)[6][2]

        # print("all_q", a_q)
        # print(self._p.getJointState(self.hopper_id, 0))
        # print(self._p.getJointState(self.hopper_id, 1))
        # print(self._p.getLinkState(self.hopper_id, 2, computeLinkVelocity=1)[0])
        # print(self._p.getLinkState(self.hopper_id, 2, computeLinkVelocity=1)[6])

        obs.extend(list(a_q[1:]))
        obs.extend(list(np.clip(a_dq / 10.0, -1, 1)))

        if self.obs_noise:
            obs = self.perturb(obs, 0.1)        # TODO

        return list(obs)


# if __name__ == "__main__":
#     import pybullet as p
#
#     hz = 500.0
#     dt = 1.0 / hz
#
#     sim = bc.BulletClient(connection_mode=p.GUI)
#
#     for n in range(100):
#         sim.resetSimulation()
#         # sim.setPhysicsEngineParameter(numSolverIterations=200)
#
#         sim.setGravity(0, 0, 0)
#         sim.setTimeStep(dt)
#         sim.setRealTimeSimulation(0)
#
#         rand, seed = gym.utils.seeding.np_random(0)
#         robot = HopperURDF(np_random=rand)
#         robot.reset(sim)
#         input("press enter")
#
#         for t in range(400):
#             # arm.apply_action(arm.np_random.uniform(low=-0.003, high=0.003, size=7+17)+np.array([-0.005]*7+[-0.01]*17))
#             sim.stepSimulation()
#             input("press enter")
#             # arm.get_robot_observation()
#             time.sleep(1. / 500.)
#         # print("final obz", arm.get_robot_observation())
#         # ls = sim.getLinkState(arm.arm_id, arm.ee_id)
#         # newPos = ls[4]
#         # print(newPos, sim.getEulerFromQuaternion(ls[5]))
#
#     sim.disconnect()
