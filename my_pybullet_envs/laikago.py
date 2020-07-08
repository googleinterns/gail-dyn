# DoF index, DoF (joint) Name, joint type (0 means hinge joint), joint lower and upper limits, child link of this joint
# (0, b'FR_hip_motor_2_chassis_joint', 0) -0.873 1.0472 b'FR_hip_motor'
# (1, b'FR_upper_leg_2_hip_motor_joint', 0) -1.3 3.4 b'FR_upper_leg'
# (2, b'FR_lower_leg_2_upper_leg_joint', 0) -2.164 0.0 b'FR_lower_leg'
# (3, b'jtoeFR', 4) 0.0 -1.0 b'toeFR'
# (4, b'FL_hip_motor_2_chassis_joint', 0) -0.873 1.0472 b'FL_hip_motor'
# (5, b'FL_upper_leg_2_hip_motor_joint', 0) -1.3 3.4 b'FL_upper_leg'
# (6, b'FL_lower_leg_2_upper_leg_joint', 0) -2.164 0.0 b'FL_lower_leg'
# (7, b'jtoeFL', 4) 0.0 -1.0 b'toeFL'
# (8, b'RR_hip_motor_2_chassis_joint', 0) -0.873 1.0472 b'RR_hip_motor'
# (9, b'RR_upper_leg_2_hip_motor_joint', 0) -1.3 3.4 b'RR_upper_leg'
# (10, b'RR_lower_leg_2_upper_leg_joint', 0) -2.164 0.0 b'RR_lower_leg'
# (11, b'jtoeRR', 4) 0.0 -1.0 b'toeRR'
# (12, b'RL_hip_motor_2_chassis_joint', 0) -0.873 1.0472 b'RL_hip_motor'
# (13, b'RL_upper_leg_2_hip_motor_joint', 0) -1.3 3.4 b'RL_upper_leg'
# (14, b'RL_lower_leg_2_upper_leg_joint', 0) -2.164 0.0 b'RL_lower_leg'
# (15, b'jtoeRL', 4) 0.0 -1.0 b'toeRL'
# ctrl dofs: [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]

import pybullet_utils.bullet_client as bc
import time
import gym, gym.utils.seeding
import numpy as np
import math

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


class LaikagoBullet:
    def __init__(self,
                 init_noise=True,
                 obs_noise=True,
                 act_noise=True,
                 time_step=1. / 500,
                 np_random=None,
                 using_torque_ctrl=True             # TODO: pos control
                 ):

        self.init_noise = init_noise
        self.obs_noise = obs_noise
        self.act_noise = act_noise

        self._ts = time_step
        self.np_random = np_random
        self.using_torque_ctrl = using_torque_ctrl

        self.base_init_pos = np.array([0, 0, .5])         # starting position
        self.base_init_euler = np.array([1.5708, 0, 1.5708])         # starting orientation

        self.feet = [3, 7, 11, 15]

        self.max_forces = [30.0] * 12       # joint torque limits    TODO
        self.init_q = [0.0, 0.0, -0.5] * 4                          # TODO
        self.ctrl_dofs = []

        self._p = None  # bullet session to connect to
        self.go_id = -2      # bullet id for the loaded humanoid, to be overwritten
        self.torque = None      # if using torque control, the current torque vector to apply

        self.ll = None      # stores joint lower limits
        self.ul = None      # stores joint upper limits

    def reset(
             self,
             bullet_client
             ):
        self._p = bullet_client
        self.go_id = self._p.loadURDF(os.path.join(currentdir,
                                                   "assets/laikago/laikago_toes_limits.urdf"),
                                      list(self.base_init_pos),
                                      list(self._p.getQuaternionFromEuler(list(self.base_init_euler))),
                                      flags=self._p.URDF_USE_SELF_COLLISION,
                                      useFixedBase=0)
        # self.go_id = self._p.loadURDF(os.path.join(currentdir,
        #                                                "assets/laikago/laikago_limits.urdf"),
        #                               list(self.base_init_pos),
        #                               list(self._p.getQuaternionFromEuler([1.5708, 0, 1.5708])),
        #                               flags=self._p.URDF_USE_SELF_COLLISION,
        #                               useFixedBase=0)
        # self.go_id = self._p.loadURDF(os.path.join(currentdir,
        #                                                "assets/laikago/laikago_toes_limits.urdf"),
        #                               list(self.base_init_pos),
        #                               list(self.base_init_quat),
        #                               useFixedBase=0)

        # self.print_all_joints_info()

        self.ctrl_dofs = []
        for j in range(self._p.getNumJoints(self.go_id)):
            # self._p.changeDynamics(self.go_id, j, linearDamping=0, angularDamping=0)     # TODO
            self._p.changeDynamics(self.go_id, j, jointDamping=0.5)  # TODO
            info = self._p.getJointInfo(self.go_id, j)
            joint_type = info[2]
            if joint_type == self._p.JOINT_PRISMATIC or joint_type == self._p.JOINT_REVOLUTE:
                self.ctrl_dofs.append(j)

        # print("ctrl dofs:", self.ctrl_dofs)

        if self.init_noise:
            init_q = self.perturb(self.init_q, 0.01)
            init_dq = self.perturb([0.0] * len(self.ctrl_dofs), 0.1)
        else:
            init_q = self.init_q
            init_dq = [0.0] * len(self.ctrl_dofs)

        for pt, ind in enumerate(self.ctrl_dofs):
            self._p.resetJointState(self.go_id, ind, init_q[pt], init_dq[pt])

        # turn off root default control:
        # use torque control
        self._p.setJointMotorControlArray(
            bodyIndex=self.go_id,
            jointIndices=self.ctrl_dofs,
            controlMode=self._p.VELOCITY_CONTROL,
            forces=[0.0] * len(self.ctrl_dofs))
        self.torque = [0.0] * len(self.ctrl_dofs)

        assert len(self.max_forces) == len(self.ctrl_dofs)

        self.ll = np.array([self._p.getJointInfo(self.go_id, i)[8] for i in self.ctrl_dofs])
        self.ul = np.array([self._p.getJointInfo(self.go_id, i)[9] for i in self.ctrl_dofs])
        assert len(self.max_forces) == len(self.ll)

    def soft_reset(
            self,
            bullet_client
            ):
        self._p = bullet_client

        self._p.resetBaseVelocity(self.go_id, [0., 0, 0], [0., 0, 0])
        self._p.resetBasePositionAndOrientation(self.go_id,
                                                list(self.base_init_pos),
                                                list(self._p.getQuaternionFromEuler(list(self.base_init_euler)))
                                                )

        if self.init_noise:
            init_q = self.perturb(self.init_q, 0.01)
            init_dq = self.perturb([0.0] * len(self.ctrl_dofs), 0.1)
        else:
            init_q = self.init_q
            init_dq = [0.0] * len(self.ctrl_dofs)

        for pt, ind in enumerate(self.ctrl_dofs):
            self._p.resetJointState(self.go_id, ind, init_q[pt], init_dq[pt])

    def perturb(self, arr, r=0.02):
        r = np.abs(r)
        return np.copy(np.array(arr) + self.np_random.uniform(low=-r, high=r, size=len(arr)))

    def print_all_joints_info(self):
        for i in range(self._p.getNumJoints(self.go_id)):
            print(self._p.getJointInfo(self.go_id, i)[0:3],
                  self._p.getJointInfo(self.go_id, i)[8], self._p.getJointInfo(self.go_id, i)[9],
                  self._p.getJointInfo(self.go_id, i)[12])

    def apply_action(self, a):
        act = np.clip(a, -1.0, 1.0)

        self.torque = act * self.max_forces

        # 10% white noise
        if self.act_noise:
            self.torque = self.perturb(self.torque, np.array(self.max_forces)/10.0)

        self._p.setJointMotorControlArray(
            bodyIndex=self.go_id,
            jointIndices=self.ctrl_dofs,
            controlMode=self._p.TORQUE_CONTROL,
            forces=self.torque)

    def get_q_dq(self, dofs):
        joints_state = self._p.getJointStates(self.go_id, dofs)
        joints_q = np.array(joints_state)[:, [0]]
        joints_q = np.hstack(joints_q.flatten())
        joints_dq = np.array(joints_state)[:, [1]]
        joints_dq = np.hstack(joints_dq.flatten())
        return joints_q, joints_dq

    def get_link_com_xyz_orn(self, link_id, FK=1):
        # get the world transform (xyz and quaternion) of the Center of Mass of the link
        assert link_id >= -1
        if link_id == -1:
            link_com, link_quat = self._p.getBasePositionAndOrientation(self.go_id)
        else:
            link_com, link_quat, *_ = self._p.getLinkState(self.go_id, link_id, computeForwardKinematics=FK)
        return list(link_com), list(link_quat)

    def get_robot_observation(self):

        obs = []

        # root dq (6)
        base_vel, base_ang_vel = self._p.getBaseVelocity(self.go_id)
        base_vel = np.clip(np.array(base_vel) / 5.0, -1, 1)
        base_ang_vel = np.clip(np.array(base_ang_vel) / 10.0, -1, 1)

        obs.extend(base_vel)
        obs.extend(base_ang_vel)

        # root q (without root x)
        # y could be unbounded
        root_pos, root_orn = self.get_link_com_xyz_orn(-1)
        root_x, root_y, root_z = root_pos
        obs.extend(root_pos[1:])
        obs.extend(root_orn)

        # feet (offset by root)
        for link in self.feet:
            pos, _ = self.get_link_com_xyz_orn(link, FK=1)
            # print(pos)
            pos[0] -= root_x
            pos[1] -= root_y
            pos[2] -= root_z
            obs.extend(pos)

        # non-root joint q/dq
        q, dq = self.get_q_dq(self.ctrl_dofs)
        obs.extend(list(q))
        obs.extend(list(np.clip(dq / 10.0, -2, 2)))

        if self.obs_noise:
            obs = self.perturb(obs, 0.1)

        # print(np.array(obs))
        return list(obs)

    def is_root_com_in_support(self):
        root_com, _ = self._p.getBasePositionAndOrientation(self.go_id)
        feet_max_x = -1000
        feet_min_x = 1000
        feet_max_y = -1000
        feet_min_y = 1000
        for foot in self.feet:
            x, y, _ = list(self.get_link_com_xyz_orn(foot)[0])
            if x > feet_max_x:
                feet_max_x = x
            if x < feet_min_x:
                feet_min_x = x
            if y > feet_max_y:
                feet_max_y = y
            if y < feet_min_y:
                feet_min_y = y
        return (feet_min_x - 0.05 < root_com[0]) and (root_com[0] < feet_max_x + 0.05) \
            and (feet_min_y - 0.05 < root_com[1]) and (root_com[1] < feet_max_y + 0.05)


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
#         sim.setPhysicsEngineParameter(numSolverIterations=100)
#
#         sim.setGravity(0, 0, 0)
#         sim.setTimeStep(dt)
#         sim.setRealTimeSimulation(0)
#
#         rand, seed = gym.utils.seeding.np_random(0)
#         robot = LaikagoBullet(np_random=rand, init_noise=False)
#         robot.reset(sim)
#         input("press enter")
#
#         for t in range(400):
#             robot.apply_action(np.array([-0.1]*12))
#             sim.stepSimulation()
#             # input("press enter")
#             # arm.get_robot_observation()
#             time.sleep(1. / 500.)
#         # print("final obz", arm.get_robot_observation())
#         # ls = sim.getLinkState(arm.arm_id, arm.ee_id)
#         # newPos = ls[4]
#         # print(newPos, sim.getEulerFromQuaternion(ls[5]))
#
#     sim.disconnect()