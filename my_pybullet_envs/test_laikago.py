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

import pybullet as p
import pybullet_data as pd

import time

import numpy as np

p.connect(p.GUI)
# p.setAdditionalSearchPath(pd.getDataPath())
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

plane = p.loadURDF("assets/plane.urdf")
p.setGravity(0, 0, -9.8)
p.setTimeStep(1. / 500)
# p.setDefaultContactERP(0)
# urdfFlags = p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
urdfFlags = p.URDF_USE_SELF_COLLISION
quadruped = p.loadURDF("assets/laikago/laikago_toes_limits.urdf", [0, 0, .5], [0, 0.5, 0.5, 0],
                       flags=urdfFlags,
                       useFixedBase=False)

# enable collision between lower legs

for j in range(p.getNumJoints(quadruped)):
    print(p.getJointInfo(quadruped, j))

# # 2,5,8 and 11 are the lower legs
# lower_legs = [2, 5, 8, 11]
# for l0 in lower_legs:
#     for l1 in lower_legs:
#         if (l1 > l0):
#             enableCollision = 1
#             print("collision for pair", l0, l1,
#                   p.getJointInfo(quadruped, l0)[12],
#                   p.getJointInfo(quadruped, l1)[12], "enabled=", enableCollision)
#             p.setCollisionFilterPair(quadruped, quadruped, 2, 5, enableCollision)

jointIds = []
paramIds = []
jointOffsets = []
jointDirections = [-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1]
jointAngles = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(4):
    jointOffsets.append(0)
    jointOffsets.append(-0.7)
    jointOffsets.append(0.7)

maxForceId = p.addUserDebugParameter("maxForce", 0, 100, 20)

for j in range(p.getNumJoints(quadruped)):
    p.changeDynamics(quadruped, j, linearDamping=0, angularDamping=0)
    info = p.getJointInfo(quadruped, j)
    # print(info)
    jointName = info[1]
    jointType = info[2]
    if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
        jointIds.append(j)

p.getCameraImage(480, 320)
p.setRealTimeSimulation(0)

joints = []

with open(pd.getDataPath() + "/laikago/data1.txt", "r") as filestream:
    for line in filestream:
        print("line=", line)
        maxForce = p.readUserDebugParameter(maxForceId)
        currentline = line.split(",")
        # print (currentline)
        # print("-----")
        frame = currentline[0]
        t = currentline[1]
        # print("frame[",frame,"]")
        joints = currentline[2:14]
        # print("joints=",joints)
        for j in range(12):
            targetPos = float(joints[j])
            p.setJointMotorControl2(quadruped,
                                    jointIds[j],
                                    p.POSITION_CONTROL,
                                    jointDirections[j] * targetPos + jointOffsets[j],
                                    force=maxForce)
            print(p.getJointState(quadruped, jointIds[j])[-1])

        joints_state = p.getJointStates(quadruped, [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14])
        joints_q = np.array(joints_state)[:, [0]]
        joints_q = np.hstack(joints_q.flatten())
        print(joints_q)

        p.stepSimulation()
        # for lower_leg in lower_legs:
        #     # print("points for ", quadruped, " link: ", lower_leg)
        #     pts = p.getContactPoints(quadruped, -1, lower_leg)
        #     # print("num points=",len(pts))
        #     # for pt in pts:
        #     # print(pt[9])
        time.sleep(1. / 500.)

index = 0
for j in range(p.getNumJoints(quadruped)):
    p.changeDynamics(quadruped, j, linearDamping=0, angularDamping=0)
    info = p.getJointInfo(quadruped, j)
    js = p.getJointState(quadruped, j)
    # print(info)
    jointName = info[1]
    jointType = info[2]
    if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
        paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4,
                                                (js[0] - jointOffsets[index]) / jointDirections[index]))
        index = index + 1

p.setRealTimeSimulation(1)

while (1):

    for i in range(len(paramIds)):
        c = paramIds[i]
        targetPos = p.readUserDebugParameter(c)
        maxForce = p.readUserDebugParameter(maxForceId)
        p.setJointMotorControl2(quadruped,
                                jointIds[i],
                                p.POSITION_CONTROL,
                                jointDirections[i] * targetPos + jointOffsets[i],
                                force=maxForce)
