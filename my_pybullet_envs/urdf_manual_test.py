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
import time

# import pybullet_data

import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# obUids = p.loadMJCF("mjcf/humanoid.xml")
# humanoid = obUids[1]

urdf_ID = p.loadURDF(os.path.join(currentdir, "assets/hopper_my_box.urdf"),
                     [0, 0, 1.5],
                     p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
                     useFixedBase=1)

gravId = p.addUserDebugParameter("gravity", -10, 10, 0.0)
jointIds = []
paramIds = []

p.setPhysicsEngineParameter(numSolverIterations=10)
p.changeDynamics(urdf_ID, -1, linearDamping=0, angularDamping=0)

for j in range(p.getNumJoints(urdf_ID)):
    p.changeDynamics(urdf_ID, j, linearDamping=0, angularDamping=0)
    info = p.getJointInfo(urdf_ID, j)
    print(info)
    jointName = info[1]
    jointType = info[2]
    if jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE:
        jointIds.append(j)
        paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4, 0))

# input("press enter")

p.setRealTimeSimulation(1)
while 1:
    p.setGravity(0, 0, p.readUserDebugParameter(gravId))
    for i in range(len(paramIds)):
        c = paramIds[i]
        targetPos = p.readUserDebugParameter(c)
        p.setJointMotorControl2(urdf_ID, jointIds[i], p.POSITION_CONTROL, targetPos, force=240.)
    time.sleep(0.01)
    # input("press enter")
