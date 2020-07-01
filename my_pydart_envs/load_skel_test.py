import pydart2 as pydart
import numpy as np
from pydart2.gui.trackball import Trackball

from static_window import *

pydart.init()
print('pydart initialization OK')


world = pydart.World(1.0 / 500, "./assets/hopper_box.skel")
robot = world.skeletons[-1]
world.set_gravity([0.0, -10.0, 0.0])
world.set_collision_detector(world.BULLET_COLLISION_DETECTOR)
print('pydart create_world OK')

# floor_name = "./assets/ground.urdf"
# floor = world.add_skeleton(floor_name)
#
# filename = "./assets/hopper_my_box.urdf"
# robot = world.add_skeleton(filename)
# print('pydart add_skeleton OK')

# https://github.com/dartsim/dart/blob/v6.3.0/dart/constraint/ContactConstraint.cpp
for i in range(0, len(world.skeletons[0].bodynodes)):
    world.skeletons[0].bodynodes[i].set_friction_coeff(1.0)
    world.skeletons[0].bodynodes[i].set_restitution_coeff(0.2)
for i in range(0, len(world.skeletons[1].bodynodes)):
    world.skeletons[1].bodynodes[i].set_friction_coeff(1.0)
    world.skeletons[1].bodynodes[i].set_restitution_coeff(1.0)

# win = pydart.gui.pyqt5.window.PyQt5Window(world)
# win.scene.set_camera(1)  # Z-up Camera
# win.run()

for jt in range(0, len(robot.joints)):
    for dof in range(len(robot.joints[jt].dofs)):
        if robot.joints[jt].has_position_limit(dof):
            robot.joints[jt].set_position_limit_enforced(True)
robot.set_self_collision_check(False)

positions = robot.positions()
print(positions)

for joint in robot.joints:
    print(joint.name)
# rootJoint added by default
# the default name for z trans root is rootJoint_pos_z
# positions['rootJoint_pos_z'] = 1.5
# positions['rootz'] = 1.5
# robot.set_positions(positions)

# robot.joints[0].set_actuator_type(pydart.joint.Joint.LOCKED)      # TODO
# floor.joints[0].set_actuator_type(pydart.joint.Joint.LOCKED)

# pydart.gui.viewer.launch(world, default_camera=1)  # Use Z-up camera

win = StaticGLUTWindow(world, None)
win.scene.add_camera(Trackball(theta=-45.0, phi=0.0, zoom=0.1), 'gym_camera')
# win.scene.set_camera(1)
win.run()

# win = pydart.gui.pyqt5.window.PyQt5Window(world)
#
# win.scene.add_camera(Trackball(theta=-45.0, phi=0.0, zoom=0.1), 'gym_camera')
# win.scene.set_camera(1)
# win.run()

while world.t < 10.0:
    robot.set_forces([0.0]*3+[100.0]*3)
    # print(world.skeletons[1].q)
    world.step()
    # win.scene.render(sim=world)
    win.runSingleStep()
