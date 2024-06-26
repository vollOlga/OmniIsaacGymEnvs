from omni.isaac.robot_assembler import RobotAssembler,AssembledRobot 
from omni.isaac.core.articulations import Articulation
import numpy as np

base_robot_path = "/World/ur10_instanceable"
attach_robot_path = "/World/Robotiq_2F_85_edit/Robotiq_2F_85"
base_robot_mount_frame = "/ee_link"
attach_robot_mount_frame = "/base_link"
fixed_joint_offset = np.array([0.0,0.0,0.0])
fixed_joint_orient = np.array([1.0,0.0,0.0,0.0])
single_robot = False

robot_assembler = RobotAssembler()
assembled_robot = robot_assembler.assemble_articulations(
	base_robot_path,
	attach_robot_path,
	base_robot_mount_frame,
	attach_robot_mount_frame,
	fixed_joint_offset,
	fixed_joint_orient,
	mask_all_collisions = True,
	single_robot=single_robot
)

# The fixed joint in a assembled robot is editable after the fact:
# offset,orient = assembled_robot.get_fixed_joint_transform()
# assembled_robot.set_fixed_joint_transform(np.array([.3,0,0]),np.array([1,0,0,0]))

# And the assembled robot can be disassembled, after which point the AssembledRobot object will no longer function.
# assembled_robot.disassemble()

# Controlling the resulting assembled robot is different depending on the single_robot flag
if single_robot:
	# The robots will be considered to be part of a single Articulation at the base robot path
	controllable_single_robot = Articulation(base_robot_path)
else:
	# The robots are controlled independently from each other
	base_robot = Articulation(base_robot_path)
	attach_robot = Articulation(attach_robot_path)
