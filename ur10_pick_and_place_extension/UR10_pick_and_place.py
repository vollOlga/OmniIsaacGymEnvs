# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.nucleus import get_asset_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.universal_robots import UR10
from omni.isaac.universal_robots.controllers import PickPlaceController
import numpy as np
import carb


# Note: checkout the required tutorials at https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html

class UR10Playing(BaseTask):
    def __init__(self, name) -> None:
        super().__init__(name = name, offset = None)
        self._goal_position = np.array([-0.3, -0.3, 0.0515 / 2.0])
        self._task_achieved = False
        return
    
    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        self._cube = scene.add(DynamicCuboid(prim_path="/World/random_cube",
                                            name="random_cube",
                                            position=np.array([0.3, 0.3, 0.3]),
                                            scale=np.array([0.0515, 0.0515, 0.0515]),
                                            color=np.array([0, 0, 1.0])))
        self._ur10_robot = scene.add(UR10(prim_path="/World/ur10",
                                        name="ur10"))
        return
    
    # Information exposed to solve the task is returned from the task through get_observations
    def get_observations(self):
        cube_position, _ = self._cube.get_world_pose()
        current_joint_positions = self._ur10_robot.get_joint_positions()
        observations = {
            self._ur10_robot.name: {
                "joint_positions": current_joint_positions,
            },
            self._cube.name: {
                "position": cube_position,
                "goal_position": self._goal_position
            }
        }
        return observations
    
    def pre_step(self, control_index, simulation_time):
        cube_position, _ = self._cube.get_world_pose()
        if not self._task_achieved and np.mean(np.abs(self._goal_position - cube_position)) < 0.02:
            # Visual Materials are applied by default to the cube
            # in this case the cube has a visual material of type
            # PreviewSurface, we can set its color once the target is reached.
            self._cube.get_applied_visual_material().set_color(color=np.array([0, 1.0, 0]))
            self._task_achieved = True
        return

    # Called after each reset,
    # for instance we can always set the gripper to be opened at the beginning after each reset
    # also we can set the cube's color to be blue
    def post_reset(self):
        self._ur10_robot.gripper.set_joint_positions(self._ur10_robot.gripper.joint_opened_positions)
        self._cube.get_applied_visual_material().set_color(color=np.array([0, 0, 1.0]))
        self._task_achieved = False
        return

class UR10PickAndPlace(BaseSample):
    def __init__(self, name) -> None:
        super().__init__()
        return

    def setup_scene(self):

        world = self.get_world()
        world.scene.add_default_ground_plane()
        assets_root_path = get_asset_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find assets root path")
        assets_path = assets_root_path + "/Isaac/Samples/Leonardo/Stage/ur10_bin_stacking_short_suction.usd"
        add_reference_to_stage(usd_path=assets_path, prim_path="/World/ur10")
        ur10_robot = world.scene.add(UR10(prim_path="/World/ur10", name="ur10"))
        world.scene.add(
            DynamicCuboid(
                prim_path = '/World/randome_cube',
                name = 'randome_cube',
                position = np.array([0.5, 0.5, 0.5]),
                scale = np.array([0.0515, 0.0515, 0.0515]),
                color = np.array([0.0, 0.0, 1.0])
            )
        )
        print(f"Num of degrees of freedom:  {str(ur10_robot.get_num_dofs())}")
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._ur10_robot = self._world.scene.get_object("ur10")
        self._randome_cube = self._world.scene.get_object("randome_cube")
        self.controller = PickPlaceController(
            name = 'pick_place_controller',
            gripper = self._ur10_robot.gripper,
            robot_articulation = self._ur10_robot,
        )
        self._world.add_physics_callback('sim_step', callback_fn = self.physics_step)
        self._ur10_robot.gripper.set_joint_position(self._ur10_robot.gripper.joint_opened_positions)
        await self._world.play_async()

        return

    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        self._controller.reset()
        self._ur10_robot.gripper.set_joint_position(self._ur10_robot.gripper.joint_opened_positions)
        await self._world.play_async()
        return
    
    def physics_step(self, step_size):
        cube_position = self._randome_cube.get_world_pose()
        goal_position = np.array([-0.3, -0.3, 0.515 / 2.0])
        current_joint_positions = self._ur10_robot.get_joint_positions()
        actions = self._controller.forward(
            picking_position = cube_position,
            placing_position = goal_position,
            current_joint_positions = current_joint_positions,
        )
        self._ur10_robot.apply_action(actions)
        if self._controller.is_done():
            self._world.pause()

    def world_cleanup(self):
        return
