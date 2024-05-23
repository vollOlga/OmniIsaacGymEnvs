# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import random

import numpy as np
import omni
import omni.isaac.cortex.math_util as math_util
from omni.isaac.core.objects.capsule import VisualCapsule
from omni.isaac.core.objects.sphere import VisualSphere
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.cortex.cortex_rigid_prim import CortexRigidPrim
from omni.isaac.cortex.cortex_utils import get_assets_root_path
from omni.isaac.cortex.robot import CortexUr10
from omni.isaac.cortex.sample_behaviors.ur10 import bin_stacking_behavior as behavior
from omni.isaac.examples.cortex.cortex_base import CortexBase


class Ur10Assets:
    """
    Class for managing the asset paths for the UR10 robot simulation.
    """
    def __init__(self):
        """
        Initializes the asset paths.
        """
        self.assets_root_path = get_assets_root_path()

        self.ur10_table_usd = (
            self.assets_root_path + "/Isaac/Samples/Leonardo/Stage/ur10_bin_stacking_short_suction.usd"
        )
        self.small_klt_usd = self.assets_root_path + "/Isaac/Props/KLT_Bin/small_KLT.usd"
        self.background_usd = self.assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
        self.rubiks_cube_usd = self.assets_root_path + "/Isaac/Props/Rubiks_Cube/rubiks_cube.usd"


def random_bin_spawn_transform():
    """
    Generates a random transformation (position and orientation) for spawning a bin.

    Returns:
        tuple: A tuple containing the position (numpy array) and orientation (quaternion as list).
    """
    x = random.uniform(-0.15, 0.15)
    y = 1.5
    z = -0.15
    position = np.array([x, y, z])

    z = random.random() * 0.02 - 0.01
    w = random.random() * 0.02 - 0.01
    norm = np.sqrt(z**2 + w**2)
    quat = math_util.Quaternion([w / norm, 0, 0, z / norm])
    if random.random() > 0.5:
        print("<flip>")
        # flip the bin so it's upside down
        quat = quat * math_util.Quaternion([0, 0, 1, 0])
    else:
        print("<no flip>")

    return position, quat.vals


class BinStackingTask(BaseTask):
    """
    Task for bin stacking simulation involving UR10 robot.
    """
    def __init__(self, env_path, assets) -> None:
        """
        Initializes the bin stacking task.

        Args:
            env_path (str): The path to the environment.
            assets (Ur10Assets): The assets for the task.
        """
        super().__init__("bin_stacking")
        self.assets = assets

        self.env_path = env_path
        self.bins = []
        self.stashed_bins = []
        self.on_conveyor = None

    def _spawn_bin(self, rigid_bin):
        """
        Spawns a new bin at a random position and orientation.

        Args:
            rigid_bin (CortexRigidPrim): The bin to spawn.
        """
        x, q = random_bin_spawn_transform()
        rigid_bin.set_world_pose(position=x, orientation=q)
        rigid_bin.set_linear_velocity(np.array([0, -0.30, 0]))
        rigid_bin.set_visibility(True)

    def post_reset(self) -> None:
        """
        Resets the task state, removing all bins and resetting conveyor state.
        """
        if len(self.bins) > 0:
            for rigid_bin in self.bins:
                self.scene.remove_object(rigid_bin.name)
            self.bins.clear()

        self.on_conveyor = None

    def pre_step(self, time_step_index, simulation_time) -> None:
        """
        Checks if a new bin needs to be spawned and spawns it if necessary.

        Args:
            time_step_index (int): The current time step index.
            simulation_time (float): The current simulation time.
        Spawn a new randomly oriented bin if the previous bin has been placed.
        """
        spawn_new = False
        if self.on_conveyor is None:
            spawn_new = True
        else:
            (x, y, z), _ = self.on_conveyor.get_world_pose()
            is_on_conveyor = y > 0.0 and -0.4 < x and x < 0.4
            if not is_on_conveyor:
                spawn_new = True

        if spawn_new:
            name = "bin_{}".format(len(self.bins))
            prim_path = self.env_path + "/bins/{}".format(name)
            add_reference_to_stage(usd_path=self.assets.small_klt_usd, prim_path=prim_path)
            self.on_conveyor = self.scene.add(CortexRigidPrim(name=name, prim_path=prim_path))

            self._spawn_bin(self.on_conveyor)
            self.bins.append(self.on_conveyor)

    def world_cleanup(self):
        """
        Cleans up the world state by resetting the bins and conveyor state.
        """
        self.bins = []
        self.stashed_bins = []
        self.on_conveyor = None
        return


class BinStacking(CortexBase):
    """
    Main class for setting up and running the bin stacking task with the UR10 robot.
    """
    def __init__(self, monitor_fn=None):
        """
        Initializes the BinStacking class.

        Args:
            monitor_fn (callable, optional): Function to monitor the simulation diagnostics.
        """
        super().__init__()
        self._monitor_fn = monitor_fn
        self.robot = None

    def setup_scene(self):
        """
        Sets up the simulation scene, including the robot, environment, and obstacles.
        """
        world = self.get_world()
        env_path = "/World/Ur10Table"
        ur10_assets = Ur10Assets()
        add_reference_to_stage(usd_path=ur10_assets.ur10_table_usd, prim_path=env_path)
        add_reference_to_stage(usd_path=ur10_assets.background_usd, prim_path="/World/Background")
        background_prim = XFormPrim(
            "/World/Background", position=[10.00, 2.00, -1.18180], orientation=[0.7071, 0, 0, 0.7071]
        )
        self.robot = world.add_robot(CortexUr10(name="robot", prim_path="{}/ur10".format(env_path)))

        obs = world.scene.add(
            VisualSphere(
                "/World/Ur10Table/Obstacles/FlipStationSphere",
                name="flip_station_sphere",
                position=np.array([0.73, 0.76, -0.13]),
                radius=0.2,
                visible=False,
            )
        )
        self.robot.register_obstacle(obs)
        obs = world.scene.add(
            VisualSphere(
                "/World/Ur10Table/Obstacles/NavigationDome",
                name="navigation_dome_obs",
                position=[-0.031, -0.018, -1.086],
                radius=1.1,
                visible=False,
            )
        )
        self.robot.register_obstacle(obs)

        az = np.array([1.0, 0.0, -0.3])
        ax = np.array([0.0, 1.0, 0.0])
        ay = np.cross(az, ax)
        R = math_util.pack_R(ax, ay, az)
        quat = math_util.matrix_to_quat(R)
        obs = world.scene.add(
            VisualCapsule(
                "/World/Ur10Table/Obstacles/NavigationBarrier",
                name="navigation_barrier_obs",
                position=[0.471, 0.276, -0.463 - 0.1],
                orientation=quat,
                radius=0.5,
                height=0.9,
                visible=False,
            )
        )
        self.robot.register_obstacle(obs)

        obs = world.scene.add(
            VisualCapsule(
                "/World/Ur10Table/Obstacles/NavigationFlipStation",
                name="navigation_flip_station_obs",
                position=np.array([0.766, 0.755, -0.5]),
                radius=0.5,
                height=0.5,
                visible=False,
            )
        )
        self.robot.register_obstacle(obs)

    async def setup_post_load(self):
        """
        Performs additional setup tasks after the simulation world has been loaded.

        Returns:
            None
        """
        world = self.get_world()
        env_path = "/World/Ur10Table"
        ur10_assets = Ur10Assets()
        if not self.robot:
            self.robot = world._robots["robot"]
            world._current_tasks.clear()
            world._behaviors.clear()
            world._logical_state_monitors.clear()
        self.task = BinStackingTask(env_path, ur10_assets)
        print(world.scene)
        self.task.set_up_scene(world.scene)
        world.add_task(self.task)
        self.decider_network = behavior.make_decider_network(self.robot, self._on_monitor_update)
        world.add_decider_network(self.decider_network)
        return

    def _on_monitor_update(self, diagnostics):
        """
        Callback function to handle updates from the simulation monitor.

        Args:
            diagnostics (str): Diagnostics information from the monitor.
        """
        decision_stack = ""
        if self.decider_network._decider_state.stack:
            decision_stack = "\n".join(
                [
                    "{0}{1}".format("  " * i, element)
                    for i, element in enumerate(str(i) for i in self.decider_network._decider_state.stack)
                ]
            )

        if self._monitor_fn:
            self._monitor_fn(diagnostics, decision_stack)

    def _on_physics_step(self, step_size):
        """
        Callback function to perform actions on each physics step.

        Args:
            step_size (float): The size of the physics step.
        """
        world = self.get_world()
        world.step(False, False)
        return

    async def on_event_async(self):
        """
        Handles asynchronous events in the simulation.

        Returns:
            None
        """
        world = self.get_world()
        await omni.kit.app.get_app().next_update_async()
        world.reset_cortex()
        world.add_physics_callback("sim_step", self._on_physics_step)
        await world.play_async()
        return

    async def setup_pre_reset(self):
        """
        Prepares the simulation for a reset.

        Returns:
            None
        """
        world = self.get_world()
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")
        return

    def world_cleanup(self):
        """
        Cleans up the world state by resetting necessary components.
        """
        return
