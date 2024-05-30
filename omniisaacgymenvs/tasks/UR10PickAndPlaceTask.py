from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from omniisaacgymenvs.tasks.shared.pick_and_place import PickAndPlace
from omniisaacgymenvs.robots.articulations.views.UR10_view import UR10View
from omniisaacgymenvs.robots.articulations.ur10 import UR10
from omni.isaac.core.utils.prims import get_prim_at_path, add_reference_to_stage
from omni.isaac.core.utils.torch import *
from omni.isaac.gym.vec_env import VecEnvBase
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.visual_sphere import VisualSphere
from omni.isaac.core.visual_capsule import VisualCapsule
from omni.isaac.cortex.robots import CortexUr10
import numpy as np
import torch
import math

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


class UR10PickAndPlaceTask(PickAndPlace):
    def __init__(
            self,
            name: str,
            sim_config: SimConfig,
            env: VecEnvBase,
            offset=None
    ) -> None:
        '''
        Initializes the UR10PickAndPlaceTask instance with the necessary configuration.

        Parameters:
            name (str): The name of the task.
            sim_config (SimConfig): The simulation configuration object containing both simulation and task-specific configurations.
            env (VecEnvBase): The environment in which the task is operating.
            offset (Optional): Additional parameter that can be used to adjust configurations or initial states.

        Return:
            None
        '''
        self._device = "cuda:0"
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self.end_effectors_init_rot = torch.tensor([1, 0, 0, 0], device=self._device)  # w, x, y, z

        self.obs_type = self._task_cfg["env"]["observationType"]
        if not (self.obs_type in ["full"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [full]"
            )
        print("Obs type:", self.obs_type)
        self.num_obs_dict = {
            "full": 43,
            # 6: UR10 joints position (action space)
            # 1: UR10 gripper (position)
            # 6: UR10 joints velocity
            # 1: UR10 gripper velocity
            # 3: goal position
            # 4: goal rotation
            # 4: goal relative rotation
            # 6: previous action
            # 1: previous action gripper
        }
        print(f'')

        self.object_scale = torch.tensor([1.0] * 3)
        self.goal_scale = torch.tensor([2.0] * 3)

        self._num_observations = self.num_obs_dict[self.obs_type]
        self._num_actions = 6
        self._num_states = 0

        pi = math.pi
        if self._task_cfg['safety']['enabled']:
            self._dof_limits = torch.tensor([[
                [np.deg2rad(-135), np.deg2rad(135)],
                [np.deg2rad(-180), np.deg2rad(-60)],
                [np.deg2rad(0), np.deg2rad(180)],
                [np.deg2rad(-180), np.deg2rad(0)],
                [np.deg2rad(-180), np.deg2rad(0)],
                [np.deg2rad(-180), np.deg2rad(180)],
                [-np.inf, np.inf],  # Gripper can be fully open or closed
            ]], dtype=torch.float32, device=self._cfg["sim_device"])
        else:
            self._dof_limits = torch.tensor([[
                [-2 * pi, 2 * pi],  # [-2*pi, 2*pi],
                [-pi + pi / 8, 0 - pi / 8],  # [-2*pi, 2*pi],
                [-pi + pi / 8, pi - pi / 8],  # [-2*pi, 2*pi],
                [-pi, 0],  # [-2*pi, 2*pi],
                [-pi, pi],  # [-2*pi, 2*pi],
                [-2 * pi, 2 * pi],  # [-2*pi, 2*pi],
                [-1, 1],  # Gripper action range
            ]], dtype=torch.float32, device=self._cfg["sim_device"])

        PickAndPlace.__init__(self, name=name, env=env)

        # Setup Sim2Real
        sim2real_config = self._task_cfg['sim2real']
        if sim2real_config['enabled'] and self.test and self.num_envs == 1:
            self.act_moving_average /= 5  # Reduce moving speed
            self.real_world_ur10 = RealWorldUR10(
                sim2real_config['fail_quietely'],
                sim2real_config['verbose']
            )
        return

    def get_num_dof(self):
        '''
        Retrieves the number of degrees of freedom (DOF) for the robot arm.

        Parameters:
            None

        Return:
            int: The number of degrees of freedom of the robot arm.
        '''
        print(f'the number of degrees of freedom (DOF) for the robot arm: {self._arms.num_dof}')
        return self._arms.num_dof

    def get_arm(self):
        '''
        Configures and retrieves an instance of the UR10 robot arm.

        Parameters:
            None

        Return:
            None: The function sets up the UR10 robot within the simulation environment but does not return anything.
        '''
        ur10_assets = Ur10Assets()
        ur10_assets.setup_scene()

    def get_arm_view(self, scene):
        '''
        Creates a view of the UR10 robot arm within the given scene context.

        Parameters:
            scene: The simulation scene in which the robot is visualized or managed.

        Return:
            UR10View: An instance of UR10View that provides an interface to visualize or interact with the robot's configuration.
        '''
        arm_view = UR10View(prim_paths_expr="/World/Ur10Table/ur10", name="ur10_view")
        scene.add(arm_view._end_effectors)
        return arm_view

    def get_object_displacement_tensor(self):
        '''
        Generates a tensor representing the displacement of objects in the environment, used for computations in simulation.

        Parameters:
            None

        Return:
            torch.Tensor: A tensor indicating displacement values for objects in each environment instance.
        '''
        return torch.tensor([0.0, 0.05, 0.0], device=self.device).repeat((self.num_envs, 1))

    def get_observations(self):
        '''
        Retrieves observations from the simulation, depending on the observation type defined in the task configuration.

        Parameters:
            None

        Return:
            dict: A dictionary containing observation buffers for the robot arms.
        '''
        self.arm_dof_pos = self._arms.get_joint_positions()
        self.arm_dof_vel = self._arms.get_joint_velocities()

        if self.obs_type == "full_no_vel":
            self.compute_full_observations(True)
        elif self.obs_type == "full":
            self.compute_full_observations()
        else:
            print("Unknown observations type!")

        observations = {
            self._arms.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def get_reset_target_new_pos(self, n_reset_envs):
        '''
        Computes new target positions for reset environments when resetting part of the simulation environments.

        Parameters:
            n_reset_envs (int): The number of environments to reset.

        Return:
            torch.Tensor: A tensor containing new target positions for each reset environment.
        '''
        new_pos = torch_rand_float(-1, 1, (n_reset_envs, 3), device=self.device)
        if self._task_cfg['sim2real']['enabled'] and self.test and self.num_envs == 1:
            new_pos[:, 0] = torch.abs(new_pos[:, 0] * 0.1) + 0.35
            new_pos[:, 1] = torch.abs(new_pos[:, 1] * 0.1) + 0.35
            new_pos[:, 2] = torch.abs(new_pos[:, 2] * 0.5) + 0.3
        else:
            new_pos[:, 0] = new_pos[:, 0] * 0.4 + 0.5 * torch.sign(new_pos[:, 0])
            new_pos[:, 1] = new_pos[:, 1] * 0.4 + 0.5 * torch.sign(new_pos[:, 1])
            new_pos[:, 2] = torch.abs(new_pos[:, 2] * 0.8) + 0.1
        if self._task_cfg['safety']['enabled']:
            new_pos[:, 0] = torch.abs(new_pos[:, 0]) / 1.25
            new_pos[:, 1] = torch.abs(new_pos[:, 1]) / 1.25
        return new_pos

    def compute_full_observations(self, no_vel=False):
        '''
        Computes and updates the observation buffer with all required observations including joint positions, velocities, and goal information.

        Parameters:
            no_vel (bool): If true, skips the velocity calculations.

        Return:
            None: The method updates the observation buffer in-place and does not return anything.
        '''
        if no_vel:
            raise NotImplementedError()
        else:
            self.obs_buf[:, 0:self.num_arm_dofs] = unscale(self.arm_dof_pos[:, :self.num_arm_dofs],
                                                           self.arm_dof_lower_limits, self.arm_dof_upper_limits)
            self.obs_buf[:, self.num_arm_dofs:2 * self.num_arm_dofs] = self.vel_obs_scale * self.arm_dof_vel[:,
                                                                                            :self.num_arm_dofs]
            base = 2 * self.num_arm_dofs
            self.obs_buf[:, base + 0:base + 3] = self.goal_pos
            self.obs_buf[:, base + 3:base + 7] = self.goal_rot
            self.obs_buf[:, base + 7:base + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
            self.obs_buf[:, base + 11:base + 17] = self.actions

    def send_joint_pos(self, joint_pos):
        '''
        Sends the calculated joint positions to the real UR10 robot if operating in a sim-to-real scenario.

        Parameters:
            joint_pos (torch.Tensor): The joint positions to be sent to the real robot.

        Return:
            None: The function communicates with the real robot but does not return any value.
        '''
        self.real_world_ur10.send_joint_pos(joint_pos)
