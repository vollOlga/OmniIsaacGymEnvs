from abc import abstractmethod

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.prims import RigidPrimView, XFormPrim
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omni.isaac.core.utils.torch import *
# `scale` maps [-1, 1] to [L, U]; `unscale` maps [L, U] to [-1, 1]
from omni.isaac.core.utils.torch import scale, unscale
from omni.isaac.gym.vec_env import VecEnvBase

from pxr import Sdf, UsdPhysics, UsdShade

import numpy as np
import torch
import math


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
                "Unknown type of observations!\nobservationType should be one of: [full]")
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
        ur10 = UR10(prim_path=self.default_zero_env_path + "/ur10", name="UR10")
        self._sim_config.apply_articulation_settings(
            "ur10",
            get_prim_at_path(ur10.prim_path),
            self._sim_config.parse_actor_config("ur10"),
        )

    def get_arm_view(self, scene):
        '''
        Creates a view of the UR10 robot arm within the given scene context.
        Parameters:
            scene: The simulation scene in which the robot is visualized or managed.
        Return:
            UR10View: An instance of UR10View that provides an interface to visualize or interact with the robot's configuration.
        '''
        arm_view = UR10View(prim_paths_expr="/World/envs/.*/ur10", name="ur10_view")
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
        print(f'joint positions: {joint_pos}')

    def set_up_scene(self, scene: Scene) -> None:
        """
        Sets up the scene for the Reacher task by adding necessary assets and configuring the environment.
        Args:
            scene (Scene): The scene to which the task elements will be added.
        """
        self._stage = get_current_stage()  # Get the current USD stage
        self._assets_root_path = 'omniverse://localhost/Projects/J3soon/Isaac/2023.1.1'  # Path to assets

        # Retrieve and set up arm, object, and goal elements in the scene
        self.get_arm()
        self.get_object()
        self.get_goal()
        self.get_place()

        super().set_up_scene(scene)  # Call to superclass method to complete scene setup

        # Create views for arms, objects, and goals
        self._arms = self.get_arm_view(scene)
        scene.add(self._arms)
        self._objects = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/object/object",
            name="object_view",
            reset_xform_properties=False,
        )
        scene.add(self._objects)
        self._goals = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/goal/object",
            name="goal_view",
            reset_xform_properties=False,
        )
        scene.add(self._goals)

        self._goal_places = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/goal_place/object",
            name="goal_place_view",
            reset_xform_properties=False,
        )
        scene.add(self._goal_places)

    def get_object(self):
        """
        Retrieves and sets up the object in the environment, applying necessary transformations and references.
        """
        self.object_start_translation = torch.tensor([0.1585, 0.0, 0.0], device=self.device)  # Initial position
        self.object_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)  # Initial orientation
        self.object_usd_path = f"{self._assets_root_path}/Isaac/Props/Blocks/block_instanceable.usd"  # USD path for the object

        # Add object to the stage
        add_reference_to_stage(self.object_usd_path, self.default_zero_env_path + "/object")
        obj = XFormPrim(
            prim_path=self.default_zero_env_path + "/object/object",
            name="object",
            translation=self.object_start_translation,
            orientation=self.object_start_orientation,
            scale=self.object_scale
        )
        # Apply configuration settings from simulation to the object
        self._sim_config.apply_articulation_settings("object", get_prim_at_path(obj.prim_path),
                                                     self._sim_config.parse_actor_config("object"))

    def get_goal(self):
        self.goal_displacement_tensor = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.goal_start_translation = torch.tensor([0.0, 0.0, 0.0], device=self.device) + self.goal_displacement_tensor
        self.goal_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        self.goal_usd_path = f"{self._assets_root_path}/Isaac/Props/Blocks/block_instanceable.usd"
        add_reference_to_stage(self.goal_usd_path, self.default_zero_env_path + "/goal")
        goal = XFormPrim(
            prim_path=self.default_zero_env_path + "/goal/object",
            name="goal",
            translation=self.goal_start_translation,
            orientation=self.goal_start_orientation,
            scale=self.goal_scale
        )
        self.goal_scale = torch.tensor([random.uniform(0.5, 4.0), random.uniform(0.5, 4.0), random.uniform(0.5, 4.0)],
                                       device=self.device)
        print(f"Goal scale: {self.goal_scale}")

        self._sim_config.apply_articulation_settings("goal", get_prim_at_path(goal.prim_path),
                                                     self._sim_config.parse_actor_config("goal_object"))

    def get_place(self):
        self.goal_displacement_tensor = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.goal_start_translation = torch.tensor([0.0, 0.0, 0.0], device=self.device) + self.goal_displacement_tensor
        self.goal_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        self.goal_usd_path = f"{self._assets_root_path}/Isaac/Props/Blocks/block_instanceable.usd"
        add_reference_to_stage(self.goal_usd_path, self.default_zero_env_path + "/goal_place")
        goal = XFormPrim(
            prim_path=self.default_zero_env_path + "/goal_place/object",
            name="goal_place",
            translation=self.goal_start_translation,
            orientation=self.goal_start_orientation,
            scale=self.goal_scale
        )
        self.goal_scale = torch.tensor(
            [random.uniform(0.01, 0.5), random.uniform(0.01, 0.5), random.uniform(0.01, 0.5)], device=self.device)
        print(f"Place goal scale: {self.goal_scale}")

        self._sim_config.apply_articulation_settings("goal_place", get_prim_at_path(goal.prim_path),
                                                     self._sim_config.parse_actor_config("goal_place_object"))

    def calculate_accuracy(self, env_ids):
        """
        Calculate the accuracy of the placement by measuring the distance between the object's center and the goal's center.
        Args:
            env_ids: IDs of environments to calculate accuracy for.
        """
        object_centers = self._objects.get_world_poses()[0][
            env_ids]  # Get object positions for the specified environments
        goal_place_centers = self._goal_places.get_world_poses()[0][
            env_ids]  # Get goal_place positions for the specified environments

        # Calculate the Euclidean distance between the object centers and the goal_place centers
        distances = torch.norm(object_centers - goal_place_centers, p=2, dim=-1)

        # Convert distances to millimeters (assuming the distances are in meters)
        distances_mm = distances * 1000

        return distances_mm

    def post_reset(self):
        """
        Actions to perform after environment reset, including setting initial poses and calculating new targets.
        """
        self.num_arm_dofs = self.get_num_dof()  # Retrieve the number of degrees of freedom for the arm
        self.actuated_dof_indices = torch.arange(self.num_arm_dofs, dtype=torch.long,
                                                 device=self.device)  # Indices of actuated degrees of freedom

        # Initialize targets and limits for arm degrees of freedom
        self.arm_dof_targets = torch.zeros((self.num_envs, self._arms.num_dof), dtype=torch.float, device=self.device)

        self.prev_targets = torch.zeros((self.num_envs, self.num_arm_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_arm_dofs), dtype=torch.float, device=self.device)

        dof_limits = self._dof_limits
        self.arm_dof_lower_limits, self.arm_dof_upper_limits = torch.t(dof_limits[0].to(self.device))

        self.arm_dof_default_pos = torch.zeros(self.num_arm_dofs, dtype=torch.float,
                                               device=self.device)  # Default position for arm degrees of freedom
        self.arm_dof_default_vel = torch.zeros(self.num_arm_dofs, dtype=torch.float,
                                               device=self.device)  # Default velocity for arm degrees of freedom

        # Retrieve initial poses for end effectors and goals
        self.end_effectors_init_pos, self.end_effectors_init_rot = self._arms._end_effectors.get_world_poses()

        self.goal_pos, self.goal_rot = self._goals.get_world_poses()
        self.goal_pos -= self._env_pos  # Adjust goal position relative to environment position

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self):
        """
        Calculate and update metrics after each step, including rewards and success tracking.
        """
        self.fall_dist = 0  # Distance fallen, used for calculating fall penalty
        self.fall_penalty = 0  # Penalty for falling, if applicable

        # Compute rewards and update buffers and success counts
        rewards, resets, goal_resets, progress, successes, cons_successes = compute_arm_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes,
            self.consecutive_successes, self.max_episode_length, self.object_pos, self.object_rot,
            self.goal_pos, self.goal_rot, self.dist_reward_scale, self.rot_reward_scale, self.rot_eps,
            self.actions, self.action_penalty_scale, self.success_tolerance, self.reach_goal_bonus,
            self.fall_dist, self.fall_penalty, self.max_consecutive_successes, self.av_factor
        )

        # Update the accumulated rewards and steps
        self.cumulative_rewards += rewards
        self.extras['cumulative reward'] = self.cumulative_rewards
        self.goal_distances += torch.norm(self.object_pos - self.goal_pos, p=2, dim=-1)

        self.episode_lengths += 1

        # Handle resets: calculate average rewards and reset counters
        resets_indices = torch.nonzero(resets).squeeze(-1)
        if len(resets_indices) > 0:
            average_rewards = self.cumulative_rewards[resets_indices] / self.episode_lengths[resets_indices]
            average_distances = self.goal_distances[resets_indices] / self.episode_lengths[resets_indices]

            self.extras['Average reward'] = average_rewards
            self.extras['Average goal distances'] = average_distances

            # Logging and printing for debugging or monitoring
            for idx, avg_reward, avg_dist in zip(resets_indices, average_rewards, average_distances):
                print(
                    f'Episode {self.episode_count}: Environment {idx} - Average Reward: {avg_reward.item()}, Average Goal Distance: {avg_dist.item()}')
                self.episode_count += 1

            # Calculate and log accuracy for successful episodes
            success_indices = resets_indices[successes[resets_indices] > 0]
            if len(success_indices) > 0:
                accuracy_mm = self.calculate_accuracy(success_indices)
                min_accuracy = torch.min(accuracy_mm).item()
                max_accuracy = torch.max(accuracy_mm).item()
                self.extras['min_accuracy_mm'] = min_accuracy
                self.extras['max_accuracy_mm'] = max_accuracy

                print(f'Episode {self.episode_count}: Min Accuracy: {min_accuracy} mm')

            # Reset the cumulative counters for the next episode
            self.cumulative_rewards[resets_indices] = 0
            self.goal_distances[resets_indices] = 0
            self.episode_lengths[resets_indices] = 0

            # Log the timesteps for successful episodes
            for idx in resets_indices:
                if self.successes[idx] > 0:
                    self.success_timesteps.append(self.timesteps_since_start[idx].item())
                    min_timesteps = torch.min(torch.tensor(self.success_timesteps))
                    self.extras[f'timesteps_to_success'] = min_timesteps
                    self.timesteps_since_start[idx] = 0

                    # Update buffers
        self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes = rewards, resets, goal_resets, progress, successes, cons_successes

        # Update extras with average consecutive successes
        self.extras['consecutive_successes'] = cons_successes.mean()

        # Print success statistics if enabled
        if self.print_success_stat:
            self.total_resets += resets.sum()
            direct_average_successes = successes.sum()
            self.total_successes += (successes * resets).sum()
            if self.total_resets > 0:
                print("Direct average consecutive successes = {:.1f}".format(
                    direct_average_successes / self.total_resets))
                print("Post-Reset average consecutive successes = {:.1f}".format(
                    self.total_successes / self.total_resets))

    def pre_physics_step(self, actions):
        """
        Actions to perform before each physics simulation step, including resetting targets and applying actions.
        Args:
            actions: The actions to apply to the environment.
        """
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)  # IDs of environments needing reset
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(
            -1)  # IDs of environments where goals need resetting

        # Retrieve current positions and orientations for end effectors
        end_effectors_pos, end_effectors_rot = self._arms._end_effectors.get_world_poses()

        self.object_pos = end_effectors_pos + quat_rotate(end_effectors_rot,
                                                          quat_rotate_inverse(self.end_effectors_init_rot,
                                                                              self.get_object_displacement_tensor()))
        self.object_pos -= self._env_pos  # subtract world env pos # Adjust object position relative to environment position
        self.object_rot = end_effectors_rot
        object_pos = self.object_pos + self._env_pos
        object_rot = self.object_rot
        self._objects.set_world_poses(object_pos, object_rot)

        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids)
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)  # Clone actions to device
        self.actions[:, 5] = 0.0

        if self.use_relative_control:
            targets = self.prev_targets[:,
                      self.actuated_dof_indices] + self.arm_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.arm_dof_lower_limits[
                                                                              self.actuated_dof_indices],
                                                                          self.arm_dof_upper_limits[
                                                                              self.actuated_dof_indices])
        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions,
                                                                   self.arm_dof_lower_limits[self.actuated_dof_indices],
                                                                   self.arm_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                       self.actuated_dof_indices] + \
                                                             (1.0 - self.act_moving_average) * self.prev_targets[:,
                                                                                               self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
                self.cur_targets[:, self.actuated_dof_indices],
                self.arm_dof_lower_limits[self.actuated_dof_indices],
                self.arm_dof_upper_limits[self.actuated_dof_indices])

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        self._arms.set_joint_position_targets(
            self.cur_targets[:, self.actuated_dof_indices], indices=None, joint_indices=self.actuated_dof_indices
        )
        if self._task_cfg['sim2real']['enabled'] and self.test and self.num_envs == 1:
            cur_joint_pos = self._arms.get_joint_positions(indices=[0], joint_indices=self.actuated_dof_indices)
            joint_pos = cur_joint_pos[0]
            if torch.any(joint_pos < self.arm_dof_lower_limits) or torch.any(joint_pos > self.arm_dof_upper_limits):
                print("get_joint_positions out of bound, send_joint_pos skipped")
            else:
                self.send_joint_pos(joint_pos)
        self.timesteps_since_start += 1

    def reset_target_pose(self, env_ids):
        """
        Resets the target pose for specified environments based on randomization.
        Args:
            env_ids: IDs of environments where targets need resetting.
        """
        indices = env_ids.to(dtype=torch.int32)
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_pos = self.get_reset_target_new_pos(len(env_ids))
        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids],
                                     self.y_unit_tensor[env_ids])

        self.goal_pos[env_ids] = new_pos
        self.goal_rot[env_ids] = new_rot

        goal_pos, goal_rot = self.goal_pos.clone(), self.goal_rot.clone()
        goal_pos[env_ids] = self.goal_pos[env_ids] + self._env_pos[env_ids]

        self._goals.set_world_poses(goal_pos[env_ids], goal_rot[env_ids], indices)
        self.reset_goal_buf[env_ids] = 0

    def reset_idx(self, env_ids):
        """
        Resets specified environments to initial states, including arm and target poses.
        Args:
            env_ids: IDs of environments to reset.
        """
        indices = env_ids.to(dtype=torch.int32)
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_arm_dofs * 2 + 5), device=self.device)

        self.reset_target_pose(env_ids)

        delta_max = self.arm_dof_upper_limits - self.arm_dof_default_pos
        delta_min = self.arm_dof_lower_limits - self.arm_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * (rand_floats[:, 5:5 + self.num_arm_dofs] + 1.0) * 0.5

        pos = self.arm_dof_default_pos + self.reset_dof_pos_noise * rand_delta
        dof_pos = torch.zeros((self.num_envs, self._arms.num_dof), device=self.device)
        dof_pos[env_ids, :self.num_arm_dofs] = pos

        dof_vel = torch.zeros((self.num_envs, self._arms.num_dof), device=self.device)
        dof_vel[env_ids, :self.num_arm_dofs] = self.arm_dof_default_vel + \
                                               self.reset_dof_vel_noise * rand_floats[:,
                                                                          5 + self.num_arm_dofs:5 + self.num_arm_dofs * 2]

        self.prev_targets[env_ids, :self.num_arm_dofs] = pos
        self.cur_targets[env_ids, :self.num_arm_dofs] = pos
        self.arm_dof_targets[env_ids, :self.num_arm_dofs] = pos

        self._arms.set_joint_position_targets(self.arm_dof_targets[env_ids], indices)
        self._arms.set_joint_positions(dof_pos[env_ids], indices)
        self._arms.set_joint_velocities(dof_vel[env_ids], indices)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

        self.cumulative_rewards[env_ids] = 0
        self.goal_distances[env_ids] = 0

        self.timesteps_since_start[env_ids] = 0
        self.accuracy[env_ids] = 0


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    """
    Randomizes rotation based on input random floats and unit tensors for the x and y axes.
    Args:
        rand0: Random float for rotation around the x-axis.
        rand1: Random float for rotation around the y-axis.
        x_unit_tensor: Unit tensor for the x-axis.
        y_unit_tensor: Unit tensor for the y-axis.
    Returns:
        A quaternion representing the randomized rotation.
    """
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def compute_arm_reward(
        rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
        max_episode_length: float, object_pos, object_rot, target_pos, target_rot,
        dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
        actions, action_penalty_scale: float,
        success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
        fall_penalty: float, max_consecutive_successes: int, av_factor: float
):
    """
    Computes rewards for the Reacher task based on distances, orientations, actions, and success conditions.
    Args:
        rew_buf (Tensor): Buffer to store computed rewards.
        reset_buf (Tensor): Buffer indicating which environments need to be reset.
        reset_goal_buf (Tensor): Buffer for resetting goals.
        progress_buf (Tensor): Buffer to track progress of the environments.
        successes (Tensor): Tensor tracking the number of successes.
        consecutive_successes (Tensor): Tensor tracking the number of consecutive successes.
        max_episode_length (float): Maximum length of an episode.
        object_pos (Tensor): Positions of objects in the environments.
        object_rot (Tensor): Orientations of objects in the environments.
        target_pos (Tensor): Target positions in the environments.
        target_rot (Tensor): Target orientations in the environments.
        dist_reward_scale (float): Scaling factor for distance-based rewards.
        rot_reward_scale (float): Scaling factor for rotation-based rewards.
        rot_eps (float): Small epsilon value for rotation calculations to improve stability.
        actions (Tensor): Actions taken by the agents.
        action_penalty_scale (float): Scaling factor for penalties based on the magnitude of actions.
        success_tolerance (float): Tolerance for considering an action successful.
        reach_goal_bonus (float): Bonus given for reaching the goal.
        fall_dist (float): Distance fallen, used for calculating penalties.
        fall_penalty (float): Penalty for falling.
        max_consecutive_successes (int): Maximum number of consecutive successes allowed before a reset.
        av_factor (float): Averaging factor used in computing the rolling average of successes.
    Returns:
        Tuple containing updated rewards, reset states, goal resets, progress states, success counts, and consecutive success counts.
    """

    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)

    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    reward = dist_rew + action_penalty * action_penalty_scale

    goal_resets = torch.where(torch.abs(goal_dist) <= success_tolerance, torch.ones_like(reset_goal_buf),
                              reset_goal_buf)
    successes = successes + goal_resets

    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    resets = reset_buf
    if max_consecutive_successes > 0:
        progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf),
                                   progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor * finished_cons_successes / num_resets + (
                1.0 - av_factor) * consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes

