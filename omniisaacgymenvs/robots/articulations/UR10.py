# Copyright (c) 2018-2022, NVIDIA Corporation
# Copyright (c) 2022-2023, Johnson Sun
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from typing import Optional
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

import carb

class UR10(Robot):
    """
    Represents a UR10 robotic arm in a simulation environment. This class is used to instantiate a UR10 robot
    with specific configurations and add it to the simulation stage.

    Attributes:
        _usd_path (str): The path to the USD file that defines the visual and physical properties of the robot.
        _name (str): The name of the robot instance in the simulation.
        _position (torch.tensor): The initial position of the robot in the simulation environment.
        _orientation (torch.tensor): The initial orientation of the robot in the simulation environment.

    Args:
        prim_path (str): The path in the simulation scene graph where the robot will be added.
        name (Optional[str]): The name assigned to the robot instance. Defaults to "UR10".
        usd_path (Optional[str]): The path to a USD file for the robot. If None, a default path is set.
        translation (Optional[torch.tensor]): The initial position of the robot. If None, defaults to [0.0, 0.0, 0.0].
        orientation (Optional[torch.tensor]): The initial orientation of the robot. If None, defaults to [1.0, 0.0, 0.0, 0.0].
    """
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "UR10",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        """
        Initializes a new instance of the UR10 robot with the specified parameters and adds it to the simulation stage.

        The robot is positioned and oriented according to the provided translation and orientation tensors.
        If no USD path is provided, the default asset path is set. This is based on the assumption that there is a
        pre-configured USD path set up in the simulation environment.

        Raises:
            RuntimeError: If the assets root path cannot be determined when no specific USD path is provided.
        """

        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            #self._usd_path = "omniverse://localhost/Projects/J3soon/Isaac/2023.1.1/Isaac/Robots/UR10/ur10_long_suction.usd"
            self._usd_path = "omniverse://localhost/Projects/J3soon/Isaac/2023.1.1/Isaac/Robots/UR10/ur10_short_suction.usd"
            #self._usd_path = "omniverse://localhost/Projects/J3soon/Isaac/2023.1.1/Isaac/Robots/UR10/ur10_with_hand_e.usd"
            #self._usd_path = 'omniverse://localhost/Projects/J3soon/Isaac/2023.1.1/Isaac/Robots/UR10/ur10_with_2f_140_gripper.usd'
            #self._usd_path = 'omniverse://localhost/Projects/J3soon/Isaac/2023.1.1/Isaac/Robots/UR10/ur10_instanceable.usd'
            #self._usd_path = 'omniverse://localhost/Projects/J3soon/Isaac/2023.1.1/Isaac/Robots/UR10/ur10_gripper_140_instanceable.usd'


        # Depends on your real robot setup
        self._position = torch.tensor([0.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )