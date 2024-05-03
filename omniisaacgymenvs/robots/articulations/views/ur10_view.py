
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

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

import torch

class UR10View(ArticulationView):
    """
    A specialized view for interacting with UR10 robotic arms in the simulation environment.

    This class extends `ArticulationView` to manage instances of UR10 robots, providing specific 
    functionalities to handle end effectors associated with these robotic arms.

    Attributes:
        _end_effectors (RigidPrimView): A view for the end effector of the UR10, allowing for
                                        interaction with and manipulation of the end effector's
                                        properties and state within the simulation.

    Args:
        prim_paths_expr (str): The expression used to locate the UR10 robots in the scene. This should
                               match the specific structure of your scene graph.
        name (Optional[str]): An optional name for the view. Defaults to "UR10View".
    """
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "UR10View",
    ) -> None:
        """
        Initializes the UR10 view with the specified path expression and name.

        Calls the superclass initializer and sets up a view for the UR10's end effector
        using `RigidPrimView` to manage rigid transformation properties.

        Args:
            prim_paths_expr (str): The expression to locate UR10 robots in the simulation environment.
            name (Optional[str]): The name of the view. Defaults to "UR10View".
        """

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=False
        )

        # Use RigidPrimView instead of XFormPrimView, since the XForm is not updated when running
        self._end_effectors = RigidPrimView(prim_paths_expr="/World/envs/.*/ur10/ee_link", name="end_effector_view", reset_xform_properties=False)

    def initialize(self, physics_sim_view):
        """
        Initializes the view with a physics simulation view, ensuring that all components are
        properly synchronized with the physics state.

        Args:
            physics_sim_view: The simulation view associated with the physics engine being used.
        """
        super().initialize(physics_sim_view)
