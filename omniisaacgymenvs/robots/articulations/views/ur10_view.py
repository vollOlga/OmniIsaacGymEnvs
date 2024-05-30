from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

import torch

class UR10View(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "UR10View",
    ) -> None:

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=False
        )

        # Use RigidPrimView instead of XFormPrimView, since the XForm is not updated when running
        self._end_effectors = RigidPrimView(prim_paths_expr="/World/envs/.*/ur10/ee_link", name="end_effector_view", reset_xform_properties=False)

    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)