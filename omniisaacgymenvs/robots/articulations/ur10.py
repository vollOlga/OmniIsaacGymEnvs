from omni.isaac.universal_robots.ur10 import UR10
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.materials import OmniGlass
from omni.isaac.surface_gripper import SurfaceGripper
class UR10WithGripperAssets(UR10):
    def __init__(self,
                prim_path,
                name,
                position, 
                attach_gripper=True):
        super().__init__(prim_path, name, position)
        #self._usd_path = usd_path
        self.assets_root_path = get_assets_root_path()
        self._name = name

        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            self._usd_path = "omniverse://localhost/Projects/J3soon/Isaac/2023.1.1/Isaac/Robots/UR10/ur10_short_suction.usd"

        if attach_gripper:
            self.attach_gripper()

    def create_ur10(world):
        # Create UR10 robot
        ur10 = world.scene.add(
            UR10(
                prim_path="/World/UR10",
                name="UR10",
                position=np.array([0, 0, 51.5]), 
                attach_gripper=True
                )
            )

        # Set the robot's material to glass
        ur10.set_material(OmniGlass)
