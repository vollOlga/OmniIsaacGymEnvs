import os

from omni.isaac.examples.base_sample import BaseSampleExtension
from omni.isaac.examples.user_examples import UR10_pick_and_place


class UR10PickAndPlaceExtension(BaseSampleExtension):
    def on_startup(self, ext_id: str):
        super().on_startup(ext_id)
        super().start_extension(
            menu_name="",
            submenu_name="",
            name="Hello World",
            title="Hello World Example",
            doc_link="https://docs.omniverse.nvidia.com/isaacsim/latest/core_api_tutorials/tutorial_core_hello_world.html",
            overview="This Example introduces the user on how to do cool stuff with Isaac Sim through scripting in asynchronous mode.",
            file_path=os.path.abspath(__file__),
            sample=UR10_pick_and_place(),
        )
        return