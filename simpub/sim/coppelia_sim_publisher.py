import numpy as np

from ..parser.coppelia_sim import CoppeliasSimParser
from ..parser.coppelia_sim_pyrep import CoppeliasSimPyRepParser
from ..simdata import SimObject, SimScene, SimTransform, SimVisual
from ..simdata import SimMaterial, SimTexture, SimMesh
from ..simdata import VisualType
from ..core.log import logger
from pyrep.backend.sim import *

from typing import List, Dict, Tuple, Optional
import numpy as np

from ..core.simpub_server import SimPublisher
from ..simdata import SimObject


class CoppeliaSimPublisher(SimPublisher):

    def __init__(
        self, sim,
        host: str = "127.0.0.1",
        no_rendered_objects: Optional[List[str]] = None,
        no_tracked_objects: Optional[List[str]] = None,
        visual_layer_list: Optional[List[int]] = None,
        visual_keyword_list: Optional[List[str]] = None,
        use_pyrep: bool = False,
    ) -> None:
        # self.mj_model = mj_model
        # self.mj_data = mj_data
        self.sim = sim
        self.use_pyrep = use_pyrep
        if not use_pyrep:
            self.parser = CoppeliasSimParser(sim, visual_layer_list, visual_keyword_list)
        else:
            self.parser = CoppeliasSimPyRepParser(sim, visual_layer_list, visual_keyword_list)
        sim_scene = self.parser.parse()

        self.tracked_obj_trans: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        super().__init__(
            sim_scene,
            host,
            no_rendered_objects,
            no_tracked_objects,
        )
        self.set_update_objects(self.sim_scene.root, "")

    def set_update_objects(self, obj: Optional[SimObject], parent_id):
        if obj is None:
            return
        if obj.name in self.no_tracked_objects:
            return

        body_id = obj.name
        if body_id == "world":
            pos = np.array([0., 0., 0.])
            quat = np.array([0., 0., 0., 1.])
            body_id = -1
        else:
            body_id = int(body_id)

            if self.use_pyrep:
                pos = simGetObjectPosition(body_id, -1)
                quat = simGetObjectQuaternion(body_id, -1)
            else:
                pos = self.sim.getObjectPosition(body_id, -1)
                quat = self.sim.getObjectQuaternion(body_id, -1)

        trans: Tuple[np.ndarray, np.ndarray] = (pos, quat)
        self.tracked_obj_trans[obj.name] = trans
        for child in obj.children:
            self.set_update_objects(child, parent_id=body_id)

    def get_update(self) -> Dict[str, List[float]]:
        self.set_update_objects(self.sim_scene.root, "")
        try:
            state = {}
            for name, trans in self.tracked_obj_trans.items():
                pos, rot = trans
                state[name] = [
                    -pos[1], pos[2], pos[0], rot[1], -rot[2], -rot[0], rot[3]
                ]
            # print(state['91'])
        except Exception as e:
            logger.error(e)
            import traceback
            traceback.print_exc()
            state = {}
        return state
