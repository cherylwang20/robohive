from typing import List

import mujoco
import numpy as np
from numpy.linalg import pinv, inv
import glfw
from collections import deque
import enum
from functools import partial

xml = './ur10e/scene_chem_vel.xml' 
contacts = deque(maxlen=1000)

class ObjLabels(enum.Enum):
    LEFT_GRIP = 0
    RIGHT_GRIP = 1
    ENV = 2

class BodyIdInfo:
    def __init__(self, model: mujoco.MjModel):

        left_bodies = [model.body(i).id for i in range(model.nbody) if model.body(i).name.startswith("Lgrip/")]
        self.left_range = (min(left_bodies), max(left_bodies))

        right_bodies = [model.body(i).id for i in range(model.nbody) if model.body(i).name.startswith("Rgrip/")]
        self.right_range = (min(right_bodies), max(right_bodies))

def arm_control(model: mujoco.MjModel, data: mujoco.MjData, id_info: BodyIdInfo):
    # `model` contains static information about the modeled system, e.g. their indices in dynamics matrices
    # `data` contains the current dynamic state of the system
    touching_objects = set(get_touching_objects(model, data, id_info))
    contacts.append(touching_objects)
    return touching_objects

def get_touching_objects(model: mujoco.MjModel, data: mujoco.MjData, id_info: BodyIdInfo, object_name):
    object_id = model.body(object_name).id
    for con in data.contact:
        if model.geom(con.geom1).bodyid == object_id:
            yield body_id_to_label(model.geom(con.geom2).bodyid, id_info)
        elif model.geom(con.geom2).bodyid == object_id:
            yield body_id_to_label(model.geom(con.geom1).bodyid, id_info)


def body_id_to_label(body_id, id_info: BodyIdInfo):
    #print(id_info.left_range[0], id_info.right_range[0], body_id)
    if id_info.left_range[0]  - 1 <= body_id < id_info.left_range[1]:
        return ObjLabels.LEFT_GRIP
    elif id_info.right_range[0] - 1 <= body_id < id_info.right_range[1]:
        return ObjLabels.RIGHT_GRIP
    else:
        return ObjLabels.ENV

def load_callback(model=None, data=None, filename = None):
    # Clear the control callback before loading a new model
    # or a Python exception is raised
    mujoco.set_mjcb_control(None)

    # `model` contains static information about the modeled system
    model = mujoco.MjModel.from_xml_path(filename=xml, assets=None)

    # `data` contains the current dynamic state of the system
    data = mujoco.MjData(model)

    id_info = BodyIdInfo(model)

    if model is not None:
        # Can set initial state

        # The provided "callback" function will be called once per physics time step.
        # (After forward kinematics, before forward dynamics and integration)
        # see https://mujoco.readthedocs.io/en/stable/programming.html#simulation-loop for more info
        mujoco.set_mjcb_control(partial(arm_control, id_info=id_info))

    return model, data

