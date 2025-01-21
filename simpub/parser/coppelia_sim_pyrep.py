import tqdm

from pyrep.backend.sim import *
from pyrep.backend.simConst import *
from pyrep.backend._sim_cffi import ffi, lib

from typing import List

import numpy as np
from ..simdata import SimObject, SimScene, SimTransform, SimVisual
from ..simdata import SimMaterial, SimTexture, SimMesh
from ..simdata import VisualType
# from ..core.log import logger
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


sim_handle_world = -1  # The world handle was not defined in PyRep SimConst

def get_scene_obj_type_str(obj_type_id: int):
    # TODO, MAKE THIS A DICT
    # Get obj type name from obj type id
    if obj_type_id == sim_object_shape_type:
        return "shape"
    if obj_type_id == sim_object_joint_type:
        return "joint"
    if obj_type_id == sim_object_graph_type:
        return "graph"
    if obj_type_id == sim_object_camera_type:
        return "camera"
    if obj_type_id == sim_object_light_type:
        return "light"
    if obj_type_id == sim_object_dummy_type:
        return "dummy"
    if obj_type_id == sim_object_proximitysensor_type:
        return "proximitysensor"
    if obj_type_id == sim_object_octree_type:
        return "octree"
    if obj_type_id == sim_object_pointcloud_type:
        return "pointcloud"
    if obj_type_id == sim_object_visionsensor_type:
        return "visionsensor"
    if obj_type_id == sim_object_forcesensor_type:
        return "forcesensor"
    if obj_type_id == sim_object_path_type:
        return "path"
    if obj_type_id == sim_object_mill_type:
        return "mill"
    if obj_type_id == sim_object_mirror_type:
        return "mirror"
    print("")
    raise ValueError(f"Unknown object type id: {obj_type_id}")


def get_primitive_type_str(primitive_shape_id):
    # TODO, MAKE THIS A DICT
    if primitive_shape_id == sim_pure_primitive_none:
        return "none"
    if primitive_shape_id == sim_pure_primitive_plane:
        return "plane"
    if primitive_shape_id == sim_pure_primitive_disc:
        return "disc"
    if primitive_shape_id == sim_pure_primitive_cuboid:
        return "cuboid"
    if primitive_shape_id == sim_pure_primitive_spheroid:
        return "spheroid"
    if primitive_shape_id == sim_pure_primitive_cylinder:
        return "cylinder"
    if primitive_shape_id == sim_pure_primitive_cone:
        return "cone"
    if primitive_shape_id == sim_pure_primitive_heightfield:
        return "heightfield"
    raise ValueError(f"Unknown primitive type: {primitive_shape_id}")


def get_bit_positions(number):
    bit_positions = []
    position = 0
    while number > 0:
        if number & 1:
            bit_positions.append(position)
        number >>= 1
        position += 1
    return bit_positions


def ungroup_compound_objects(visual_layer_list, visual_keyword_list):
    # Exhaust flag
    flag = True

    # Get all objects id in a list.
    objects_id_list = simGetObjectsInTree(sim_handle_scene, sim_handle_all, 0)

    # Ungroup compound objects
    for idx in objects_id_list:

        # Check visualization
        visualize = False
        if visual_layer_list is not None:
            # fixme, the API does not exit
            obj_layers = get_bit_positions(simGetModelProperty(idx, "layer"))
            if bool(set(visual_layer_list) & set(obj_layers)):
                visualize = True
            else:
                visualize = False
        elif visual_keyword_list is not None:
            obj_name = simGetObjectName(idx)
            # make the obj_name lowercase
            if any(keyword.lower() in obj_name.lower() and "floor" not in obj_name.lower()
                   for keyword in visual_keyword_list):
                visualize = True
            else:
                visualize = False
        else:
            visualize = True

        # Check object type
        obj_type_id = simGetObjectType(idx)
        is_shape = get_scene_obj_type_str(obj_type_id) == "shape"

        # In case of a visual shape, Check if shape is compound
        if is_shape and visualize:
            # Official API:
            int_data = ffi.new("int[5]")
            floatData = ffi.new("float[5]")
            void = ffi.NULL
            result = lib.simGetShapeGeomInfo(idx, int_data, floatData, void)
            is_compound = bool(set(get_bit_positions(result)) & {0})
            if is_compound:
                print(simGetObjectName(idx))
                handles = simUngroupShape(idx)
                for h in handles:
                    simSetObjectName(h, f"unity_viz{h}")
                flag = False
    if not flag:
        ungroup_compound_objects(visual_layer_list, visual_keyword_list)


def get_objects_info_dict(cs_sim, visual_layer_list=None,
                          visual_keyword_list=None,
                          name_as_key=False):
    # Ungroup compound objects
    ungroup_compound_objects(visual_layer_list, visual_keyword_list)

    objects_id_list = simGetObjectsInTree(sim_handle_scene, sim_handle_all, 0)

    obj_info_dict = {}

    # tqdm progress bar
    for idx in tqdm.tqdm(objects_id_list):
        obj_name = simGetObjectName(idx)

        # print name and id
        print(f"obj_name:{obj_name}, idx:{idx}")

        visualize = False
        if visual_layer_list is not None:
            # fixme, the API does not exit
            obj_layers = get_bit_positions(simGetModelProperty(idx, "layer"))
            if bool(set(visual_layer_list) & set(obj_layers)):
                visualize = True
            else:
                visualize = False
        elif visual_keyword_list is not None:
            obj_name = simGetObjectName(idx)
            # make the obj_name lowercase
            if any(keyword.lower() in obj_name.lower() and "floor" not in obj_name.lower()
                   for keyword in visual_keyword_list):
                visualize = True
            else:
                visualize = False
        else:
            visualize = True
        try:
            parent_id = str(simGetObjectParent(idx))
        except RuntimeError as e:
            parent_id = "world"

        obj_type_id = simGetObjectType(idx)
        obj_type_str = get_scene_obj_type_str(obj_type_id)
        idx_str = str(idx)
        obj_info_dict[idx_str] = {"name": obj_name,
                                  "parent_id": parent_id,
                                  "type": obj_type_str,
                                  "visualize": visualize}

        if obj_type_str == "shape" and visualize and "floor" not in obj_name.lower():
            int_data = ffi.new("int[5]")
            floatData = ffi.new("float[5]")
            void = ffi.NULL
            result = lib.simGetShapeGeomInfo(idx, int_data, floatData, void)
            pureType = int_data[0]
            dimensions = [floatData[0], floatData[1], floatData[2], floatData[3]] # x, y, z, scale
            primitive_type_str = get_primitive_type_str(pureType)
            vertices, indices, normals = simGetShapeMesh(idx)

            vertices = np.asarray(vertices).reshape(-1, 3)
            indices = np.asarray(indices).reshape(-1, 3)
            num_indices = len(indices)
            normals = np.asarray(normals).reshape(num_indices, 3, 3) # fixme, this normal is not functioning
            # normals = np.asarray(normals).reshape(3, num_indices, 3).swapaxes(0,1)
            normals = normals.mean(axis=1)
            # transfer indices_normals to face_normals

            ambient_diffuse = simGetShapeColor(
                idx, None, sim_colorcomponent_ambient_diffuse)
            diffuse = simGetShapeColor(
                idx, None, sim_colorcomponent_diffuse)
            specular = simGetShapeColor(
                idx, None, sim_colorcomponent_specular)
            emission = simGetShapeColor(
                idx, None, sim_colorcomponent_emission)
            transparency = simGetShapeColor(
                idx, None, sim_colorcomponent_transparency)
            auxiliary = simGetShapeColor(
                idx, None, sim_colorcomponent_auxiliary)

            try:
                texture_id = simGetShapeTextureId(idx)
            except RuntimeError as e:
                texture_id = -1

            if texture_id != -1:
                shape_viz_info = simGetShapeViz(idx, 0)
                texture_data = shape_viz_info.texture
                textureRes = shape_viz_info.textureRes
                texture_width = textureRes[0]
                texture_height = textureRes[1]
                texture_data = np.asarray(texture_data, dtype=np.uint8).reshape(
                    texture_height, texture_width, 4)[..., :-1]
                texture_coord = shape_viz_info.textureCoords
                num_indices = len(indices)
                texture_coord = np.asarray(texture_coord)
                texture_coord = texture_coord.reshape(num_indices, -1)

                texture_repeat_u = None
                texture_repeat_v = None

            else:
                texture_data = None
                texture_width = None
                texture_height = None
                texture_repeat_u = None
                texture_repeat_v = None
                texture_coord = None

            obj_info_dict[idx_str].update({
                "shape_result": result,
                "primitive_type": primitive_type_str,
                "dimensions": dimensions,
                "shape_vertices": vertices,
                "shape_indices": indices,
                "shape_normals": normals,
                "color_ambient_diffuse": ambient_diffuse,
                "color_diffuse": diffuse,
                "color_specular": specular,
                "color_emission": emission,
                "color_transparency": transparency,
                "color_auxiliary": auxiliary,
                "texture_data": texture_data,
                "texture_width": texture_width,
                "texture_height": texture_height,
                "texture_repeat_u": texture_repeat_u,
                "texture_repeat_v": texture_repeat_v,
                "texture_coord": texture_coord,
            })

        # Get Transform info, absolute to world
        if parent_id == "world":
            pos = simGetObjectPosition(idx, sim_handle_world)
            quat = simGetObjectQuaternion(idx, sim_handle_world)
        else:
            pos = simGetObjectPosition(idx, int(parent_id))
            quat = simGetObjectQuaternion(idx, int(parent_id))
        obj_info_dict[idx_str].update({
            "pos": pos,
            "quat": quat
        })

    # Add a virtual root object
    obj_info_dict["world"] = {"name": "world",
                              "visualize": False,
                              "parent_id": -1,
                              "type": "root",
                              "pos": [0, 0, 0],
                              "quat": [0, 0, 0, 1]}

    if name_as_key:
        name_as_key_obj_info_dict = {}
        # todo, when alias repeat, add a counter to the name
        # swap key and id in the value
        for id, info in obj_info_dict.items():
            name = info.pop("name")
            info["id"] = id
            name_as_key_obj_info_dict[name] = info
        return name_as_key_obj_info_dict

    else:
        return obj_info_dict


def coppelia_sim2unity_pos(pos: List[float]) -> List[float]:
    return [-pos[1], pos[2], pos[0]]


def coppelia_sim2unity_quat(quat: List[float]) -> List[float]:
    # return [quat[2], -quat[3], -quat[1], quat[0]] # Mujoco
    return [quat[1], -quat[2], -quat[0], quat[3]]  # Vrep


class CoppeliasSimPyRepParser:
    def __init__(self, coppelia_sim, visual_layer_list, visual_keyword_list):
        self.sim_scene = None
        self.visual_layer_list = visual_layer_list
        self.visual_keyword_list = visual_keyword_list
        if visual_keyword_list is not None:
            visual_keyword_list.append("unity_viz")
        self.parse_scene(coppelia_sim)
        self.sim_scene.process_sim_obj(self.sim_scene.root)

    def parse(self):
        return self.sim_scene

    def parse_scene(self, cs_sim) -> SimScene:
        """
        Parse CoppeliaSim scene to SimScene
        """

        # Create SimScene
        sim_scene = SimScene()
        self.sim_scene = sim_scene

        # Info dict
        info_dict_id_as_key = get_objects_info_dict(
            cs_sim, self.visual_layer_list, self.visual_keyword_list, name_as_key=False)

        # Build the hierarchy tree
        body_hierarchy = {}
        for obj_id, obj_info in info_dict_id_as_key.items():
            sim_object = self.process_body(obj_id, obj_info)

            body_hierarchy[obj_id] = {
                "parent_id": obj_info["parent_id"],
                "sim_object": sim_object,
            }

        # create a tree structure from the body hierarchy
        for body_id, body_info in body_hierarchy.items():
            parent_id = body_info["parent_id"]
            if parent_id == -1:
                continue
            if parent_id in body_hierarchy:
                parent_info = body_hierarchy[parent_id]
                parent_object: SimObject = parent_info["sim_object"]
                parent_object.children.append(body_info["sim_object"])

        return sim_scene

    def process_body(self, obj_id, obj_info):

        # Create SimObject
        body_name = str(obj_id)
        object_name = obj_info["name"]
        sim_object = SimObject(name=body_name)

        # Check parent
        parent_id = obj_info["parent_id"]
        if parent_id == -1:
            self.sim_scene.root = sim_object

        # Get Transform info, absolute to world
        trans = sim_object.trans
        trans.pos = coppelia_sim2unity_pos(obj_info["pos"])
        trans.rot = coppelia_sim2unity_quat(obj_info["quat"])

        # Geometry info
        obj_type = obj_info["type"]
        obj_visualize = obj_info["visualize"]

        # Fixme, floor is special
        # if obj_type != "shape" or not obj_visualize or "Floor" in object_name:
        if "floor" in body_name.lower() or obj_type != "shape" or not obj_visualize:
            return sim_object
        else:  # visual shape

            # Process Material / Texture
            ambient_diffuse = obj_info["color_ambient_diffuse"]
            diffuse = obj_info["color_diffuse"]
            specular = obj_info["color_specular"]
            emission = obj_info["color_emission"]
            transparency = obj_info["color_transparency"]
            auxiliary = obj_info["color_auxiliary"]

            # Texture
            texture_data = obj_info["texture_data"]
            if texture_data is not None:
                # texture_data = np.asarray(texture_data, dtype=np.uint8)
                texture_width = obj_info["texture_width"]
                texture_height = obj_info["texture_height"]
                texture_repeat_u = obj_info["texture_repeat_u"]
                texture_repeat_v = obj_info["texture_repeat_v"]
                texture_coord = obj_info["texture_coord"]
                mat_texture = SimTexture.create_texture(
                    texture_data, texture_height,
                    texture_width, self.sim_scene)
                mat_texture.textureScale = [1, 1]
                plt.imshow(texture_data)
            else:
                mat_texture = None
                texture_coord = None
            # mat_texture = None

            # FIXME, THESE COLOR PROPERTIES MISMATCH BETWEEN VREP AND UNITY
            # FIXME, THE COLOR RENDERING HAS ISSUE, ESPECIALLY FOR THE FLOOR
            # Fixme, Setting transparency and emission are buggy
            material = SimMaterial(color=ambient_diffuse,
                                   emissionColor=[0., 0., 0., 0.],
                                   # specular=specular,
                                   # shininess=mat_shininess,
                                   # reflectance=mat_reflectance,
                                   texture=mat_texture)

            # Process Mesh
            vertices = obj_info["shape_vertices"]
            indices = obj_info["shape_indices"]
            normals = obj_info["shape_normals"]  # fixme, this is not being used
            sim_trans = SimTransform()  # todo, specify some arguments here?
            sim_visual = SimVisual(
                name=body_name,
                type=VisualType.MESH,
                trans=sim_trans,
                # material=SimMaterial(color=[0.5, 0.5, 0.5]),
                material=material,
            )
            sim_visual.mesh = SimMesh.create_mesh(
                scene=self.sim_scene,
                vertices=vertices,
                faces=indices,
                face_normals=normals,
                # mesh_texcoord=texture_coord, # Fixme, texture details?
                faces_uv=texture_coord,
            )
            sim_object.visuals.append(sim_visual)
            return sim_object
