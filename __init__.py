# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name": "BlenderCV",
    "author": "David Hafner",
    "description": "",
    "blender": (2, 90, 0),
    "version": (0, 0, 1),
    "location": "",
    "warning": "",
    "category": "Render",
}

import os
import bpy
from bpy.app.handlers import persistent
import numpy as np


@persistent
def on_render_init(scene):
    print("INIT")

    # TODO: save old values
    global use_nodes
    use_nodes = scene.use_nodes

    global use_pass_z
    use_pass_z = bpy.context.view_layer.use_pass_z

    scene.use_nodes = True

    global render_layers_node
    render_layers_node = scene.node_tree.nodes.new("CompositorNodeRLayers")

    global output_node
    output_node = scene.node_tree.nodes.new("CompositorNodeOutputFile")
    output_node.file_slots.clear()

    output_node.file_slots.new("Image")
    output_node.file_slots["Image"].use_node_format = False
    output_node.file_slots["Image"].format.file_format = "PNG"

    scene.node_tree.links.new(
        render_layers_node.outputs["Image"], output_node.inputs["Image"]
    )

    bpy.context.view_layer.use_pass_z = True

    output_node.file_slots.new("Depth")
    output_node.file_slots["Depth"].use_node_format = False
    output_node.file_slots["Depth"].format.file_format = "OPEN_EXR"

    scene.node_tree.links.new(
        render_layers_node.outputs["Depth"], output_node.inputs["Depth"]
    )


def load_image(img_path):
    img = bpy.data.images.load(img_path)

    data = np.reshape(img.pixels[:], (img.size[1], img.size[0], img.channels))
    data = np.flip(data, 0)

    bpy.data.images.remove(img)

    return data


@persistent
def on_render_post(scene):
    print("POST")

    out = {}

    # TODO: camera
    out["intrinsic"] = None
    out["extrinsic"] = None

    # image
    img_path = f"{output_node.base_path}Image{scene.frame_current:04d}.png"
    data = load_image(img_path)
    out["image"] = data
    os.remove(img_path)

    # depth
    img_path = f"{output_node.base_path}Depth{scene.frame_current:04d}.exr"
    data = load_image(img_path)
    data[data > scene.camera.data.clip_end] = -1
    out["depth"] = data
    os.remove(img_path)

    # save
    np.savez_compressed(f"{output_node.base_path}{scene.frame_current:04d}.npz", **out)


@persistent
def on_render_complete(scene):
    print("COMPLETE")

    # remove nodes
    scene.node_tree.nodes.remove(render_layers_node)
    scene.node_tree.nodes.remove(output_node)

    # restore old values
    scene.use_nodes = use_nodes
    bpy.context.view_layer.use_pass_z = use_pass_z


def register():
    bpy.app.handlers.render_init.append(on_render_init)
    bpy.app.handlers.render_post.append(on_render_post)
    bpy.app.handlers.render_complete.append(on_render_complete)


def unregister():
    bpy.app.handlers.render_init.remove(on_render_init)
    bpy.app.handlers.render_post.remove(on_render_post)
    bpy.app.handlers.render_complete.remove(on_render_complete)


if __name__ == "__main__":
    register()
