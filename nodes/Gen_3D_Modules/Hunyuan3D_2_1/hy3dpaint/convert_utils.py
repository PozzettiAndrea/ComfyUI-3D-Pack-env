import trimesh
import pygltflib
import numpy as np
from PIL import Image
import base64
import io
import tempfile
import os


def combine_metallic_roughness(metallic_path, roughness_path, output_path):
    """
    Merge the metallic and roughness maps into a single map
    The GLB format requires metallic in the B channel and roughness in the G channel
    """
    # load the maps
    metallic_img = Image.open(metallic_path).convert("L")  # convert to greyscale
    roughness_img = Image.open(roughness_path).convert("L")  # convert to greyscale

    # make sure the sizes match
    if metallic_img.size != roughness_img.size:
        roughness_img = roughness_img.resize(metallic_img.size)

    # create the RGB image
    width, height = metallic_img.size
    combined = Image.new("RGB", (width, height))

    # convert to a numpy array for easier manipulation
    metallic_array = np.array(metallic_img)
    roughness_array = np.array(roughness_img)

    # create the merged array (R, G, B) = (AO, Roughness, Metallic)
    combined_array = np.zeros((height, width, 3), dtype=np.uint8)
    combined_array[:, :, 0] = 255  # R channel: AO (set to white when there is no AO map)
    combined_array[:, :, 1] = roughness_array  # G channel: Roughness
    combined_array[:, :, 2] = metallic_array  # B channel: Metallic

    # convert back to a PIL image and save
    combined = Image.fromarray(combined_array)
    combined.save(output_path)
    return output_path


def create_glb_with_pbr_materials(obj_path, textures_dict, output_path):
    """
    Create a GLB file with full PBR materials using pygltflib

    textures_dict = {
        'albedo': 'path/to/albedo.png',
        'metallic': 'path/to/metallic.png',
        'roughness': 'path/to/roughness.png',
        'normal': 'path/to/normal.png',  # optional
        'ao': 'path/to/ao.png'  # optional
    }
    """
    temp_files = []
    
    try:
        # 1. load the OBJ file
        mesh = trimesh.load(obj_path)

        # 2. first export to a temporary GLB
        with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as temp_glb_file:
            temp_glb = temp_glb_file.name
        temp_files.append(temp_glb)
        mesh.export(temp_glb)

        # 3. load the GLB file for material editing
        gltf = pygltflib.GLTF2().load(temp_glb)

        # 4. prepare texture data
        def image_to_data_uri(image_path):
            """Convert an image to a data URI"""
            with open(image_path, "rb") as f:
                image_data = f.read()
            encoded = base64.b64encode(image_data).decode()
            return f"data:image/png;base64,{encoded}"

        # 5. merge metallic and roughness
        mr_combined_path = None
        if "metallic" in textures_dict and "roughness" in textures_dict:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_mr_file:
                mr_combined_path = temp_mr_file.name
            temp_files.append(mr_combined_path)
            combine_metallic_roughness(textures_dict["metallic"], textures_dict["roughness"], mr_combined_path)
            textures_dict["metallicRoughness"] = mr_combined_path

        # 6. add images to the GLTF
        images = []
        textures = []

        texture_mapping = {
            "albedo": "baseColorTexture",
            "metallicRoughness": "metallicRoughnessTexture",
            "normal": "normalTexture",
            "ao": "occlusionTexture",
        }

        for tex_type, tex_path in textures_dict.items():
            if tex_type in texture_mapping and tex_path and os.path.exists(tex_path):
                # add image
                image = pygltflib.Image(uri=image_to_data_uri(tex_path))
                images.append(image)

                # add texture
                texture = pygltflib.Texture(source=len(images) - 1)
                textures.append(texture)

        # 7. create the PBR material
        pbr_metallic_roughness = pygltflib.PbrMetallicRoughness(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0], metallicFactor=1.0, roughnessFactor=1.0
        )

        # set the texture index
        texture_index = 0
        if "albedo" in textures_dict and os.path.exists(textures_dict["albedo"]):
            pbr_metallic_roughness.baseColorTexture = pygltflib.TextureInfo(index=texture_index)
            texture_index += 1

        if "metallicRoughness" in textures_dict and os.path.exists(textures_dict["metallicRoughness"]):
            pbr_metallic_roughness.metallicRoughnessTexture = pygltflib.TextureInfo(index=texture_index)
            texture_index += 1

        # create the material
        material = pygltflib.Material(name="PBR_Material", pbrMetallicRoughness=pbr_metallic_roughness)

        # add the normal map
        if "normal" in textures_dict and os.path.exists(textures_dict["normal"]):
            material.normalTexture = pygltflib.NormalTextureInfo(index=texture_index)
            texture_index += 1

        # add the AO map
        if "ao" in textures_dict and os.path.exists(textures_dict["ao"]):
            material.occlusionTexture = pygltflib.OcclusionTextureInfo(index=texture_index)

        # 8. update the GLTF
        gltf.images = images
        gltf.textures = textures
        gltf.materials = [material]

        # make sure the mesh uses the material
        if gltf.meshes:
            for primitive in gltf.meshes[0].primitives:
                primitive.material = 0

        # 9. save the final GLB
        gltf.save(output_path)
        print(f"PBR GLB file saved: {output_path}")
        
    except Exception as e:
        print(f"Error creating GLB with PBR materials: {e}")
        raise e
        
    finally:
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Warning: Failed to remove temporary file {temp_file}: {e}")
