import yaml
import numpy as np

def make_translation(x, y, z):
    return np.array([[1,0,0,x], [0,1,0,y], [0,0,1,z], [0,0,0,1]], dtype=np.float32)

def make_rotation(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    
    Rx = np.array([[1,0,0,0], [0,cx,-sx,0], [0,sx,cx,0], [0,0,0,1]], dtype=np.float32)
    Ry = np.array([[cy,0,sy,0], [0,1,0,0], [-sy,0,cy,0], [0,0,0,1]], dtype=np.float32)
    Rz = np.array([[cz,-sz,0,0], [sz,cz,0,0], [0,0,1,0], [0,0,0,1]], dtype=np.float32)
    
    return Rz @ Ry @ Rx

def parse_scene(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        
    # Arrays for Numba
    object_types = []
    inv_matrices = []
    scales = []
    material_ids = []
    operations = []
    
    materials = []
    lights = []
    
    # ... (Materials/Lights parsing same) ...
    if 'Materials' not in data:
        materials.append([0.8, 0.8, 0.8, 0.0, 0.5, 0.0, 0.0, 0.0]) # Default Grey
    else:
        for m in data['Materials']:
            # [R, G, B, Metallic, Roughness, EmR, EmG, EmB]
            base = m['Color'] + [m.get('Metallic', 0.0), m.get('Roughness', 0.5)]
            emission = m.get('Emission', [0.0, 0.0, 0.0])
            materials.append(base + emission)
            
    if 'Lights' not in data:
        lights.append([5, 8, 5, 1, 1, 1, 1.0])
    else:
        for l in data['Lights']:
            lights.append(l['Pos'] + l['Color'] + [l['Intensity']])

    # Parse Objects
    for obj in data['Objects']:
        typ = obj['Type']
        pos = obj.get('Pos', [0,0,0])
        rot = obj.get('Rot', [0,0,0])
        scale = obj.get('Scale', [1,1,1])
        mat_id = obj.get('Mat', 0)
        op = obj.get('Op', 0) # 0=Union, 1=Sub, 2=Int
        
        T = make_translation(*pos)
        R = make_rotation(*rot)
        world_mat = T @ R
        inv_mat = np.linalg.inv(world_mat).astype(np.float32)
        
        object_types.append(typ)
        inv_matrices.append(inv_mat)
        scales.append(scale)
        material_ids.append(mat_id)
        operations.append(op)
        
    cam = data.get('Camera', {'Pos': [8, 5, 8], 'LookAt': [0, 2, 0], 'FOV': 50})
        
    return {
        'num_objects': len(object_types),
        'object_types': np.array(object_types, dtype=np.int32),
        'inv_matrices': np.array(inv_matrices, dtype=np.float32),
        'scales': np.array(scales, dtype=np.float32),
        'material_ids': np.array(material_ids, dtype=np.float32),
        'operations': np.array(operations, dtype=np.float32),
        'materials': np.array(materials, dtype=np.float32),
        'lights': np.array(lights, dtype=np.float32),
        'camera': cam
    }
