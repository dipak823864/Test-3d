import sys
import os
import numpy as np

# Add Library to Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kalpana3d import render_yaml
from kalpana3d.utils.parser import parse_scene
from kalpana3d.core.engine import render_scene as engine_render
from PIL import Image

def render_multi_view(yaml_path, output_base, width=400, height=300):
    print(f"Loading Scene: {yaml_path}")
    scene = parse_scene(yaml_path)
    
    # Views
    views = [
        ('main', scene['camera']['Pos'], scene['camera']['LookAt']),
        ('front', [0, 2, 8], [0, 0, 0]),
        ('back', [0, 2, -8], [0, 0, 0]),
        ('left', [-8, 2, 0], [0, 0, 0]),
        ('right', [8, 2, 0], [0, 0, 0])
    ]
    
    buffer = np.zeros((height * width * 3), dtype=np.float32)
    
    for name, pos, target in views:
        print(f"  Rendering View: {name}")
        
        engine_render(
            width, height, 
            np.array(pos, dtype=np.float32),
            np.array(target, dtype=np.float32),
            float(scene['camera']['FOV']),
            scene['num_objects'],
            scene['object_types'],
            scene['inv_matrices'],
            scene['scales'],
            scene['material_ids'],
            scene['operations'],
            scene['materials'],
            scene['lights'],
            buffer
        )
        
        img_data = (buffer.reshape((height, width, 3)) * 255).astype(np.uint8)
        out_file = f"{output_base}_{name}.png"
        Image.fromarray(img_data).save(out_file)

def main():
    scenes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scenes'))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../gallery/images'))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    tests = [
        '01_basic_primitives.yaml',
        '02_transform_stress.yaml',
        '03_material_study.yaml',
        '04_boolean_logic.yaml',
        '05_hierarchical_chain.yaml',
        '06_organic_blending.yaml',
        '07_infinite_grid.yaml',
        '08_complex_structure.yaml'
    ]
    
    for test in tests:
        input_path = os.path.join(scenes_dir, test)
        output_base = os.path.join(output_dir, test.replace('.yaml', ''))
        
        if os.path.exists(input_path):
            print(f"Running Test Suite: {test}")
            try:
                render_multi_view(input_path, output_base)
            except Exception as e:
                print(f"FAILED: {test} - {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"MISSING: {input_path}")

if __name__ == "__main__":
    main()
