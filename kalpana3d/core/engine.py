import numpy as np
from numba import njit, prange
from .math import vec3, normalize, dot, cross, length, mix, clamp, reflect
from ..shapes.sdf import sdSphere, sdBox, sdRoundBox, sdCylinder, sdPlane, sdTorus, sdCapsule, sdCone, sdHexPrism
from ..shapes.sdf_extras import sdMandelbulb, sdSierpinski

# CONSTANTS
MAX_STEPS = 256
MAX_DIST = 100.0
SURF_DIST = 0.001
PI = 3.14159265359

# SHAPE TYPES
SHAPE_SPHERE = 0
SHAPE_BOX = 1
SHAPE_CYLINDER = 2
SHAPE_ROUNDBOX = 3
SHAPE_PLANE = 4
SHAPE_TORUS = 5
SHAPE_CAPSULE = 6
SHAPE_CONE = 7
SHAPE_HEXPRISM = 8
SHAPE_MANDELBULB = 9
SHAPE_SIERPINSKI = 10

# OPERATIONS
OP_UNION = 0
OP_SUBTRACT = 1
OP_INTERSECT = 2
OP_SMOOTH_UNION = 3
OP_SMOOTH_SUBTRACT = 4
OP_SMOOTH_INTERSECT = 5

@njit(fastmath=True)
def smin(a, b, k):
    h = max(k - abs(a - b), 0.0) / k
    return min(a, b) - h * h * k * (1.0 / 4.0)

@njit(fastmath=True)
def transform_point(inv_mat, p):
    x = inv_mat[0,0]*p[0] + inv_mat[0,1]*p[1] + inv_mat[0,2]*p[2] + inv_mat[0,3]
    y = inv_mat[1,0]*p[0] + inv_mat[1,1]*p[1] + inv_mat[1,2]*p[2] + inv_mat[1,3]
    z = inv_mat[2,0]*p[0] + inv_mat[2,1]*p[1] + inv_mat[2,2]*p[2] + inv_mat[2,3]
    return vec3(x, y, z)

@njit(fastmath=True)
def map_scene(p, num_objects, object_types, inv_matrices, scales, material_ids, operations):
    d_final = MAX_DIST
    m_final = -1.0
    
    for i in range(num_objects):
        op = int(operations[i])
        local_p = transform_point(inv_matrices[i], p)
        
        # Domain Repetition (Infinite Grid)
        if op == 6: # OP_REPEAT
            # Repeat every 4.0 units
            s = 4.0
            local_p[0] = (local_p[0] + s*0.5) % s - s*0.5
            local_p[1] = (local_p[1] + s*0.5) % s - s*0.5
            local_p[2] = (local_p[2] + s*0.5) % s - s*0.5
            
        d = MAX_DIST
        typ = object_types[i]
        scale = scales[i]
        
        if typ == SHAPE_SPHERE: d = sdSphere(local_p, scale[0])
        elif typ == SHAPE_BOX: d = sdBox(local_p, scale)
        elif typ == SHAPE_CYLINDER: d = sdCylinder(local_p, scale[1], scale[0])
        elif typ == SHAPE_ROUNDBOX: d = sdRoundBox(local_p, scale, 0.1)
        elif typ == SHAPE_PLANE: d = local_p[1]
        elif typ == SHAPE_TORUS: d = sdTorus(local_p, vec3(scale[0], scale[1], 0))
        elif typ == SHAPE_CAPSULE: d = sdCapsule(local_p, vec3(0, -scale[1]/2, 0), vec3(0, scale[1]/2, 0), scale[0])
        elif typ == SHAPE_CONE: d = sdCone(local_p, vec3(scale[0], scale[1], 0), scale[2])
        elif typ == SHAPE_HEXPRISM: d = sdHexPrism(local_p, vec3(scale[0], scale[1], 0))
        # elif typ == SHAPE_MANDELBULB: d = sdMandelbulb(local_p)
        # elif typ == SHAPE_SIERPINSKI: d = sdSierpinski(local_p)
        
        mat = material_ids[i]
        
        if i == 0:
            d_final = d
            m_final = mat
        else:
            if op == OP_UNION or op == 6: # Treat Repeat as Union
                if d < d_final:
                    d_final = d
                    m_final = mat
            elif op == OP_SUBTRACT:
                if -d > d_final:
                    d_final = -d
            elif op == OP_INTERSECT:
                if d > d_final:
                    d_final = d
                    m_final = mat
            elif op == OP_SMOOTH_UNION:
                k = 0.5 # Smoothness factor
                d_final = smin(d_final, d, k)
                if d < d_final + 0.1: # Approximate material blending
                     m_final = mat
            elif op == OP_SMOOTH_SUBTRACT:
                k = 0.1
                h = max(k - abs(-d - d_final), 0.0) / k
                d_final = max(-d, d_final) + h * h * k * (1.0 / 4.0)
            elif op == OP_SMOOTH_INTERSECT:
                k = 0.1
                h = max(k - abs(d - d_final), 0.0) / k
                d_final = max(d, d_final) - h * h * k * (1.0 / 4.0)
                if d > d_final - 0.1:
                    m_final = mat
                    
    return d_final, m_final

@njit(fastmath=True)
def calc_normal(p, num_objects, object_types, inv_matrices, scales, material_ids, operations):
    e = vec3(SURF_DIST, 0.0, 0.0)
    d = map_scene(p, num_objects, object_types, inv_matrices, scales, material_ids, operations)[0]
    n = vec3(
        d - map_scene(p - vec3(e[0], 0, 0), num_objects, object_types, inv_matrices, scales, material_ids, operations)[0],
        d - map_scene(p - vec3(0, e[0], 0), num_objects, object_types, inv_matrices, scales, material_ids, operations)[0],
        d - map_scene(p - vec3(0, 0, e[0]), num_objects, object_types, inv_matrices, scales, material_ids, operations)[0]
    )
    return normalize(n)

@njit(fastmath=True)
def calc_softshadow(ro, rd, tmin, tmax, k, num_objects, object_types, inv_matrices, scales, material_ids, operations):
    res = 1.0
    t = tmin
    for i in range(16):
        h = map_scene(ro + rd * t, num_objects, object_types, inv_matrices, scales, material_ids, operations)[0]
        res = min(res, k * h / t)
        t += clamp(h, 0.02, 0.2)
        if res < 0.005 or t > tmax: break
    return clamp(res, 0.0, 1.0)

@njit(fastmath=True)
def calc_ao(p, n, num_objects, object_types, inv_matrices, scales, material_ids, operations):
    occ = 0.0
    sca = 1.0
    for i in range(5):
        h = 0.01 + 0.12 * float(i) / 4.0
        d = map_scene(p + n * h, num_objects, object_types, inv_matrices, scales, material_ids, operations)[0]
        occ += (h - d) * sca
        sca *= 0.95
        if sca < 0.1: break
    return clamp(1.0 - 3.0 * occ, 0.0, 1.0)

@njit(fastmath=True)
def get_sky(rd, roughness):
    # Improved Skybox with Sun
    sun_dir = normalize(vec3(0.5, 0.8, 0.5))
    sun_dot = max(dot(rd, sun_dir), 0.0)
    
    # Blur sun based on roughness
    sun_exp = mix(512.0, 1.0, roughness)
    sun = np.float32(pow(sun_dot, sun_exp))
    
    # Fade out sun for rough materials
    sun_fade = clamp(1.0 - roughness * 1.5, 0.0, 1.0)
    sun_intensity = 1.0 * sun_fade
    
    t = np.float32(0.5) * (rd[1] + np.float32(1.0))
    
    # Richer Sky Gradient
    col_a = vec3(0.5, 0.7, 0.9) # Zenith Blue
    col_b = vec3(0.9, 0.9, 0.9) # Horizon White
    sky = col_a * (np.float32(1.0) - t) + col_b * t
    
    # Fake Horizon Detail (Simple Band) to break up the reflection
    horizon = clamp(1.0 - abs(rd[1]) * 5.0, 0.0, 1.0)
    sky = mix(sky, vec3(0.2, 0.2, 0.3), horizon * 0.5)
    
    # Distinct Ground for Reflection
    if rd[1] < 0.0:
        ground_col = vec3(0.1, 0.1, 0.1) # Dark Ground
        sky = mix(sky, ground_col, clamp(-rd[1] * 5.0, 0.0, 1.0))
    
    avg_sky = vec3(0.5, 0.6, 0.7)
    sky = mix(sky, avg_sky, roughness)
    
    # HIGH-END PRODUCT RENDER STYLE
    
    # Dark, Moody Background (Dark Blue-Grey)
    col_top = vec3(0.1, 0.12, 0.15)
    col_bot = vec3(0.05, 0.05, 0.05)
    t = np.float32(0.5) * (rd[1] + np.float32(1.0))
    sky = col_top * (np.float32(1.0) - t) + col_bot * t
    
    # Bright Sun for Highlights
    sun_dir = normalize(vec3(0.5, 0.8, 0.5))
    sun_dot = max(dot(rd, sun_dir), 0.0)
    sun = np.float32(pow(sun_dot, 256.0)) # Sharp sun
    
    # Floor Reflection (Checkerboard)
    if rd[1] < 0.0:
        # Reflective Floor Logic
        # We can't trace secondary rays easily in this engine structure without recursion limits.
        # So we fake it by mixing the sky color with a checkerboard pattern.
        
        # Simple Grid Pattern on Floor
        scale = 2.0
        p_floor = rd * (1.0 / -rd[1]) # Project ray to y=-1 (approx)
        cx = np.floor(p_floor[0] * scale)
        cz = np.floor(p_floor[2] * scale)
        checker = (cx + cz) % 2.0
        
        floor_col = vec3(0.1, 0.1, 0.1)
        if checker != 0.0:
             floor_col = vec3(0.15, 0.15, 0.15)
             
        # Fog/Fade to distance
        dist = length(p_floor)
        fog = clamp(1.0 - dist * 0.1, 0.0, 1.0)
        
        sky = mix(sky, floor_col, fog)

    return sky + vec3(1.0, 0.9, 0.8) * sun * 2.0 # Warm Sun Highlight

@njit(fastmath=True)
def get_material_albedo(p, mat_id, materials):
    mat_idx = int(mat_id)
    albedo = materials[mat_idx, 0:3]
    
    # Procedural Checkerboard for Floor (Material ID 3)
    if mat_idx == 3:
        cx = np.floor(p[0])
        cz = np.floor(p[2])
        if (cx + cz) % 2.0 != 0.0:
            albedo = albedo * vec3(0.5, 0.5, 0.5)
    return albedo

@njit(fastmath=True)
def trace_reflection(ro, rd, num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, roughness):
    dO = 0.02
    hit = False
    hit_m = -1.0
    hit_p = vec3(0.0, 0.0, 0.0)
    
    for i in range(16): # Reduced steps for speed
        p = ro + rd * dO
        d, m = map_scene(p, num_objects, object_types, inv_matrices, scales, material_ids, operations)
        if d < 0.005:
            hit = True
            hit_m = m
            hit_p = p
            break
        dO += d
        if dO > 5.0: # Reduced distance
            break
            
    if hit:
        return get_material_albedo(hit_p, hit_m, materials)
    else:
        return vec3(-1.0, -1.0, -1.0) # Miss Sentinel

@njit(fastmath=True)
def get_lighting(p, n, rd, mat_id, materials, lights, num_objects, object_types, inv_matrices, scales, material_ids, operations):
    mat_idx = int(mat_id)
    albedo = get_material_albedo(p, mat_id, materials)
    metallic = materials[mat_idx, 3]
    roughness = materials[mat_idx, 4]
    emission = materials[mat_idx, 5:8]
    
    col = vec3(0.0, 0.0, 0.0)
    col += emission * 5.0
    
    # Ambient Occlusion
    ao = calc_ao(p, n, num_objects, object_types, inv_matrices, scales, material_ids, operations)
    
    # Ambient
    # FIX: Add "Metal Ambient" (Fake GI)
    # Physically, metals are black in diffuse. Artistically, we want them to show color in shadow.
    metal_ambient = albedo * 0.3 * metallic 
    dielectric_ambient = albedo * 0.2 * (1.0 - metallic)
    
    col += (metal_ambient + dielectric_ambient) * ao
    
    # Direct Light (Loop through all lights)
    for i in range(lights.shape[0]):
        l_pos = lights[i, 0:3]
        l_col = lights[i, 3:6]
        l_int = lights[i, 6]
        
        l = normalize(l_pos - p)
        ndotl = max(dot(n, l), 0.0)
        
        # Shadow (Only for the first/main light to save perf)
        shadow = 1.0
        if i == 0:
             # Soft Shadows (k=4.0 for studio look, was 8.0)
             shadow = calc_softshadow(p + n * 0.01, l, 0.02, length(l_pos - p), 4.0, num_objects, object_types, inv_matrices, scales, material_ids, operations)
        
        # Specular (Blinn-Phong)
        v = -rd
        h = normalize(l + v)
        ndoth = max(dot(n, h), 0.0)
        
        # Specular Power
        spec_pow = mix(256.0, 10.0, roughness)
        spec = pow(ndoth, spec_pow) * (1.0 - roughness)
        
        # PBR Specular Color (Fresnel-ish)
        f0 = mix(vec3(0.04, 0.04, 0.04), albedo, metallic)
        
        # Combine Diffuse + Specular
        diffuse_factor = (1.0 - metallic)
        
        # Add Light Contribution
        col += (albedo * diffuse_factor * ndotl + f0 * spec) * l_col * l_int * 0.8 * shadow
        
    # Environment Reflection
    ref = reflect(rd, n)
    
    # Trace Reflection (Disabled due to Numba limits)
    # We use the improved Skybox with Sun and Ground
    sky_col = get_sky(ref, roughness)
    
    # Fresnel (Schlick)
    f0 = mix(vec3(0.04, 0.04, 0.04), albedo, metallic)
    reflection = sky_col * f0
    
    col += reflection * ao
    
    return col

@njit(fastmath=True)
def raymarch_kernel(ro, rd, num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, lights):
    dO = 0.0
    for i in range(MAX_STEPS):
        p = ro + rd * dO
        d, m = map_scene(p, num_objects, object_types, inv_matrices, scales, material_ids, operations)
        
        if d < SURF_DIST:
            n = calc_normal(p, num_objects, object_types, inv_matrices, scales, material_ids, operations)
            return get_lighting(p, n, rd, m, materials, lights, num_objects, object_types, inv_matrices, scales, material_ids, operations)
        
        dO += d
        if dO > MAX_DIST:
            break
            
    return np.array([0.05, 0.05, 0.08], dtype=np.float32) # Sky Color

@njit(parallel=True, fastmath=True)
def render_scene(width, height, cam_pos, cam_target, fov, 
                 num_objects, object_types, inv_matrices, scales, material_ids, operations,
                 materials, lights, output_buffer):
    
    ro = cam_pos
    lookat = cam_target
    f = normalize(lookat - ro)
    r = normalize(cross(f, vec3(0, 1, 0)))
    u = cross(r, f)
    zoom = 1.0 / np.tan((fov * np.pi / 180.0) / 2.0)
    
    for y in prange(height):
        for x in range(width):
            col_acc = vec3(0.0, 0.0, 0.0)
            
            # 4x Super-Sampling (SSAA)
            # Sample 4 sub-pixel positions: (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)
            for dy in [0.25, 0.75]:
                for dx in [0.25, 0.75]:
                    uv_x = ((x + dx) - width / 2.0) / height
                    uv_y = -((y + dy) - height / 2.0) / height
                    
                    rd = normalize(f * zoom + r * uv_x + u * uv_y)
                    
                    col_acc += raymarch_kernel(ro, rd, num_objects, object_types, inv_matrices, scales, material_ids, operations, materials, lights)
            
            col = col_acc * 0.25 # Average the 4 samples
            
            # VIVID STANDARD MAPPING (No ACES)
            col = col * 1.0 # Standard Exposure
            col = np.maximum(np.minimum(col, 1.0), 0.0) # Simple Clamp
            
            # Gamma Correction
            col = np.power(col, 1.0/2.2)
            
            idx = (y * width + x) * 3
            output_buffer[idx] = col[0]
            output_buffer[idx+1] = col[1]
            output_buffer[idx+2] = col[2]
