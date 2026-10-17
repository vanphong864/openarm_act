"""Compare real and simulated camera frames at identical recorded qpos."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / 'simulation'), str(root / 'src')]
import h5py
import numpy as np
from PIL import Image, ImageDraw
from openarm_sim import DEFAULT_CONFIG, DEFAULT_SCENE
from openarm_sim.scene import load_config, load_scene
from openarm_sim.runner import Sim
from openarm_sim.sensor import CalibratedCamera, ChestCamera
from openarm_flow.pointcloud import Camera

out = root / 'outputs/camera_comparison'
out.mkdir(parents=True, exist_ok=True)
m, _ = load_scene(DEFAULT_SCENE)
sim = Sim(m, load_config(DEFAULT_CONFIG))
cal = Camera.load(root / 'conf/flow_camera.d435i_049322070569.yaml')
camera = CalibratedCamera(m, 'chest_depth', cal)
legacy = ChestCamera(m, 'chest_color', out_size=(640, 480), depth_model=False)
p = sorted((root / 'data/raw/data_openarm_8_10').glob('*.hdf5'))[0]
canvas = Image.new('RGB', (1920, 780))
def depth_image(d):
    value = (np.clip((d.astype(float) - 200) / 1000, 0, 1) * 255).astype('uint8')
    value[d == 0] = 0
    return Image.fromarray(value).convert('RGB')
try:
    with h5py.File(p) as f:
        q = f['observations/qpos']
        for row, index in enumerate([0, len(q)//3, 2*len(q)//3]):
            sim.reset(qpos16=q[index])
            rgb, depth = camera.observe(sim.d)
            old_rgb, old_depth = legacy.observe(sim.d)
            real_rgb = f['observations/images/chest_rgb'][index]
            real_depth = f['observations/images/chest_depth'][index]
            images = [Image.fromarray(real_rgb), Image.fromarray(rgb), Image.fromarray(old_rgb), depth_image(real_depth), depth_image(depth), depth_image(old_depth)]
            labels = ['Real RGB', 'Corrected native depth view', 'Original color view 42deg', 'Real depth', 'Corrected depth', 'Original color-view depth']
            for col, (im, label) in enumerate(zip(images, labels)):
                canvas.paste(im.resize((320,240)), (col*320,row*260+20))
                ImageDraw.Draw(canvas).text((col*320+3,row*260+3), f'{label} frame={index}', fill='white')
            valid = (real_depth>200)&(real_depth<1200)&(depth>200)&(depth<1200)
            print(index, 'overlap depth MAE mm', float(np.abs(real_depth.astype(float)-depth)[valid].mean()))
            common = valid & (old_depth > 200) & (old_depth < 1200)
            # Same pixels for both comparisons; scene mismatch still contributes.
            print('common pixels', int(common.sum()), 'corrected MAE', float(np.abs(real_depth.astype(float)-depth)[common].mean()), 'original MAE', float(np.abs(real_depth.astype(float)-old_depth)[common].mean()))
    print('episode',p.name)
    print('depth camera world xyz',sim.d.cam_xpos[m.camera('chest_depth').id])
    canvas.save(out / 'comparison_variants.png')
finally:
    camera.close()
    legacy.close()
