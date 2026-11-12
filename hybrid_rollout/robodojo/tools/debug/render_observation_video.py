"""Legacy sparse-observation preview; prefer the native per-control-step video."""
import argparse
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--controller', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fps', type=float, default=25)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--control-dt', type=float, default=0.04,
                   help='simulation control period; used to preserve elapsed time')
    args = p.parse_args()
    obs = sorted((args.controller/'observations').glob('[0-9]'*3))
    if not obs:
        raise SystemExit('no observation directories found')
    frames = []
    selected = obs[::max(1,args.stride)]
    for index, directory in enumerate(selected):
        imgs = [Image.open(directory/name).convert('RGB') for name in
                ('cam_high.png','cam_left_wrist.png','cam_right_wrist.png')]
        w, h = imgs[0].size
        canvas = Image.new('RGB', (w*3, h+32), 'black')
        for i, img in enumerate(imgs): canvas.paste(img, (i*w, 32))
        ImageDraw.Draw(canvas).text((8, 8), directory.name, fill='white')
        # Observations are decision-level, while the simulator advances at
        # control_dt. Hold each image until the next recorded step_id.
        try:
            step = int(__import__('json').loads((directory/'observation.json').read_text())['step_id'])
            next_step = int(__import__('json').loads((selected[index+1]/'observation.json').read_text())['step_id']) \
                if index + 1 < len(selected) else step + 1
            repeats = max(1, round((next_step - step) * args.control_dt * args.fps))
        except (KeyError, ValueError, TypeError, OSError):
            repeats = 1
        frames.extend([np.asarray(canvas)] * repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, frames, fps=args.fps, codec='libx264',
                    macro_block_size=2, output_params=['-preset','fast','-crf','20'])
    print(f'wrote {args.output} ({len(frames)} frames)')

if __name__ == '__main__': main()
