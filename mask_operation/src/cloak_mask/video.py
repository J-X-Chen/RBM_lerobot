"""Stream synchronized head/left/right mask and RGB-overlay contact-sheet videos."""
from fractions import Fraction
from pathlib import Path
import av
import cv2
import numpy as np
from .renderer import overlay


class MaskVideos:
    cameras = ('head', 'left', 'right')

    def __init__(self, output, episode, info):
        self.output = Path(output) / f'episode_{episode:04d}'
        self.output.mkdir(parents=True, exist_ok=True)
        self.fps = Fraction(str(info['fps']))
        self.shapes = {n: tuple(info['features'][f'observation.images.{n}']['shape'][:2]) for n in self.cameras}
        self.height = min(s[0] for s in self.shapes.values())
        self.widths = {n: round(w*self.height/h) for n,(h,w) in self.shapes.items()}
        self.width = sum(self.widths.values())
        if self.height % 2 or self.width % 2:
            raise ValueError('Composite dimensions must be even for H.264 yuv420p')
        self.count = 0
        self.items = {}
        try:
            for name in ('masks', 'mask_rgb'):
                dest = self.output / f'{name}.mp4'
                if dest.exists():
                    raise FileExistsError(f'Video exists; choose a new --output: {dest}')
                temp = dest.with_suffix('.partial.mp4')
                if temp.exists():
                    raise FileExistsError(f'Partial video exists: {temp}')
                container = av.open(str(temp), 'w', format='mp4')
                self.items[name] = [container, None, temp, dest]
                stream = container.add_stream('libx264', rate=self.fps)
                stream.width, stream.height = self.width, self.height
                stream.pix_fmt = 'yuv420p'
                stream.options = {'crf': '0' if name == 'masks' else '18', 'preset': 'fast'}
                self.items[name][1] = stream
        except BaseException:
            self.close()
            raise

    def compose(self, masks, rgbs):
        if set(masks) != set(self.cameras) or set(rgbs) != set(self.cameras):
            raise ValueError('Expected head, left and right masks and RGB frames')
        binary, mixed = [], []
        for name in self.cameras:
            mask, rgb = masks[name], rgbs[name]
            if mask.shape != self.shapes[name] or rgb.shape != (*self.shapes[name],3):
                raise ValueError(f'{name}: unexpected mask/RGB resolution')
            size = (self.widths[name], self.height)
            binary.append(cv2.resize((mask > 0).astype(np.uint8)*255, size, interpolation=cv2.INTER_NEAREST))
            mixed.append(cv2.resize(overlay(rgb,mask), size, interpolation=cv2.INTER_AREA))
        return {'masks': np.concatenate(binary, axis=1), 'mask_rgb': np.concatenate(mixed, axis=1)}

    def write(self, masks, rgbs):
        for name, pixels in self.compose(masks, rgbs).items():
            container, stream = self.items[name][:2]
            frame = av.VideoFrame.from_ndarray(pixels, format='gray' if name == 'masks' else 'rgb24')
            frame.pts = self.count
            frame.time_base = 1 / self.fps
            for packet in stream.encode(frame):
                container.mux(packet)
        self.count += 1

    def finish(self):
        if not self.count:
            raise ValueError('Cannot export an empty episode')
        for item in self.items.values():
            container, stream = item[:2]
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            item[0] = None
        for _, _, temp, dest in self.items.values():
            temp.rename(dest)
        return {'frames': self.count, 'fps': float(self.fps),
                'codec': {'masks':'libx264 lossless CRF 0','mask_rgb':'libx264 CRF 18'},
                'layout': 'horizontal: head, left, right', 'height':self.height, 'width':self.width,
                'panel_widths':self.widths, 'pixels': 'masks:0=background,255=robot; mask_rgb:RGB with colored native labels',
                'videos': {n: str(i[3]) for n, i in self.items.items()}}

    def close(self):
        for container, _, temp, _ in self.items.values():
            if container is not None:
                container.close()
            temp.unlink(missing_ok=True)
