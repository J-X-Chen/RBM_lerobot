"""Calibrated pinhole/Brown-Conrady segmentation, preserving original RGB pixels."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MESA_SHADER_CACHE_DISABLE', 'true')

import cv2
import mujoco
import numpy as np

from .config import calibration_errors, feedback_positions

LABELS = {'torso': 1, 'head': 2, 'left_arm': 3, 'right_arm': 4, 'left_gripper': 5, 'right_gripper': 6}
COLORS = np.array([[0,0,0], [160,160,255], [255,200,0], [0,160,255], [255,100,0], [0,255,120], [255,0,180]], np.uint8)


def constrain_wrist_mask(labels, camera):
    """User-required image-space prior: wrist masks only occupy the lower half."""
    if camera in ('left', 'right'):
        labels = labels.copy()
        labels[:(labels.shape[0]+1)//2] = 0
    return labels


def set_frustum(scene, k, width, height):
    """OpenCV pixel centers to OpenGL pixel edges; full fx, fy, cx, cy, no fovy approximation."""
    for camera in scene.camera:
        near = camera.frustum_near
        # MuJoCo 3.17's renderer uses this as half-width. Guarded by the
        # off-centre, unequal-focal-length OpenCV projection regression test.
        camera.frustum_width = near * width / (2 * k[0, 0])
        camera.frustum_center = near * (width / 2 - k[0, 2] - 0.5) / k[0, 0]
        camera.frustum_top = near * (k[1, 2] + 0.5) / k[1, 1]
        camera.frustum_bottom = -near * (height - k[1, 2] - 0.5) / k[1, 1]


def projection_plan(cfg):
    width, height = cfg['width'], cfg['height']
    k = np.array(cfg['K'], dtype=float)
    if cfg['distortion_model'] == 'none':
        return width, height, k, None
    # For every raw pixel find the corresponding undistorted render pixel.
    # Expand the render canvas so distortion never samples a cropped-away edge.
    yy, xx = np.mgrid[:height, :width]
    grid = np.stack([xx, yy], axis=-1).astype(np.float64)
    source = cv2.undistortPoints(grid.reshape(-1,1,2), k,
        np.array(cfg['distortion_coefficients']), P=k).reshape(height,width,2)
    if not np.isfinite(source).all():
        raise ValueError('Distortion maps to non-finite pixels')
    minimum = np.floor(source.min(axis=(0,1))).astype(int) - 2
    maximum = np.ceil(source.max(axis=(0,1))).astype(int) + 2
    rw, rh = (maximum-minimum+1).tolist()
    if rw > 4096 or rh > 4096 or rw * rh > 16_000_000:
        raise ValueError(f'Invalid/extreme distortion render extent {rw}x{rh}; inspect calibration')
    shifted = k.copy()
    shifted[:2, 2] -= minimum
    mapping = (source-minimum).astype(np.float32)
    return rw, rh, shifted, mapping


class AstribotMaskRenderer:
    def __init__(self, visual_model, config, dataset_info):
        if tuple(map(int, mujoco.mj_versionString().split('.')[:2])) < (3, 17):
            raise RuntimeError('Calibrated frustum rendering requires MuJoCo >=3.17 (validated with 3.17.0); use the independent mask environment')
        errors = calibration_errors(config, dataset_info, visual_model)
        if errors:
            raise ValueError('Calibration incomplete:\n- ' + '\n- '.join(errors))
        self.config, self.visual_model = config, visual_model
        self.plans = {name: projection_plan(cfg) for name, cfg in config['cameras'].items()}
        width, height = max(p[0] for p in self.plans.values()), max(p[1] for p in self.plans.values())
        self.model = mujoco.MjModel.from_xml_string(visual_model.mjcf(width, height))
        self.data = mujoco.MjData(self.model)
        self.renderers = {}
        self.geom_labels = np.zeros(self.model.ngeom, np.uint8)
        for name, group in visual_model.geom_groups.items():
            self.geom_labels[self.model.geom(name).id] = LABELS[group]
        self.option = mujoco.MjvOption()
        self.option.sitegroup[:] = 0
        try:
            for name, plan in self.plans.items():
                renderer = mujoco.Renderer(self.model, height=plan[1], width=plan[0])
                renderer.enable_segmentation_rendering()
                self.renderers[name] = renderer
        except Exception:
            self.close()
            raise

    def render(self, state):
        positions = feedback_positions(state, self.config)
        self.visual_model.set_feedback(self.model, self.data, positions)
        result = {}
        camera_id = self.model.camera('calibrated_camera').id
        for name, cfg in self.config['cameras'].items():
            body_id = self.model.body(cfg['parent_link']).id
            parent = np.eye(4)
            parent[:3, :3] = self.data.xmat[body_id].reshape(3,3)
            parent[:3, 3] = self.data.xpos[body_id]
            optical = parent @ np.asarray(cfg['T_parent_optical'])
            self.data.cam_xpos[camera_id] = optical[:3, 3]
            self.data.cam_xmat[camera_id] = (optical[:3,:3] @ np.diag([1,-1,-1])).ravel()
            renderer = self.renderers[name]
            renderer.update_scene(self.data, camera='calibrated_camera', scene_option=self.option)
            width, height, k, mapping = self.plans[name]
            set_frustum(renderer.scene, k, width, height)
            segmentation = renderer.render()
            ids = segmentation[...,0]
            valid = (segmentation[...,1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0) & (ids < self.model.ngeom)
            labels = np.zeros(ids.shape, np.uint8)
            labels[valid] = self.geom_labels[ids[valid]]
            if mapping is not None:
                labels = cv2.remap(labels, mapping[...,0], mapping[...,1], cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            result[name] = constrain_wrist_mask(labels, name)
        return result

    def close(self):
        for renderer in self.renderers.values():
            renderer.close()
        self.renderers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def overlay(rgb, labels, alpha=0.45):
    if rgb.shape[:2] != labels.shape:
        raise ValueError('RGB and mask dimensions differ')
    result = rgb.copy()
    masked = labels > 0
    result[masked] = np.round((1-alpha)*rgb[masked] + alpha*COLORS[labels[masked]]).astype(np.uint8)
    contours, _ = cv2.findContours(masked.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255,255,255), 1)
    return result
