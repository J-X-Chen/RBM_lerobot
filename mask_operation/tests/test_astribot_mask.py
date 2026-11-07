"""Renderer checks use synthetic calibration ONLY, never real S1 camera claims."""
import copy
from pathlib import Path

import cv2
import numpy as np
import pytest

from cloak_mask.config import calibration_errors, feedback_positions, template
from cloak_mask.dataset import DatasetFrames
from cloak_mask.model import AstribotVisualModel
from cloak_mask.renderer import mujoco, projection_plan, set_frustum
from cloak_mask.sdk import GROUPS, read_current_feedback

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def visual_model():
    return AstribotVisualModel(ROOT/'astribot_sdk_ros2')


@pytest.fixture(scope='module')
def dataset():
    return DatasetFrames(ROOT/'lerobotv3_dataset/lerobotv3_red_cup_only')


def test_missing_calibration_is_explicit(visual_model, dataset):
    draft = template(dataset.info,visual_model)
    errors = calibration_errors(draft,dataset.info,visual_model)
    assert any('confirm measured feedback' in e for e in errors)
    for camera in ('head','left','right'):
        assert any(f'cameras.{camera}.K:' in e for e in errors)
        assert draft['cameras'][camera]['K'] is None
    for side in ('left','right'):
        assert any(f'grippers.{side}.samples' in e for e in errors)


def test_full_mesh_model_and_independent_arm_fk(visual_model):
    model = mujoco.MjModel.from_xml_string(visual_model.mjcf(320,240))
    data = mujoco.MjData(model)
    q = {n:0.0 for n in visual_model.moving_joints}
    visual_model.set_feedback(model,data,q)
    left = data.body('astribot_arm_left_link_7').xpos.copy()
    right = data.body('astribot_arm_right_link_7').xpos.copy()
    q['astribot_arm_left_joint_2'] = 0.3
    visual_model.set_feedback(model,data,q)
    assert not np.allclose(left,data.body('astribot_arm_left_link_7').xpos)
    np.testing.assert_allclose(right,data.body('astribot_arm_right_link_7').xpos)
    assert set(visual_model.geom_groups.values()) == {'torso','head','left_arm','right_arm','left_gripper','right_gripper'}
    assert len([n for n in q if n.startswith('astribot_gripper_')]) == 12
    with pytest.raises(ValueError,match='No measured'):
        visual_model.set_feedback(model,data,{})


@pytest.mark.parametrize('k', [np.array([[250.,0,160],[0,260,120],[0,0,1]]),
                               np.array([[280.,0,112],[0,210,91],[0,0,1]])])
def test_projection_matches_opencv_with_offcenter_principal_point(k):
    # A tiny sphere at a known OpenCV point (x right, y down, z forward).
    point = np.array([0.17,-0.09,1.4])
    xml = f'''<mujoco><visual><global offwidth="320" offheight="240"/></visual><worldbody>
    <geom name="target" type="sphere" size="0.018" pos="{point[0]} {-point[1]} {-point[2]}"/>
    <camera name="test" ipd="0"/>
    </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    renderer = mujoco.Renderer(model,height=240,width=320)
    try:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data,camera='test')
        set_frustum(renderer.scene,k,320,240)
        seg = renderer.render()
        y,x = np.nonzero(seg[...,0] == model.geom('target').id)
        expected = (k @ point)[:2] / point[2]
        assert len(x)>5
        np.testing.assert_allclose([x.mean(),y.mean()],expected,atol=0.7)
    finally:
        renderer.close()


def test_distortion_remap_roundtrip():
    k = np.array([[280.,0,112],[0,210,91],[0,0,1]])
    distortion = [0.08,-0.02,0.001,-0.002,0]
    cfg = {'width':320,'height':240,'K':k.tolist(),
           'distortion_model':'opencv','distortion_coefficients':distortion}
    w,h,shifted,mapping = projection_plan(cfg)
    raw = np.array([[[210.,140.]]])
    undistorted = cv2.undistortPoints(raw,k,np.array(distortion),P=shifted)[0,0]
    np.testing.assert_allclose(mapping[140,210],undistorted,atol=1e-4)
    assert mapping[...,0].min()>=0 and mapping[...,0].max()<w
    assert mapping[...,1].min()>=0 and mapping[...,1].max()<h


def test_gripper_lookup_refuses_unknown_or_out_of_range(visual_model,dataset):
    cfg = template(dataset.info,visual_model)
    state = np.zeros(25)
    with pytest.raises(TypeError):
        feedback_positions(state,cfg)
    for side in ('left','right'):
        # Artificial LUT solely for exercising interpolation; not saved as S1 calibration.
        n = len(cfg['grippers'][side]['joint_names'])
        cfg['grippers'][side]['samples'] = [
            {'feedback':0.,'angles_rad':[0.]*n}, {'feedback':100.,'angles_rad':[1.]*n}]
    state[14] = 25
    positions = feedback_positions(state,cfg)
    assert positions['astribot_gripper_left_joint_L1'] == 0.25
    state[14] = -1
    with pytest.raises(ValueError,match='no clamping'):
        feedback_positions(state,cfg)


def test_dataset_uses_recorded_state_and_episode_video_offset(dataset):
    # Episode 1 shares the video shard, so row.timestamp alone is insufficient.
    row = dataset.row(1,0)
    assert 'action' not in row
    for camera in ('head','left','right'):
        feature = f'observation.images.{camera}'
        rgb,timing = dataset.rgb(row,feature)
        assert tuple(rgb.shape) == tuple(dataset.info['features'][feature]['shape'])
        assert timing['target_timestamp'] == dataset.episodes[1][f'videos/{feature}/from_timestamp']
        assert timing['delta_seconds'] < 0.51/dataset.info['fps']


def test_sdk_adapter_calls_only_current_feedback():
    class ReadOnlyFake:
        def get_current_joints_position(self,groups):
            return [[0.1]*len(GROUPS[group]) for group in groups]
    values,timing = read_current_feedback(ReadOnlyFake(),['astribot_arm_left_joint_1','astribot_gripper_right'])
    np.testing.assert_allclose(values,[0.1,0.1])
    assert timing['hardware_timestamp'] is None


def test_full_three_camera_pipeline_synthetic_calibration_only(visual_model,dataset):
    from cloak_mask.renderer import AstribotMaskRenderer, overlay
    info=copy.deepcopy(dataset.info)
    for name in ('head','left','right'):
        info['features'][f'observation.images.{name}']['shape']=[180,320,3]
    cfg=template(info,visual_model)
    cfg['state'].update(confirmed_measured_feedback=True,recording_source='SYNTHETIC TEST ONLY')
    for side in ('left','right'):
        n=len(cfg['grippers'][side]['joint_names'])
        cfg['grippers'][side].update(source='SYNTHETIC TEST ONLY',samples=[
            {'feedback':0.,'angles_rad':[0.]*n},{'feedback':100.,'angles_rad':[0.1]*n}])
    for name,camera in cfg['cameras'].items():
        camera.update(parent_link=visual_model.root,source='SYNTHETIC TEST ONLY',
            confirmed_recording_setup=True, rgb_time_offset_seconds=0.,
            K=[[220.,0,160.],[0,200.,90.],[0,0,1]],distortion_model='none',distortion_coefficients=[],
            T_parent_optical=[[1,0,0,0],[0,0,1,-2],[0,-1,0,0.5],[0,0,0,1]])
    cfg['cameras']['right'].update(distortion_model='opencv',distortion_coefficients=[0.05,-0.01,0.,0.,0.])
    assert not calibration_errors(cfg,info,visual_model)
    with AstribotMaskRenderer(visual_model,cfg,info) as renderer:
        masks=renderer.render(np.zeros(25))
        assert set(masks)=={'head','left','right'}
        for labels in masks.values():
            assert labels.shape==(180,320)
            assert labels.dtype==np.uint8
            assert labels.max()<=6 and (labels>0).sum()>20
            rgb=np.full((180,320,3),80,np.uint8)
            assert overlay(rgb,labels).shape==rgb.shape


def test_static_tf_direction_and_ros_import(visual_model,dataset):
    from cloak_mask.ros_import import nearest_static_mount,import_candidates
    transforms=[{'topic':'/tf_static','parent':'astribot_arm_left_link_7','child':'mount',
        'translation':[0.1,0,0],'quaternion_xyzw':[0,0,0,1]},
        {'topic':'/tf_static','parent':'mount','child':'left_optical',
        'translation':[0,0.2,0],'quaternion_xyzw':[0,0,0,1]}]
    link,t,path=nearest_static_mount('left_optical',transforms,set(visual_model.links))
    assert link=='astribot_arm_left_link_7'
    np.testing.assert_allclose(t[:3,3],[0.1,0.2,0])
    assert path==['left_optical','mount','astribot_arm_left_link_7']
    item={'topic':'/left_wrist/color/camera_info','frame_id':'left_optical','width':640,'height':360,
        'K':[420,0,310,0,425,175,0,0,1],'D':[0.,0.,0.,0.,0.],'distortion_model':'plumb_bob'}
    candidate,notes=import_candidates(template(dataset.info,visual_model),{'camera_info':[item],'tf':transforms},visual_model)
    assert candidate['cameras']['left']['K'][0][0]==420
    assert candidate['cameras']['left']['confirmed_recording_setup'] is False
    assert candidate['cameras']['head']['K'] is None
    assert calibration_errors(candidate,dataset.info,visual_model)


def test_simulation_affine_coupling_respects_reference_and_rejects_conflict():
    from cloak_mask.simulation import coupled_positions
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><joint name="a" ref="0.2"/><geom size=".01" type="sphere"/></body>
      <body><joint name="b" ref="0.1"/><geom size=".01" type="sphere"/></body>
      </worldbody><equality><joint joint1="a" joint2="b" polycoef="0.03 -2 0 0 0"/></equality></mujoco>''')
    # XML defaults to degrees; use compiled qpos0 in the assertion.
    values = coupled_positions(model,{'b':.4})
    assert values['a'] == pytest.approx(model.qpos0[0]+.03-2*(.4-model.qpos0[1]))
    assert coupled_positions(model,{'a':values['a']})['b'] == pytest.approx(.4)
    with pytest.raises(ValueError,match='Inconsistent'):
        coupled_positions(model,{'a':1.,'b':.4})


def test_official_simulation_cameras_and_finger_constraints():
    from cloak_mask.simulation import load_simulation, coupled_positions, camera_evidence
    root = ROOT/'astribot_simulation'
    if not (root/'astribot_descriptions/mjcf').exists():
        pytest.skip('Official description submodule not installed')
    model = load_simulation(root)
    data = mujoco.MjData(model)
    for side in ('left','right'):
        positions = coupled_positions(model,{f'astribot_gripper_{side}_joint_L1':.465})
        assert len(positions) == 6
        for name,value in positions.items():
            data.qpos[model.joint(name).qposadr[0]]=value
        for suffix in ('L1','L11','R1','R2'):
            assert positions[f'astribot_gripper_{side}_joint_{suffix}'] == pytest.approx(.465)
        for suffix in ('L2','R11'):
            assert positions[f'astribot_gripper_{side}_joint_{suffix}'] == pytest.approx(-.465)
    mujoco.mj_forward(model,data)
    sdk = AstribotVisualModel(ROOT/'astribot_sdk_ros2')
    for name in ('head_rgbd','left_wrist_rgbd','right_wrist_rgbd'):
        info = camera_evidence(model,data,name,sdk.links)
        parent = model.body(info['parent_link']).id
        transform = np.asarray(info['T_parent_optical'])
        np.testing.assert_allclose(data.xpos[parent]+data.xmat[parent].reshape(3,3)@transform[:3,3], data.cam_xpos[model.camera(name).id],atol=1e-12)
        assert info['fovy_degrees'] > 0


def test_head_camera_fk_includes_head_pitch_and_preview_roll():
    from cloak_mask.simulation import frame_camera_transforms
    # A nontrivial fixed camera mount on a head hinge; no vendor assets needed.
    model = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/>
      <worldbody><body name="head" pos="0.1 0.2 1.0">
      <joint name="pitch" axis="0 1 0"/><geom type="sphere" size=".02"/>
      <camera name="head_rgbd" pos=".06 -.21 -.03" euler=".1 .2 .3"/>
      </body></worldbody></mujoco>''')
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    zero=frame_camera_transforms(model,data,'head_rgbd')
    data.qpos[0]=.901317834854126
    mujoco.mj_forward(model,data)
    actual=frame_camera_transforms(model,data,'head_rgbd')
    t0=np.asarray(zero['T_world_optical_raw'])
    t1=np.asarray(actual['T_world_optical_raw'])
    angle=np.arccos(np.clip((np.trace(t0[:3,:3].T@t1[:3,:3])-1)/2,-1,1))
    assert angle==pytest.approx(data.qpos[0])
    assert not np.allclose(t0[:3,3],t1[:3,3])
    preview=np.asarray(actual['T_world_optical_preview'])
    np.testing.assert_allclose(preview,t1@np.diag([-1.,-1.,1.,1.]))
    np.testing.assert_allclose(np.asarray(actual['T_optical_preview_world'])@preview,np.eye(4),atol=1e-12)


@pytest.mark.parametrize('camera_y,cy_offset',[(0.,0.),(.06,0.),(0.,-20.)])
def test_native_projection_and_head_offset_sign(camera_y,cy_offset):
    # Independently project a known 3D marker through the actual default fovy path.
    h,w=240,320
    fovy=51.4175
    focal=h/(2*np.tan(np.deg2rad(fovy)/2))
    model=mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <geom name="marker" type="sphere" size=".008" pos=".1 .1 -1"/>
      <camera name="head" pos="0 {camera_y} 0" fovy="{fovy}"/>
      </worldbody></mujoco>''')
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    renderer=mujoco.Renderer(model,height=h,width=w)
    try:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data,camera='head')
        if cy_offset:
            k=np.array([[focal,0,(w-1)/2],[0,focal,(h-1)/2-cy_offset],[0,0,1]])
            set_frustum(renderer.scene,k,w,h)
        seg=cv2.flip(renderer.render(),-1)
        y,x=np.nonzero(seg[...,0]==model.geom('marker').id)
        expected=[(w-1)/2-focal*.1,(h-1)/2+focal*(.1-camera_y)+cy_offset]
        np.testing.assert_allclose([x.mean(),y.mean()],expected,atol=.7)
    finally:
        renderer.close()


def test_tcp_projection_matches_render_with_resize_flip_and_intrinsics():
    from cloak_mask.tcp import project_tcp
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="marker" type="sphere" size=".012" pos=".1 .08 -1"/>
      <site name="arm_left_tool" pos=".1 .08 -1"/>
      <site name="arm_right_tool" pos="0 0 1"/>
      <camera name="test" pos=".02 0 0" euler="0 3 0"/>
      </worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    renderer=mujoco.Renderer(model,height=240,width=320)
    k=np.array([[210.,0,130.],[0,195.,100.],[0,0,1.]])
    option=mujoco.MjvOption();option.sitegroup[:]=0
    try:
        renderer.enable_segmentation_rendering();renderer.update_scene(data,camera='test',scene_option=option)
        set_frustum(renderer.scene,k,320,240)
        seg=renderer.render();mask=(seg[...,0]==model.geom('marker').id).astype(np.uint8)
        mask=cv2.flip(cv2.resize(mask,(640,360),interpolation=cv2.INTER_NEAREST),-1)
        y,x=np.nonzero(mask)
        result=project_tcp(model,data,'test',k,(320,240),(640,360))
        assert result['left']['in_frame']
        np.testing.assert_allclose(result['left']['pixel_xy'],[x.mean(),y.mean()],atol=1.)
        assert result['right']['pixel_xy'] is None
        assert not result['right']['in_frame']
    finally:renderer.close()
