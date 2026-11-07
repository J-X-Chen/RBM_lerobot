"""Synthetic calibration tests do not establish real-camera accuracy."""
from pathlib import Path
import copy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from cloak_mask.aloha import (AlohaVisualModel, AlohaRenderer, DEFAULT_URDF,
                              positions, template, validate, collada_to_obj)
from cloak_mask.dataset import DatasetFrames
from cloak_mask.renderer import mujoco

ROOT = Path(__file__).resolve().parents[2]

@pytest.fixture(scope='module')
def visual(tmp_path_factory):
    return AlohaVisualModel(ROOT / DEFAULT_URDF, tmp_path_factory.mktemp('aloha_mesh'))

@pytest.fixture(scope='module')
def dataset():
    return DatasetFrames(ROOT/'lerobotv3_dataset/aloha_cup_selection_red')


def synthetic_config(dataset, visual):
    cfg = template(dataset,visual)
    cfg['status']='CANDIDATE'
    cfg['grippers']={s:dict(scale=1.,offset_m=0.,source='SYNTHETIC TEST ONLY') for s in ('left','right')}
    for c in cfg['cameras'].values():
        t=np.eye(4)
        t[:3,:3]=np.diag([1.,-1.,-1.])
        t[:3,3]=[.2,0,2.]
        c.update(parent_link='body_Link',K=[[500,0,319.5],[0,500,239.5],[0,0,1]],
                 T_parent_optical=t.tolist(),distortion_model='none',distortion_coefficients=[],source='SYNTHETIC TEST ONLY')
    return cfg


def test_missing_parameters_and_invalid_pose_rejected(visual,dataset):
    cfg=template(dataset,visual)
    assert len(validate(cfg,dataset,visual,True))==6
    cfg=synthetic_config(dataset,visual)
    assert not validate(cfg,dataset,visual,True)
    assert validate(cfg,dataset,visual,False)
    cfg['cameras']['head']['T_parent_optical'][0][0]=2
    assert any('rigid' in e for e in validate(cfg,dataset,visual,True))


def test_dual_arm_fk_and_gripper_no_clipping(visual):
    model=mujoco.MjModel.from_xml_string(visual.mjcf(640,480))
    data=mujoco.MjData(model)
    g={s:dict(scale=1.,offset_m=0.) for s in ('left','right')}
    q=np.zeros(14); q[[6,13]]=.1
    mapped=positions(q,g)
    assert mapped['fl_joint7']==.05 and mapped['fl_joint8']==-.05
    visual.set_feedback(model,data,mapped)
    before={n:data.body(n).xpos.copy() for n in ('fl_link6','fr_link6','fl_link7','fl_link8')}
    # Independent matrix composition verifies the URDF chain against MuJoCo.
    chain=np.eye(4)
    for name in ['fl_base_joint']+[f'fl_joint{i}' for i in range(1,7)]:
        chain=chain@visual.joints[name].relative(mapped.get(name,0.))
    np.testing.assert_allclose(chain[:3,3],before['fl_link6'],atol=1e-10)
    q[1]=.4; q[6]=.02
    visual.set_feedback(model,data,positions(q,g))
    assert not np.allclose(before['fl_link6'],data.body('fl_link6').xpos)
    np.testing.assert_allclose(before['fr_link6'],data.body('fr_link6').xpos)
    assert all(not n.startswith(('br_','bl_')) for n in visual.links)


def test_renderer_produces_dual_arm_labels(visual,dataset):
    cfg=synthetic_config(dataset,visual)
    renderer=AlohaRenderer(visual,cfg)
    try:
        q=np.zeros(14); q[[6,13]]=.05
        masks,poses=renderer.render(q)
        assert set(masks)=={'head','left','right'}
        for m in masks.values():
            assert m.shape==(480,640)
            assert {3,4}.issubset(set(np.unique(m)))
        np.testing.assert_allclose(masks['head'],masks['left'])
        subset, subset_poses = renderer.render(q, camera_names=['left'])
        assert set(subset) == set(subset_poses) == {'left'}
        np.testing.assert_array_equal(subset['left'], masks['left'])
        with pytest.raises(ValueError, match='Unknown camera'):
            renderer.render(q, camera_names=['unknown'])
        q[1]=.5
        changed,_=renderer.render(q)
        assert np.any(changed['head']!=masks['head'])
    finally:
        renderer.close()


def test_nominal_preview_keeps_capture_and_unverified_status(visual, dataset, tmp_path):
    import importlib.util
    from types import SimpleNamespace
    import json
    spec=importlib.util.spec_from_file_location('prepare_preview',ROOT/'mask_utils/scripts/prepare_aloha_preview.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    output=tmp_path/'candidate.json'
    capture=ROOT/'aloha_calibration_reference_20261031/capture.json'
    module.prepare(SimpleNamespace(dataset_root=dataset.root,urdf=visual.urdf,
        mesh_cache=tmp_path/'meshes',capture=capture,head_height=0.,head_pitch=0.,output=output))
    cfg=json.loads(output.read_text())
    source=json.loads(capture.read_text())
    assert not validate(cfg,dataset,visual,True)
    assert validate(cfg,dataset,visual,False)
    for name,c in cfg['cameras'].items():
        np.testing.assert_array_equal(np.asarray(c['K']).ravel(),source['cameras'][name]['camera_info']['K'])
        assert c['distortion_coefficients']==source['cameras'][name]['camera_info']['D']
    assert cfg['preview_assumptions']['head_stand_height_m']==0
    assert not cfg['grippers']['left']['verified']
    with pytest.raises(FileExistsError):
        module.prepare(SimpleNamespace(dataset_root=dataset.root,urdf=visual.urdf,
            mesh_cache=tmp_path/'meshes',capture=capture,head_height=0.,head_pitch=0.,output=output))


def test_camera_housing_exclusion_cannot_hide_fingers(visual,dataset):
    cfg=synthetic_config(dataset,visual)
    cfg['cameras']['left']['exclude_camera_housing']=['fl_link7']
    renderer=AlohaRenderer(visual,cfg)
    try:
        with pytest.raises(ValueError,match='Only the viewing camera housing'):
            renderer.render(np.zeros(14))
        cfg['cameras']['left']['exclude_camera_housing']=['fl_camera_link']
        groups=renderer.model.geom_group.copy()
        masks,_=renderer.render(np.zeros(14))
        np.testing.assert_array_equal(renderer.model.geom_group,groups)
        assert np.any(masks['head'])
    finally:
        renderer.close()


def test_front_plane_trial_preserves_mount_orientation_and_geometry(visual,dataset,tmp_path):
    import importlib.util,json
    from types import SimpleNamespace
    spec=importlib.util.spec_from_file_location('prepare_preview_front',ROOT/'mask_utils/scripts/prepare_aloha_preview.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    results=[]
    for mode in ('mesh-origin','mesh-front'):
        output=tmp_path/f'{mode}.json'
        module.prepare(SimpleNamespace(dataset_root=dataset.root,urdf=visual.urdf,
            mesh_cache=tmp_path/'meshes',capture=ROOT/'aloha_calibration_reference_20261031/capture.json',
            head_height=0.,head_pitch=0.,wrist_optical_origin=mode,output=output))
        results.append(json.loads(output.read_text()))
    old,new=results
    assert old['cameras']['head']==new['cameras']['head']
    for name in ('left','right'):
        a=np.array(old['cameras'][name]['T_parent_optical']);b=np.array(new['cameras'][name]['T_parent_optical'])
        np.testing.assert_array_equal(a[:3,:3],b[:3,:3])
        assert old['cameras'][name]['K']==new['cameras'][name]['K']
        delta=a[:3,:3].T@(b[:3,3]-a[:3,3])
        np.testing.assert_allclose(delta,[0,0,new['cameras'][name]['optical_origin_trial']['front_surface_x_m']],atol=1e-10)
        assert new['cameras'][name]['exclude_camera_housing']==old['cameras'][name]['exclude_camera_housing']
    masks=[];models=[]
    for cfg in results:
        renderer=AlohaRenderer(visual,cfg)
        try:
            masks.append(renderer.render(dataset.row(0,180)['observation.state'])[0])
            models.append((renderer.model.ngeom,renderer.model.mesh_vert.copy()))
        finally:renderer.close()
    assert models[0][0]==models[1][0]
    np.testing.assert_array_equal(models[0][1],models[1][1])
    np.testing.assert_array_equal(masks[0]['head']>0,masks[1]['head']>0)
    for name,label in [('left',3),('right',4)]:
        assert (masks[0][name][400:480,240:400]==label).sum()>1000
        assert (masks[1][name][400:480,240:400]==label).sum()==0
