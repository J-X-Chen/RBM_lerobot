#!/usr/bin/env python3
"""Read ROS1 CameraInfo and TF only; run in the camera service's ROS Noetic environment."""
import argparse
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=15)
    p.add_argument('--head-parent', default='body_Link')
    p.add_argument('--left-parent', default='fl_link6')
    p.add_argument('--right-parent', default='fr_link6')
    a = p.parse_args()
    if a.output.exists() or not 0 < a.seconds <= 60:
        p.error('Choose a new output path and seconds in (0,60]')
    import rospy
    import tf2_ros
    from sensor_msgs.msg import CameraInfo
    rospy.init_node('aloha_mask_calibration_capture', anonymous=True, disable_signals=True)
    buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buffer)
    records = {}
    def callback(msg, name):
        records[name] = dict(width=msg.width, height=msg.height, K=list(msg.K),
                             D=list(msg.D), R=list(msg.R), P=list(msg.P),
                             distortion_model=msg.distortion_model, frame_id=msg.header.frame_id,
                             stamp=msg.header.stamp.to_sec())
    subscribers = [rospy.Subscriber(f'/camera_{suffix}/color/camera_info', CameraInfo, callback, name)
                   for name, suffix in [('head','f'),('left','l'),('right','r')]]
    deadline = time.monotonic() + a.seconds
    while time.monotonic() < deadline and not rospy.is_shutdown():
        time.sleep(.05)
    result = dict(status='CAPTURE_ONLY_NOT_VERIFIED', cameras={},
                  note='Driver CameraInfo can be a fallback. Verify device calibration provenance and recording configuration.')
    for name in ('head', 'left', 'right'):
        info = records.get(name)
        entry = dict(camera_info=info, parent_link=getattr(a, name+'_parent'))
        if info is not None:
            # Camera-internal factory TF is useful even when the robot TF is absent.
            camera_base = 'camera_' + {'head':'f', 'left':'l', 'right':'r'}[name] + '_link'
            try:
                t = buffer.lookup_transform(camera_base, info['frame_id'],
                                            rospy.Time.from_sec(info['stamp']), rospy.Duration(.2))
                q, v = t.transform.rotation, t.transform.translation
                entry['camera_internal_tf'] = dict(parent_link=camera_base, optical_frame=info['frame_id'],
                    translation_xyz_m=[v.x,v.y,v.z], quaternion_xyzw=[q.x,q.y,q.z,q.w],
                    note='Camera driver base to RGB optical; NOT robot mounting extrinsics. URDF housing origin equivalence must be checked.')
            except Exception as e:
                entry['camera_internal_tf_error'] = str(e)
            try:
                # Use the CameraInfo timestamp; do not silently substitute latest dynamic TF.
                t = buffer.lookup_transform(entry['parent_link'], info['frame_id'],
                                            rospy.Time.from_sec(info['stamp']), rospy.Duration(.2))
                q, v = t.transform.rotation, t.transform.translation
                entry['T_parent_optical_translation_xyz'] = [v.x,v.y,v.z]
                entry['T_parent_optical_quaternion_xyzw'] = [q.x,q.y,q.z,q.w]
                entry['tf_stamp'] = t.header.stamp.to_sec()
            except Exception as e:
                entry['tf_error'] = str(e)
        else:
            entry['error'] = 'No CameraInfo received'
        result['cameras'][name] = entry
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
    print(a.output)

if __name__ == '__main__':
    main()
