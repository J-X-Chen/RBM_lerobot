"""ROS2-only read-only calibration capture; no NumPy, MuJoCo, SDK or policy dependency."""
import importlib.util
import os
import shutil
import time


def probe_ros(timeout=5.0):
    """Read CameraInfo and TF if ROS2 is actually available; no camera/motion services."""
    report = {'ros2_executable':shutil.which('ros2'), 'status':'UNAVAILABLE',
              'camera_info':[], 'tf':[], 'environment':{
                  k:os.environ.get(k) for k in ('ROS_DISTRO','ROS_DOMAIN_ID','RMW_IMPLEMENTATION')}}
    missing = [n for n in ('rclpy','sensor_msgs','tf2_msgs') if importlib.util.find_spec(n) is None]
    if missing:
        report['reason'] = f'ROS2 Python modules not installed in this interpreter: {missing}'
        return report
    import rclpy
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo
    from tf2_msgs.msg import TFMessage
    rclpy.init()
    node = rclpy.create_node('cloak_mask_calibration_readonly')
    subscriptions = []
    seen = set()
    camera_samples, transforms = {}, {}
    try:
        deadline = time.monotonic()+timeout
        while time.monotonic()<deadline:
            for topic, types in node.get_topic_names_and_types():
                if topic in seen:
                    continue
                if 'sensor_msgs/msg/CameraInfo' in types:
                    def callback(msg, topic=topic):
                        camera_samples[topic] = {'topic':topic, 'frame_id':msg.header.frame_id,
                            'width':msg.width,'height':msg.height,'K':list(msg.k),'D':list(msg.d),
                            'R':list(msg.r),'P':list(msg.p),'distortion_model':msg.distortion_model,
                            'stamp':{'sec':msg.header.stamp.sec,'nanosec':msg.header.stamp.nanosec},
                            'binning_x':msg.binning_x,'binning_y':msg.binning_y,
                            'roi':{k:getattr(msg.roi,k) for k in ('x_offset','y_offset','width','height','do_rectify')}}
                    subscriptions.append(node.create_subscription(CameraInfo,topic,callback,
                        QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)))
                    seen.add(topic)
                elif 'tf2_msgs/msg/TFMessage' in types and topic in ('/tf','/tf_static'):
                    def callback(msg, topic=topic):
                        for t in msg.transforms:
                            transforms[(t.header.frame_id,t.child_frame_id)] = {
                                'topic':topic,'parent':t.header.frame_id,'child':t.child_frame_id,
                                'translation':[t.transform.translation.x,t.transform.translation.y,t.transform.translation.z],
                                'quaternion_xyzw':[t.transform.rotation.x,t.transform.rotation.y,t.transform.rotation.z,t.transform.rotation.w],
                                'stamp':{'sec':t.header.stamp.sec,'nanosec':t.header.stamp.nanosec}}
                    qos = QoSProfile(depth=100,reliability=ReliabilityPolicy.RELIABLE,
                        durability=DurabilityPolicy.TRANSIENT_LOCAL if topic=='/tf_static' else DurabilityPolicy.VOLATILE)
                    subscriptions.append(node.create_subscription(TFMessage,topic,callback,qos))
                    seen.add(topic)
            rclpy.spin_once(node,timeout_sec=0.1)
        report.update(status='CAPTURED' if camera_samples else 'NO_CAMERA_INFO_RECEIVED',
            camera_info=list(camera_samples.values()),tf=list(transforms.values()),
            discovered_topics=sorted(seen))
        report['note']='Live values are evidence, not proof that the historical recording used the same camera/resize/crop.'
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return report

