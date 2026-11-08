"""H1 remote interface factory; native SDK belongs exclusively on the H1 service host."""
from zerith_image_service.cameras import CameraSubscriber
from zerith_lerobot.eval_robot.robot_control.robot_arm_zerith import ZerithH1Controller


def make_robot(host, control_port=6200, camera_ports=(55555, 55556, 55557)):
    controller = ZerithH1Controller(host, control_port)
    try:
        cameras = CameraSubscriber(host, camera_ports)
    except Exception:
        controller.close()
        raise
    return {"arm_ctrl": controller, "cameras": cameras, "state_dim": 23, "action_dim": 19}
