"""Read-only H1 camera/state viewer."""
import argparse
import time
import cv2
from .cameras import CameraSubscriber
from .contract import ControlClient, NAMES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=6200)
    p.add_argument('--camera-ports', nargs=3, type=int, default=[55555, 55556, 55557])
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--frames', type=int, default=0, help='0 runs until q/Ctrl-C')
    args = p.parse_args()
    client, cameras = ControlClient(args.host, args.port), None
    try:
        cameras = CameraSubscriber(args.host, args.camera_ports)
        count = 0
        while True:
            reply = client.request(get_state=True)
            frames = cameras.read(server_time=reply["control"]["server_time"])
            if reply['control']['fault']:
                raise RuntimeError(reply['control']['fault'])
            if count % 30 == 0:
                print(dict(zip(NAMES, reply['state'])), reply['control'], flush=True)
            count += 1
            if not args.no_display:
                for name, frame in frames.items():
                    cv2.imshow(name, frame)
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
            if args.frames and count >= args.frames:
                break
            time.sleep(1/30)
    finally:
        client.close()
        if cameras:
            cameras.close()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
