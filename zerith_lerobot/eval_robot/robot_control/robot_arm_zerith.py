"""Remote H1 adapter. The workstation never imports native robot SDK bindings."""
from zerith_image_service.contract import ControlClient, vector


class ZerithH1Controller:
    def __init__(self, host, port=6200, timeout=2.):
        self.client = ControlClient(host, port, timeout)
        self.armed = False

    def state(self):
        reply = self.client.request(get_state=True)
        control = reply['control']
        if control['fault'] or control['state_age_s'] > .5:
            raise RuntimeError(f'H1 feedback unavailable: {control}')
        return vector(reply['state'], 23), control

    def arm(self):
        _, status = self.state()
        if not status['execute']:
            raise RuntimeError('H1 service is read-only; --execute-control is required on H1')
        self.client.request(arm=True)
        self.armed = True

    def heartbeat(self):
        return self.client.request(heartbeat=True)

    def action(self, action):
        return self.client.request(action=vector(action, 19).tolist())

    def hold(self):
        return self.client.request(hold_current=True)

    def close(self):
        try:
            if self.armed:
                self.client.request(disarm=True)
        finally:
            self.client.close()
