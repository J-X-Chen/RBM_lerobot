import socket, threading, json
import numpy as np
from server import receive, validate, Bridge
assert validate([0]*14).shape == (14,)
a=[0]*14; a[6]=.1
assert validate(a)[6]==.08
for a in ([0]*13,[float('nan')]*14,[4]*14):
    try: validate(a)
    except ValueError: pass
    else: raise AssertionError(a)
a,b=socket.socketpair()
t=threading.Thread(target=lambda: [a.sendall(x) for x in [b'000',b'000',b'0002',b'{}']]); t.start()
assert int(receive(b,10))==2
assert json.loads(receive(b,2))=={}
t.join(); a.close(); b.close()
print('validation and fragmented TCP framing passed')
