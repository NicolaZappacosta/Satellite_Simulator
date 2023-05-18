### Guidance Functions

import numpy as np

# Control Quaternion feedback w/o angular acceleration

def contQuatfeedback(self, kp=[0., 0., 0.], kd=[0., 0., 0.], a = 0. ):
    if len(kp) == 1:
        kp = [kp, kp, kp]
    if len(kd) == 1:
        kd = [kd, kd, kd]

    kp = np.array(kp)
    kd = np.array(kd)

    def fun_control(x, x_ref):

        q = x[0:4].reshape(4)
        q_ref = x_ref[0:4]

        w = x[4:7].reshape(3)

        w_ref = x_ref[4:7]
        dq = Math.errorquat(q, q_ref)[0:3]
        T = -np.multiply(kp, dq) - np.multiply(kd, w - w_ref)

        return T

    self.fun_control = fun_control