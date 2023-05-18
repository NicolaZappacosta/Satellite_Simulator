### Guidance Functions

import numpy as np

# Radial North-East-Down

def radialNED(self):  # TBD: add w_dot_c

    alpha_0 = self.alpha_0
    omega = self.w_e

    def fun_guidance(t, x):
        q = x[0:4].reshape(4)
        X = x[7:10].reshape(3)
        V = x[10:13].reshape(3)

        Xecef = Math.ECI2ECEF(X, alpha_0, omega, t)

        lat, lon, R = Math.GC(Xecef)

        Aeci2ned = Math.RotMat_ECI2NED(alpha_0, omega, lon, lat, t)

        q_c = Math.mat2quat(Aeci2ned)

        w_c_eci = np.cross(X, V) / np.linalg.norm(X) ** 2

        A = Math.quat2mat(q)

        w_c = np.matmul(A, w_c_eci)

        return np.array([q_c[0], q_c[1], q_c[2], q_c[3], w_c[0], w_c[1], w_c[2]])

    return fun_guidance