import numpy as np
import pandas as pd

import Math
import EarthModel
import Guidance

from Math import quatmolt
from scipy import integrate as inte
from matplotlib import pyplot as plt

class Satellite(object):

    def __init__(self, mass = 1000, inertia = [100.,0.,0.,0.,200.,0.,0.,0.,300.], q = [0.,0.,0.,1.], w=[0.,0.,0.], position = [7000,0,0], velocity = [0,7,0],t = 0.):

        self.alpha_0 = float(0)
        self.w_e = float(2*np.pi/86164)
        self.mass = mass
        self.inertia = np.array(inertia).reshape(3,3)
        self.q = np.array(q).reshape(4)
        self.w = np.array(w).reshape(3)
        self.position = np.array(position).reshape(3)
        self.velocity = np.array(velocity).reshape(3)
        self.t = t
        self.fun_control    = lambda x,x_ref: np.array([0.,0.,0.])
        self.fun_reference  = lambda t,x: np.array([0.,0.,0.,1.,0.,0.,0.])
        self.fun_gravity    = lambda x: np.array([0.,0.,0.])
        self.fun_actuation  = lambda T: T
        self.actuation_flag = 'none'

        history = pd.DataFrame()
        history['t']  = [t]
        history['q1'] = [q[0]]
        history['q2'] = [q[1]]
        history['q3'] = [q[2]]
        history['q0'] = [q[3]]
        history['w1'] = [w[0]]
        history['w2'] = [w[1]]
        history['w3'] = [w[2]]
        history['x1'] = [position[0]]
        history['x2'] = [position[1]]
        history['x3'] = [position[2]]
        history['v1'] = [velocity[0]]
        history['v2'] = [velocity[1]]
        history['v3'] = [velocity[2]]

        self.history = history

## GUIDANCE PROBLEM

# RADIAL NED

    def guidance(self, type = 'radialNED'):

        if type == 'radialNED':

            self.fun_guidance = Guidance.radialNED(self)

## CONTROL PROBLEM


    # TBD add contQuatfeedback with angular rate

    def contQuatfeedback(self, kp = [0., 0., 0.], kd = [0., 0., 0.]):

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
            T = -np.multiply(kp, dq)-np.multiply(kd, w-w_ref)

            return T

        self.fun_control =  fun_control

# ACTUATION PROBLEM

    # TDB Magnetotorques

    def actuationRW(self, G = [np.sqrt(3)/3,-np.sqrt(3)/3,-np.sqrt(3)/3,np.sqrt(3)/3, np.sqrt(3)/3,-np.sqrt(3)/3, np.sqrt(3)/3,-np.sqrt(3)/3, np.sqrt(3)/3,np.sqrt(3)/3 ,-np.sqrt(3)/3,-np.sqrt(3)/3],w_RW = [0.,0.,0.],flag_RW = [1.,1.,1.,1.]):

        G = np.array(G).reshape(3,4)

        if not hasattr(self, 'w_RW'):

            self.w_RW = w_RW

        def fun_actuation(T):

            G_plus = np.linalg.pinv(G)
            tau_d = np.dot(G_plus,T)
            # INSERT ACTUATION EQUATIONS
            tau_a = np.multiply(flag_RW,tau_d)
            Ta = np.matmul(G,tau_a)
            Y = np.array([Ta[0],Ta[1],Ta[2],tau_a[0],tau_a[1],tau_a[2]])

            return Y

        self.fun_actuation = fun_actuation
        self.actuation_flag = 'RWs'

# PROPAGATION
    def multistep(self,step_time = 1., n = 2) -> None:

        for i in range(n):
            self.step(step_time = step_time)

    def step(self,step_time = 1.):

        x_0 = np.append(np.append(self.q, self.w), np.append(self.position, self.velocity))
        t_0 = self.t

        def f(t,x):

            invI = np.linalg.inv(self.inertia)
            w = x[4:7]
            w_aug = np.append(w, 0.)
            q = x[0:4]/np.linalg.norm(x[0:4])                 # Stability normalization
            velocity = np.array(x[10:13]).reshape(3)
            x_ref = self.fun_reference(t_0, x_0)
            T = self.fun_control(x_0, x_ref)
            Y = self.fun_actuation(T)
            Ta = Y[0:3]

            g = self.fun_gravity(x)

            dq1, dq2, dq3, dq0 = 1/2*quatmolt(q, w_aug)
            dw1, dw2, dw3 = -np.matmul(invI,np.cross(w.reshape(3),np.matmul(self.inertia,w).reshape(3))-Ta)
            dx1, dx2, dx3 = velocity[0], velocity[1], velocity[2]
            dv1, dv2, dv3 = g[0], g[1], g[2]

            return np.array([dq1, dq2, dq3, dq0, dw1, dw2, dw3, dx1, dx2, dx3, dv1, dv2, dv3])

        solution = inte.RK45(f, self.t, x_0, t_bound = step_time+self.t, vectorized = True, rtol = 1e-9, atol = 1e-9)

        t_values, q1_values, q2_values, q3_values, q0_values = [], [], [], [], []
        w1_values, w2_values, w3_values = [], [], []
        x1_values, x2_values, x3_values = [], [], []
        v1_values, v2_values, v3_values = [], [], []

        while solution.status != 'finished':
            # get solution step state
            solution.step()
            t_values.append(solution.t)
            q1_values.append(solution.y[0])
            q2_values.append(solution.y[1])
            q3_values.append(solution.y[2])
            q0_values.append(solution.y[3])
            w1_values.append(solution.y[4])
            w2_values.append(solution.y[5])
            w3_values.append(solution.y[6])
            x1_values.append(solution.y[7])
            x2_values.append(solution.y[8])
            x3_values.append(solution.y[9])
            v1_values.append(solution.y[10])
            v2_values.append(solution.y[11])
            v3_values.append(solution.y[12])

        del t_values[0]
        del q1_values[0]
        del q2_values[0]
        del q3_values[0]
        del q0_values[0]
        del w1_values[0]
        del w2_values[0]
        del w3_values[0]
        del x1_values[0]
        del x2_values[0]
        del x3_values[0]
        del v1_values[0]
        del v2_values[0]
        del v3_values[0]

        temp = pd.DataFrame()
        temp['t']  = t_values
        temp['q1'] = q1_values
        temp['q2'] = q2_values
        temp['q3'] = q3_values
        temp['q0'] = q0_values
        temp['w1'] = w1_values
        temp['w2'] = w2_values
        temp['w3'] = w3_values
        temp['x1'] = x1_values
        temp['x2'] = x2_values
        temp['x3'] = x3_values
        temp['v1'] = v1_values
        temp['v2'] = v2_values
        temp['v3'] = v3_values

        self.q = np.array([q1_values[-1],q2_values[-1],q3_values[-1],q0_values[-1]]).reshape(4)
        self.w = np.array([w1_values[-1],w2_values[-1],w3_values[-1]]).reshape(3)
        self.t = t_values[-1]
        self.position = np.array([x1_values[-1], x2_values[-1], x3_values[-1]]).reshape(3)
        self.velocity = np.array([v1_values[-1], v2_values[-1], v3_values[-1]]).reshape(3)

        self.history = pd.concat([self.history,temp])

        return solution

    def propagation(self,time):

        x_0 = np.append(np.append(self.q,self.w),np.append(self.position,self.velocity))

        def f(t,x):

            invI = np.linalg.inv(self.inertia)
            w = x[4:7]
            w_aug = np.append(w,0.)
            q = x[0:4]/np.linalg.norm(x[0:4])                 # Stability normalization
            velocity = np.array(x[10:13]).reshape(3)
            x_ref = self.fun_reference(t, x)
            T = self.fun_control(x,x_ref)
            Y = self.fun_actuation(T)
            Ta = Y[0:3]

            g = self.fun_gravity(x)

            dq1, dq2, dq3, dq0 = 1/2*quatmolt(q, w_aug)
            dw1, dw2, dw3 = -np.matmul(invI,np.cross(w.reshape(3),np.matmul(self.inertia,w).reshape(3))-Ta)
            dx1, dx2, dx3 = velocity[0], velocity[1], velocity[2]
            dv1, dv2, dv3 = g[0], g[1], g[2]

            return np.array([dq1, dq2, dq3, dq0, dw1, dw2, dw3, dx1, dx2, dx3, dv1, dv2, dv3])

        solution = inte.RK45(f, self.t, x_0, t_bound = time+self.t, vectorized = True, rtol = 1e-9, atol = 1e-9)

        t_values, q1_values, q2_values, q3_values, q0_values = [], [], [], [], []
        w1_values, w2_values, w3_values = [], [], []
        x1_values, x2_values, x3_values = [], [], []
        v1_values, v2_values, v3_values = [], [], []

        while solution.status != 'finished':
            # get solution step state
            solution.step()
            t_values.append(solution.t)
            q1_values.append(solution.y[0])
            q2_values.append(solution.y[1])
            q3_values.append(solution.y[2])
            q0_values.append(solution.y[3])
            w1_values.append(solution.y[4])
            w2_values.append(solution.y[5])
            w3_values.append(solution.y[6])
            x1_values.append(solution.y[7])
            x2_values.append(solution.y[8])
            x3_values.append(solution.y[9])
            v1_values.append(solution.y[10])
            v2_values.append(solution.y[11])
            v3_values.append(solution.y[12])

        del t_values[0]
        del q1_values[0]
        del q2_values[0]
        del q3_values[0]
        del q0_values[0]
        del w1_values[0]
        del w2_values[0]
        del w3_values[0]
        del x1_values[0]
        del x2_values[0]
        del x3_values[0]
        del v1_values[0]
        del v2_values[0]
        del v3_values[0]

        temp = pd.DataFrame()
        temp['t']  = t_values
        temp['q1'] = q1_values
        temp['q2'] = q2_values
        temp['q3'] = q3_values
        temp['q0'] = q0_values
        temp['w1'] = w1_values
        temp['w2'] = w2_values
        temp['w3'] = w3_values
        temp['x1'] = x1_values
        temp['x2'] = x2_values
        temp['x3'] = x3_values
        temp['v1'] = v1_values
        temp['v2'] = v2_values
        temp['v3'] = v3_values

        self.q = np.array([q1_values[-1],q2_values[-1],q3_values[-1],q0_values[-1]]).reshape(4)
        self.w = np.array([w1_values[-1],w2_values[-1],w3_values[-1]]).reshape(3)
        self.t = t_values[-1]
        self.position = np.array([x1_values[-1], x2_values[-1], x3_values[-1]]).reshape(3)
        self.velocity = np.array([v1_values[-1], v2_values[-1], v3_values[-1]]).reshape(3)

        self.history = pd.concat([self.history,temp])

        return solution

    def gravity(self, type = 'spherical'):

        mu = 398600.4418

        if type == 'spherical':

            self.fun_gravity = EarthModel.sphericGravity()

        elif type == 'harmonic':

            self.fun_gravity = EarthModel.harmonicGravity()

# From: Department of Defense World Geodetic System 1984, Its Definition and Relationship with Local Geodetic Systems
        #if type = 'WGS84'


# PLOTTING

    def plotquaternions(self):

        history = self.history

        t_values = history['t']
        q1_values = history['q1']
        q2_values = history['q2']
        q3_values = history['q3']
        q0_values = history['q0']

        plt.figure()
        plt.plot(t_values,q1_values)
        plt.plot(t_values,q2_values)
        plt.plot(t_values,q3_values)
        plt.plot(t_values,q0_values)
        plt.legend(['q1', 'q2', 'q3', 'q0'])

    def plotangularrate(self):

        history = self.history

        t_values = history['t']
        w1_values = history['w1']
        w2_values = history['w2']
        w3_values = history['w3']

        plt.figure()
        plt.plot(t_values,w1_values)
        plt.plot(t_values,w2_values)
        plt.plot(t_values,w3_values)
        plt.legend(['w1', 'w2', 'w3'])

    def plotposition(self):

        history = self.history

        t_values = history['t']
        x1_values = history['x1']
        x2_values = history['x2']
        x3_values = history['x3']

        plt.figure()
        plt.plot(t_values, x1_values)
        plt.plot(t_values, x2_values)
        plt.plot(t_values, x3_values)
        plt.legend(['x1', 'x2', 'x3'])

    def plotvelocity(self):

        history = self.history

        t_values = history['t']
        v1_values = history['v1']
        v2_values = history['v2']
        v3_values = history['v3']

        plt.figure()
        plt.plot(t_values, v1_values)
        plt.plot(t_values, v2_values)
        plt.plot(t_values, v3_values)
        plt.legend(['v1', 'v2', 'v3'])

    def plotSSP(self):

        history = self.history

        alpha_0 = self.alpha_0
        omega = self.w_e

        t_values = history['t'].values
        x1_values = history['x1'].values
        x2_values = history['x2'].values
        x3_values = history['x3'].values

        lat_values = []
        lon_values = []
        R_values = []

        for i in range(len(x1_values)):

            Xeci = np.array([x1_values[i], x2_values[i], x3_values[i]])
            Xecef = Math.ECI2ECEF(Xeci,alpha_0,omega,t_values[i])
            x, y, z = Math.GC(Xecef)
            lat_values.append(x*180/np.pi)
            lon_values.append(y*180/np.pi)
            R_values.append(z)

        plt.figure()
        plt.plot(lon_values, lat_values,'o')
        plt.legend(['SSP'])

        plt.figure()
        plt.plot(t_values, R_values)
        plt.legend(['Radius'])

    def plotAttitude(self):

        history = self.history

        alpha_0 = self.alpha_0
        omega = self.w_e

        q1_values = history['q1'].values
        q2_values = history['q2'].values
        q3_values = history['q3'].values
        q0_values = history['q0'].values

        X1_1 = []
        X1_2 = []
        X1_3 = []
        X2_1 = []
        X2_2 = []
        X2_3 = []
        X3_1 = []
        X3_2 = []
        X3_3 = []


        for i in range(len(q1_values)):

            q = np.array([q1_values[i],q2_values[i],q3_values[i],q0_values[i]])

            A = Math.quat2mat(q)

            X1_1.append(A[0, 0])
            X1_2.append(A[0, 1])
            X1_3.append(A[0, 2])
            X2_1.append(A[1, 0])
            X2_2.append(A[1, 1])
            X2_3.append(A[1, 2])
            X3_1.append(A[2, 0])
            X3_2.append(A[2, 1])
            X3_3.append(A[2, 2])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(len(q1_values)):

            ax.quiver(0, 0, 0, X1_1[i], X1_2[i], X1_3[i])
            ax.quiver(0, 0, 0, X2_1[i], X2_2[i], X2_3[i])
            ax.quiver(0, 0, 0, X3_1[i], X3_2[i], X3_3[i])
            ax.quiver(0, 0, 0, 1, 0, 0, 'black')
            ax.quiver(0, 0, 0, 0, 1, 0, 'black')
            ax.quiver(0, 0, 0, 0, 0, 1, 'black')

            plt.pause(0.1)

            fig.canvas.draw()
            fig.canvas.flush_events()

    def plotOrbParam(self):

        history = self.history

        t_values = history['t']
        v1_values = history['v1'].values
        v2_values = history['v2'].values
        v3_values = history['v3'].values

        x1_values = history['x1'].values
        x2_values = history['x2'].values
        x3_values = history['x3'].values

        a = []
        e = []
        theta = []
        RAAN = []
        w = []
        i = []
        E = []

        for k in range(len(x1_values)):

            x = np.array([x1_values[k], x2_values[k], x3_values[k]])
            v = np.array([v1_values[k], v2_values[k], v3_values[k]])

            ta, te, ttheta, tRAAN, tw, ti, tE = Math.orbitalparam(x,v)

            a.append(ta)
            e.append(te)
            theta.append(ttheta*180/np.pi)
            RAAN.append(tRAAN*180/np.pi)
            w.append(tw*180/np.pi)
            i.append(ti*180/np.pi)
            E.append(tE)

        plt.figure()
        plt.plot(t_values, a)
        plt.title('a [km]')
        plt.figure()
        plt.plot(t_values, e)
        plt.title('e')
        plt.figure()
        plt.plot(t_values, theta)
        plt.title('theta [deg]')
        plt.figure()
        plt.plot(t_values, RAAN)
        plt.title('RAAN [deg]')
        plt.figure()
        plt.plot(t_values, w)
        plt.title('w [deg]')
        plt.figure()
        plt.plot(t_values, i)
        plt.title('i [deg]')
        plt.figure()
        plt.plot(t_values, E)
        plt.title('E')

    def plot(self):

        plt.show()
