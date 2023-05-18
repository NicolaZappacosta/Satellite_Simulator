##
import numpy as np

## Quaternion handling

def quatmolt(q1,q2):

    x1, x2, x3, x0 = q1
    y1, y2, y3, y0 = q2

    result = np.array([+x0 * y1 + x3 * y2 - x2 * y3 + x1 * y0, -x3 * y1 + x0 * y2 + x1 * y3 + x2 * y0, x2 * y1 - x1 * y2 + x0 * y3 + x3 * y0, -x1 * y1 - x2 * y2 - x3 * y3 + x0 * y0],dtype= np.float64)
    result = np.reshape(result,4)

    return result

def errorquat(q1,q2):

    x1, x2, x3, x0 = q1[0], q1[1], q1[2], q1[3]
    y1, y2, y3, y0 = -q2[0], -q2[1], -q2[2], q2[3]

    result = np.array([+x0 * y1 + x3 * y2 - x2 * y3 + x1 * y0, -x3 * y1 + x0 * y2 + x1 * y3 + x2 * y0,
                       x2 * y1 - x1 * y2 + x0 * y3 + x3 * y0, -x1 * y1 - x2 * y2 - x3 * y3 + x0 * y0], dtype=np.float64)
    result = np.reshape(result, 4)

    return result

def mat2quat(A): # http://www.ladispe.polito.it/corsi/Meccatronica/02JHCOR/2011-12/Slides/Shuster_Pub_1993h_J_Repsurv_scan.pdf

    x0 = 1/2*np.sqrt(1+A[0,0]**2+A[1,1]**2+A[2,2]**2)
    x1 = 1/(4*x0)*(A[1,2]-A[2,1])
    x2 = 1/(4*x0)*(A[2,0]-A[0,2])
    x3 = 1/(4*x0)*(A[0,1]-A[1,0])

    result = np.reshape([x1, x2, x3, x0], 4)

    return result

def quat2mat(q): # http://www.ladispe.polito.it/corsi/Meccatronica/02JHCOR/2011-12/Slides/Shuster_Pub_1993h_J_Repsurv_scan.pdf

    skew_q = np.array([0, q[2], -q[1], -q[2], 0, q[0], q[2], -q[1], 0])
    skew_q = skew_q.reshape(3, 3)

    A = (q[3]**2 - np.linalg.norm(q[0:3])**2)*np.identity(3)+2*np.matmul(q[0:3], q[0:3].T)+2*q[3]*skew_q

    return A

## Vector Rotation Handling

def ECI2ECEF(X,alpha_0,omega,t):

    angle = omega*t-alpha_0
    R = np.array([np.cos(angle), np.sin(angle), 0, -np.sin(angle), np.cos(angle), 0, 0, 0, 1])
    R = R.reshape(3, 3)

    return np.matmul(R, X)

def ECEF2ECI(X,alpha_0,omega,t):

    angle = omega * t - alpha_0
    R = np.array([np.cos(angle), np.sin(angle), 0, -np.sin(angle), np.cos(angle), 0, 0, 0, 1])
    R = R.reshape(3, 3)
    R = R.T

    return np.matmul(R, X)

def ECEF2GC(X,lon,lat):

    R1 = np.array([np.cos(lon), np.sin(lon), 0, -np.sin(lon), np.cos(lon), 0, 0, 0, 1])
    R1 = R1.reshape(3, 3)

    R2 = np.array([np.cos(lat),0,np.sin(lat),0,1,0,-np.sin(lat),0,np.cos(lat)])
    R2 = R2.reshape(3, 3)

    R = np.matmul(R2,R1)

    return np.matmul(R, X)

def GC2NED(X):

    T = np.array([0,0,1,0,1,0,-1,0,0])

    return np.matmul(T, X)

def ECI2NED(X,alpha_0,omega,lon,lat,t):

    Xecef = ECI2ECEF(X,alpha_0,omega,t)
    Xgc = ECEF2GC(Xecef,lat,lon)
    Xned = GC2NED(Xgc)

    return Xned

## Matrix Rotation handling

def RotMat_ECI2ECEF(alpha_0,omega,t):

    angle = omega*t-alpha_0
    R = np.array([np.cos(angle),np.sin(angle),0,-np.sin(angle),np.cos(angle),0,0,0,1])
    R = R.reshape(3,3)

    return R

def RotMat_ECEF2ECI(alpha_0,omega,t):

    angle = omega * t - alpha_0
    R = np.array([np.cos(angle), np.sin(angle), 0, -np.sin(angle), np.cos(angle), 0, 0, 0, 1])
    R = R.reshape(3, 3)
    R = R.T

    return R

def RotMat_ECEF2GC(lon,lat):

    R1 = np.array([np.cos(lon), np.sin(lon), 0, -np.sin(lon), np.cos(lon), 0, 0, 0, 1])
    R1 = R1.reshape(3, 3)

    R2 = np.array([np.cos(lat),0,np.sin(lat),0,1,0,-np.sin(lat),0,np.cos(lat)])
    R2 = R2.reshape(3, 3)

    R = np.matmul(R2,R1)

    return R

def RotMat_GC2NED():

    T = np.array([0, 0, 1, 0, 1, 0, -1, 0, 0])

    return T.reshape(3, 3)

def RotMat_ECI2NED(alpha_0, omega, lon, lat, t):

    Reci2ecef = RotMat_ECI2ECEF(alpha_0, omega, t)
    Recef2gc = RotMat_ECEF2GC(lat, lon)
    Rgc2ned = RotMat_GC2NED()

    return np.matmul(Rgc2ned, np.matmul(Recef2gc, Reci2ecef))

## LatLon Handling

def GC(Xecef):

    lat = np.arcsin(Xecef[2]/np.linalg.norm(Xecef))
    lon = np.arctan2(Xecef[1], Xecef[0])
    R = np.linalg.norm(Xecef)

    return lat, lon, R


## Orbital Parameteres Handling

def orbitalparam(x,v, mu = 398600.4418):

    h = np.cross(x,v)

    E = -mu/np.linalg.norm(x)+np.linalg.norm(v)**2/2        # Energy

    a = -mu/(2*E)                                           # Semi-axis major
    i = np.arccos(h[2]/np.linalg.norm(h))                   # Inclination
    RAAN = np.arctan2(h[0],[1])                             # RAAN

    e_v = -x/np.linalg.norm(x) + np.cross(v, h)/mu          # Eccentricity vector
    e = np.linalg.norm(e_v)                                 # Eccentricity module

    n = np.cross(np.array([0, 0, 1]), h)/np.linalg.norm(np.cross(np.array([0, 0, 1]), h))       # Unit vector of RAAN

    w = np.arcsin(np.dot(n, e_v))
    if e_v[2] < 0:
        w += -2*np.pi                                       # Argument of pericentre

    t = np.cross(h, e_v)/np.linalg.norm(np.cross(h, e_v))

    C_theta = np.dot(e_v,x)/np.linalg.norm(x)
    S_theta = np.dot(t, x) / np.linalg.norm(x)

    theta = np.arctan2(S_theta,C_theta)

    return a,e,theta,RAAN,w,i,E