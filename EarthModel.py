### Earth Gravity model functions

# Import

import numpy as np

# Gravity functions

def sphericGravity():

    mu = 398600.4418

    def fun_gravity(x):

        position = x[7:10]
        radius = np.linalg.norm(position)
        g = -mu*position/(radius**3)
        g = np.array(g).reshape(3)

        return g

    return fun_gravity


def harmonicGravity():                  # https://spsweb.fltops.jpl.nasa.gov/portaldataops/mpg/MPG_Docs/Source%20Docs/gravity-SphericalHarmonics.pdf

    mu = 398600.4418        # Km
    ae = 63378136.3/1000    # Km

    C_20 = -484165.48*10**-12       # Adim
    C_30 = 957.12*10**-12           # Adim

    def fun_gravity(x):

        position = x[7:10]

        X = position[0]
        Y = position[1]
        Z = position[2]

        radius = np.linalg.norm(position)

        # Coefficients

        C_2 = -mu/(radius**2)*(ae/radius)**2*C_20
        C_3 = -mu/(radius**2)*(ae/radius)**3*C_30

        g = -mu*position/(radius**3)
        g[0] += C_2*(15/2*(Z/radius)**2-3/2)*X/radius+C_3*(35/2*(Z/radius)**3-15/2*Z/radius)*X/radius
        g[1] += C_2*(15/2*(Z/radius)**2-3/2)*Y/radius+C_3*(35/2*(Z/radius)**3-15/2*Z/radius)*Y/radius
        g[2] += C_2*(15/2*(Z/radius)**2-9/2)*Z/radius+C_3*(35/2*(Z/radius)**4-15*(Z/radius)**2+3/2)

        g = np.array(g).reshape(3)

        return g

    return fun_gravity