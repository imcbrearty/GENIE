#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# functions for chamber shapes

def cmp_SH_shape(l_s, m_s, f_s, Rmax, dz, theta_s, phi_s):
    """
    Compute real shapes from spherical harmonic superpositions

    Input:
    l_s, m_s, f_s = array of degrees, orders, coefficients associated with each mode
    theta_s, phi_s = array of polar [0, pi] and azimuthal angles [0, 2*pi]

    Output:
    Z, Y, Z = coordinates of surface points
    """

    # verified by comparing results to MATLAB code 
    R = np.zeros(np.shape(theta_s))

    for count, (l, m, f) in enumerate (zip(l_s, m_s, f_s)):
        Yml = sp.special.sph_harm(m, l, phi_s, theta_s)

        if m == 0:
            A = f*Yml
        else:
            
            A = 2*np.real(f*Yml)
        
        #plt.imshow(np.real(A), interpolation='none')
        #plt.colorbar()
        #plt.show()

        R = R + A
    
    R = np.real(R) # get rid of spurious complex components

    normF = np.max(R)
    R = R/normF*Rmax

    # note the absolute value is needed when R is negative, which happens when plotting
    # individual spherical harmonic modes. However, we try to avoid negative R
    # for superposition of spherical haromonics
    X, Y, Z = sph2cart(np.abs(R), phi_s, theta_s) 
    Z = Z+dz

    return X, Y, Z


def cmp_SP_shape(ra, rb, alpha, beta, gamma, dz, theta_s, phi_s):
    """
    compute spheroid shapes

    Input:
    Ra, Rb = semi-major, minor axis length
    alpha, beta, gamma = extrinsic rotation angles (counterclockwise) with regard to x, y, z axes
    dz   = positive scaler, depth
    theta_s, phi_s = array of polar [0, pi] and azimuthal angles [0, 2*pi]
 
    Output:
    Z, Y, Z = coordinates of surface points
    """

    # extrinsic rotation (from Eqn. 59 in the paper "Rotation Representations and Their Conversions", 2022)

    R11 = np.cos(beta)*np.cos(gamma);                                  R12 = -np.cos(beta)*np.sin(gamma); R13 = np.sin(beta)
    R21 = np.cos(alpha)*np.sin(gamma)+np.sin(alpha)*np.sin(beta)*np.cos(gamma); R22 = np.cos(alpha)*np.cos(gamma)-np.sin(alpha)*np.sin(beta)*np.sin(gamma); R23 = -np.sin(alpha)*np.cos(beta)
    R31 = np.sin(alpha)*np.sin(gamma)-np.cos(alpha)*np.sin(beta)*np.cos(gamma); R32 = np.sin(alpha)*np.cos(gamma)+np.cos(alpha)*np.sin(beta)*np.sin(gamma); R33 = np.cos(alpha)*np.cos(beta)

    X = np.sin(theta_s)*np.cos(phi_s); Y = np.sin(theta_s)*np.sin(phi_s); Z = np.cos(theta_s)

    R = (((R11*X + R12*Y + R13*Z)**2 + (R21*X + R22*Y + R23*Z)**2)/rb**2 +  (R31*X + R32*Y +R33*Z)**2./ra**2)**(-1/2)

    X, Y, Z = sph2cart(R, phi_s, theta_s) 
    Z = Z + dz

    return X, Y, Z


def sph2cart(r, phi, tta):
    """
    Convert from spherical rto cartesian coordinates

    Input:
    r   = array of radii [0, infinity]
    phi = array of azimuthal angle [0, 2*pi]
    tta = array of polar angle [0, pi]

    Output:
    x, y, z
    """
    x = r* np.sin(tta)* np.cos(phi)
    y = r* np.sin(tta)* np.sin(phi)
    z = r* np.cos(tta)
    return x, y, z

def generateDegreeOrder(lmax):
    """
    Generate the degree-order pairs (l-m) up to (and include) a maximum degree, lmax
    """
    ls = []; ms = []

    for l in range(lmax+1):
        ls.append(np.repeat(l, l*2+1)) 
        ms.append(np.arange(-l, l+1)) 

    ls = np.concatenate(ls)
    ms = np.concatenate(ms)

    return ls, ms

def set_axes_equal(ax: plt.Axes):
    """
    Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

    
