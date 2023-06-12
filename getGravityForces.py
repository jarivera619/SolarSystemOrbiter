#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 21:06:03 2023

@author: jamesrivera
"""

import numpy as np
from numpy.linalg import norm

G = 6.6743e-11 # N * m^2 / kg^2
m_sun = 1.9891e30
AU_2_KM = 1.496e+8 # m / AU
KM_2_M = 1e-3 # km / m


def getGForces(A, B):
    """

    Parameters
    ----------
    A : Space_Object()
        the object that is doing the exerting.
    B : Space_Object()
        the object being exerted on.

    Returns
    -------
    F : vector3
        force of gravity on object B because of object A.

    """
    M1 = A.mass
    M2 = B.mass
    r = norm(B.loc - A.loc)
    r_hat = (B.loc - A.loc) / r
    
    F = -G * M1 * M2 * r_hat / r**2 # kg*m/sec^2
    F = F * KM_2_M**3 # kg*km/sec^2 (3 copies of r above)
    # 86400 sec/day 
    F = F * 24*60*60 * 24*60*60 # kg*km/day^2
    
    
    return F

class Space_Object():
    def __init__(self, name, mass, x, y, z, vx, vy, vz):
        self.name  = name
        self.mass  = mass
        self.x     = x
        self.y     = y
        self.z     = z
        self.vx    = vx
        self.vy    = vy
        self.vz    = vz
        self.loc   = np.array([x, y, z])
        self.vel   = np.array([vx, vy, vz])
        
    def setLoc(self, new_loc):
        """

        Parameters
        ----------
        new_loc : vector3
            the new location of the object.

        Returns
        -------
        None.

        """
        self.loc = new_loc
        self.x   = new_loc[0]
        self.y   = new_loc[1]
        self.z   = new_loc[2]
        
        return None
        
    def setVel(self, new_vel):
        """

        Parameters
        ----------
        new_vel : vector3
            the new velocity of the object.

        Returns
        -------
        None.

        """
        self.vel = new_vel
        self.x   = new_vel[0]
        self.y   = new_vel[1]
        self.z   = new_vel[2]
        
        return None

class Space_Craft(Space_Object):
    def __init__(self, name, mass, x, y, z, vx, vy, vz, ax, ay, az):
        super().__init__(name, mass, x, y, z, vx, vy, vz)
        self.ax  = ax
        self.ay  = ay
        self.az  = az
        self.acc = np.array([ax, ay, az])
        self.state = np.array([x, y, z, vx, vy, vz])
        self.calcKep()
        
    def setAcc(self, new_acc):
        self.acc = new_acc
        self.ax = new_acc[0].copy()
        self.ay = new_acc[1].copy()
        self.az = new_acc[2]
        
    def setLoc(self, new_loc):
        """

        Parameters
        ----------
        new_loc : vector3
            the new location of the object.

        Returns
        -------
        None.

        """
        self.loc = new_loc.copy()
        self.x   = new_loc[0].copy()
        self.y   = new_loc[1].copy()
        self.z   = new_loc[2].copy()
        self.state = np.array([*new_loc.copy(), *self.state[3:6].copy()])
        return None
        
    def setVel(self, new_vel):
        """

        Parameters
        ----------
        new_vel : vector3
            the new velocity of the object.

        Returns
        -------
        None.

        """
        self.vel = new_vel.copy()
        self.vx   = new_vel[0].copy()
        self.vy   = new_vel[1].copy()
        self.vz   = new_vel[2].copy()
        self.state = np.array([*self.state[0:3].copy(), *new_vel.copy()])
        return None
    
    def deltaV(self, v, direction) :
        v = np.abs(v)
        
        v_hat = self.vel / norm(self.vel)
        
        if direction == 'prograde' :
            v_hat = 1 * v_hat
        elif direction == 'retrograde' :
            v_hat = -1 * v_hat
            
        delta_v = v * v_hat
        
        self.setVel(self.vel + delta_v)
        return None
        
        
    def calcKep(self) :
        mu_sun = G * m_sun # m^3 / sec^2
        mu_sun = mu_sun * 86400**2 / 1e3**3 # km^3 / day^2
        
        x_hat = np.array([1, 0, 0])
        y_hat = np.array([0, 1, 0])
        z_hat = np.array([0, 0, 1])
        
        
        self.h = np.cross(self.loc, self.vel) # angular velocity vector
        self.e_vector = 1/mu_sun * np.cross(self.vel, self.h) - self.loc/norm(self.loc) # eccentricity vector
        self.e = norm(self.e_vector) # eccentricity value 
        
        self.i = np.arccos(np.dot(self.h, z_hat) / norm(self.h)) # inclination angle 
        
        if np.abs(self.i) < 1e-4 or np.abs(np.abs(self.i) - np.pi) < 1e-4 :
            self.n_hat = x_hat
            self.big_omega = 0
        else :
            self.n_hat = np.cross(z_hat, self.h) / norm(np.cross(z_hat, self.h)) # line of nodes vector 
            self.big_omega = np.arccos(np.dot(self.n_hat, x_hat)) # longitude of nodes
        
        if np.dot(self.n_hat, y_hat) < 0 : # check for outside of [0, pi]
            self.big_omega = 2 * np.pi - self.big_omega
        
        if np.abs(self.e) < 1e-4 :
            self.nu = 0
            self.lil_omega = 0
            
        else :
            self.nu = np.arccos(np.dot(self.loc, self.e_vector) / norm(self.loc) / self.e) # angle in orbit
            self.lil_omega = np.arccos(np.dot(self.e_vector, self.n_hat) / self.e )
                        
        if np.dot(self.vel, self.loc) < 0 : # check for outside of [0, pi]
            self.nu = 2 * np.pi - self.nu
            
        if np.dot(self.e_vector, z_hat) < 0 : # check for outside of [0, pi]
            self.lil_omega = 2 * np.pi - self.lil_omega
            
        self.a = norm(self.h)**2 / mu_sun / (1 - self.e**2)
        
        # ADD PROTECTIONS FOR SPECIAL CASES WITH EULER ANGLES 
        
        
