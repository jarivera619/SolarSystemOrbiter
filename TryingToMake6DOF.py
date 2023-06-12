#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 21:39:41 2023

@author: jamesrivera
"""

# import the fxns we need 
import numpy as np 
import pandas as pd 
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# from getPositions import getPositions
from getGravityForces import Space_Object, getGForces, Space_Craft

plt.close('all')

SEC_PER_DAY = 86400

def getPositions(user_date):
    '''

    Parameters
    ----------
    user_date : datetime object
        Date of update for the space_object instances for the planets.

    Returns
    -------
    None.

    '''
    # Imports
    import numpy as np
    
    # Globals 
    global sun_df, mercury_df, venus_df, earth_df, moon_df, mars_df, \
        jupiter_df, saturn_df, uranus_df, neptune_df
    global sun, mercury, venus, earth, moon, mars, \
        jupiter, saturn, uranus, neptune

    # TEMPORARY 
    if user_date.hour != 0 :
        user_date = user_date.replace(hour=0)
    if user_date.minute != 0 :
        user_date = user_date.replace(minute=0)
    if user_date.second != 0 :
        user_date = user_date.replace(second=0)
    if user_date.microsecond != 0 :
        user_date = user_date.replace(microsecond=0)


    # find index
    idx = sun_df.Date[sun_df.Date == user_date].index[0]
    
    # next_date = user_date + timedelta(days=1)    
    # next_idx = sun_df.Date[sun_df.Date == next_date].index[0]

    # get XYZ data 
    # decide whether to call the dataframe or interpolate from previous? 
    sun_pos = sun_df[['X', 'Y', 'Z', 'RG']].loc[idx, :]
    mercury_pos = mercury_df[['X', 'Y', 'Z', 'RG']].loc[idx, :]
    venus_pos = venus_df[['X', 'Y', 'Z', 'RG']].loc[idx, :]
    earth_pos = earth_df[['X', 'Y', 'Z', 'RG']].loc[idx, :]
    moon_pos = moon_df[['X', 'Y', 'Z', 'RG']].loc[idx, :]
    mars_pos = mars_df[['X', 'Y', 'Z', 'RG']].loc[idx, :]
    jupiter_pos = jupiter_df[['X', 'Y', 'Z', 'RG']].loc[idx, :]
    saturn_pos = saturn_df[['X', 'Y', 'Z', 'RG']].loc[idx, :]
    uranus_pos = uranus_df[['X', 'Y', 'Z', 'RG']].loc[idx, :]
    neptune_pos = neptune_df[['X', 'Y', 'Z', 'RG']].loc[idx, :]    
    
    # update the objects with the positions and velocities 
    sun.setLoc(np.array([sun_pos.X, sun_pos.Y, sun_pos.Z]))
    mercury.setLoc(np.array([mercury_pos.X, mercury_pos.Y, mercury_pos.Z]))
    venus.setLoc(np.array([venus_pos.X, venus_pos.Y, venus_pos.Z]))
    earth.setLoc(np.array([earth_pos.X, earth_pos.Y, earth_pos.Z]))
    moon.setLoc(np.array([moon_pos.X, moon_pos.Y, moon_pos.Z]))
    mars.setLoc(np.array([mars_pos.X, mars_pos.Y, mars_pos.Z]))
    jupiter.setLoc(np.array([jupiter_pos.X, jupiter_pos.Y, jupiter_pos.Z]))
    saturn.setLoc(np.array([saturn_pos.X, saturn_pos.Y, saturn_pos.Z]))
    uranus.setLoc(np.array([uranus_pos.X, uranus_pos.Y, uranus_pos.Z]))
    neptune.setLoc(np.array([neptune_pos.X, neptune_pos.Y, neptune_pos.Z]))
    
    return None 



def loadPickles():
    '''
    Load the pickle files and save the information to a Pandas dataframe 

    Returns
    -------
    None.

    '''
    global sun_df, mercury_df, venus_df, earth_df, moon_df, mars_df, \
        jupiter_df, saturn_df, uranus_df, neptune_df

    sun_df = pd.read_pickle('full_pickle_Sun.pkl')
    mercury_df = pd.read_pickle('full_pickle_Mercury.pkl')
    venus_df = pd.read_pickle('full_pickle_Venus.pkl')
    earth_df = pd.read_pickle('full_pickle_Earth.pkl')
    moon_df = pd.read_pickle('full_pickle_Moon.pkl')
    mars_df = pd.read_pickle('full_pickle_Mars.pkl')
    jupiter_df = pd.read_pickle('full_pickle_Jupiter.pkl')
    saturn_df = pd.read_pickle('full_pickle_Saturn.pkl')
    uranus_df = pd.read_pickle('full_pickle_Uranus.pkl')
    neptune_df = pd.read_pickle('full_pickle_Neptune.pkl')
    

    return None



def integrateGrav():
    '''
    Obsolete symplectic Euler integrator. Actually worked well with small enough dt. 

    Returns
    -------
    None.

    '''
    global dt, me
    DAYS_2_SEC = 24*60*60
    
    me.setVel(me.vel + me.acc * dt * DAYS_2_SEC)
    me.setLoc(me.loc + me.vel * dt * DAYS_2_SEC + 0.5 * me.acc * dt**2 * DAYS_2_SEC)
    # print(me.loc)
    
    # print(me.vel)
    # print('')
    
    return None 



def getDeriv(t, x) :
    '''
    Derivative function for IVP solver

    Parameters
    ----------
    t : float
        Time input (units = days).
    x : 6x1 numpy array
        State vector for integration [x, y, z, vx, vy, vz]

    Returns
    -------
    dx : 6x1 numpy array
        First order derivative for solve_ivp function.

    '''
    global sun, mercury, venus, earth, moon, mars, jupiter, saturn, uranus, neptune
    global me
    global date_ctr, t_delta_v, delta_v
    global n_ctr, burn , frame_ctr
    
    frame_limit = 200
    
    if t > (n_ctr) :
        n_ctr += 1
        date_ctr += timedelta(days=1)
        getPositions(date_ctr)
        
        if n_ctr % 10 == 0 : 
            print('Updating positions (t = {} days)'.format(str(n_ctr)))
        
    dx = np.zeros([6]) # iniitialize dx 
    
    # velocities go to positions 
    dx[0] = x[3] # x_dot = vx
    dx[1] = x[4] # y_dot = vy
    dx[2] = x[5] # z_dot = vz 
    
     
    if (t > t_delta_v) and frame_ctr < frame_limit :#and (burn == False) 
        if frame_ctr < 1 : 
            print('BURN')
            # print(t)
            plotBurn()
        # print(dx[0], dx[1])
        me.deltaV(delta_v / frame_limit, 'retrograde')    
        dx[0] = me.vx
        dx[1] = me.vy
        dx[2] = me.vz
        x[3] = me.vx
        x[4] = me.vy
        x[5] = me.vz
        frame_ctr += 1
        # print()
        
        if frame_ctr == frame_limit:
            me.setVel(x[3:6])
            me.setLoc(x[0:3])
            me.calcKep()
            # print(t)
            plotKep()
        # print(frame_ctr)
        # print(dx[0], dx[1])
        
    # if burn == True :
    #     print('pause')

    # sum up accels from gravity (point-like assumptions)
    accel = getGForces(sun, me) / me.mass 
    accel += getGForces(mercury, me) / me.mass 
    accel += getGForces(venus, me) / me.mass 
    accel += getGForces(earth, me) / me.mass
    # accel += getGForces(moon, me) / me.mass 
    accel += getGForces(mars, me) / me.mass 
    accel += getGForces(jupiter, me) / me.mass 
    # accel += getGForces(saturn, me) / me.mass 
    # accel += getGForces(uranus, me) / me.mass 
    # accel += getGForces(neptune, me) / me.mass 
    
    # set accelerations 
    dx[3] = accel[0] # x_ddot = ax
    dx[4] = accel[1] # y_ddot = ay
    dx[5] = accel[2] # z_ddot = az
    
    # update spacecraft 
    # print(dx[0], dx[1])
    me.setVel(x[3:6])
    me.setLoc(x[0:3])
    # print(dx[0], dx[1])
    
    return dx 



def plotPlanets() :
    '''
    Plot the locations of the planets day by day 

    Returns
    -------
    None.

    '''
    global starting_date, T, ax1, fig
    global sun, mercury, venus, earth, moon, mars, jupiter, saturn, uranus, neptune
    global plot_mode
    
    # colors
    csun = '#FFFFFF'
    cmars = '#E8883F'
    cearth = '#6ACCF6'
    cvenus = '#F2E318'
    cmercury = '#AEB2B1'
    cjupiter = '#FBBE56'
    csaturn = '#F9E10B'
    curanus = '#91EDD2'
    cneptune = '#1033F9'
    # cspace = '#1F1D2A'
    if plot_mode == 'inner':
        delta = 1
    elif plot_mode == 'outer':
        delta = 10
    
    date_list = [starting_date + timedelta(days=x) for x in np.arange(0, T, delta, dtype=float)]
    
    for i, date in enumerate(date_list) :
        getPositions(date)
        
        ax1.scatter(sun.x, sun.y, sun.z, facecolor=csun, label='Sun')
        ax1.scatter(mercury.x, mercury.y, mercury.z, facecolor=cmercury, label='Mercury')
        ax1.scatter(venus.x, venus.y, venus.z, facecolor=cvenus, label='Venus')
        ax1.scatter(earth.x, earth.y, earth.z, facecolor=cearth, label='Earth')
        ax1.scatter(mars.x, mars.y, mars.z, facecolor=cmars, label='Mars')
        
        if plot_mode == 'outer' :
            ax1.scatter(jupiter.x, jupiter.y, jupiter.z, facecolor=cjupiter)
            ax1.scatter(saturn.x, saturn.y, saturn.z, facecolor=csaturn)
            ax1.scatter(uranus.x, uranus.y, uranus.z, facecolor=curanus)
            ax1.scatter(neptune.x, neptune.y, neptune.z, facecolor=cneptune)  
            
    ax1.set_aspect('equal')
    
    return None 


def plotKep():
    '''
    Plots the keplerian orbit of the initial conditions of the space craft

    Returns
    -------
    None.

    '''

    global me, ax1, fig
    
    def Rx(theta):
        return np.array([[ 1, 0           , 0           ], \
                   [ 0, np.cos(theta),-np.sin(theta)], \
                   [ 0, np.sin(theta), np.cos(theta)]])
  
    def Ry(theta):
        return np.array([[ np.cos(theta), 0, np.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-np.sin(theta), 0, np.cos(theta)]])
  
    def Rz(theta):
        return np.array([[ np.cos(theta), -np.sin(theta), 0 ],
                   [ np.sin(theta), np.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])
    
    p = me.a * (1 - me.e**2)
    nu = np.arange(0, 2*np.pi, 0.01)
    r = p / (1 + me.e*np.cos(nu))
    
    x = r*np.cos(nu)
    y = r*np.sin(nu)
    z = np.zeros([len(x)])
    
    r_table = np.array([x, y, z])
    
    r_table = np.matmul(Rz(me.lil_omega), r_table)
    r_table = np.matmul(Rx(me.i), r_table)
    r_table = np.matmul(Rz(me.big_omega), r_table)
    
    ax1.plot(r_table[0, :], r_table[1, :], r_table[2, :])
    ax1.set_aspect('equal')
    
    return None 


def plotBurn() :
    global ax1, fig, me
    
    ax1.scatter(me.x, me.y, me.z, color='red', marker='X', s=150)



'''
========================
Beginning of scripting 
========================
'''

# moding parameters
plot_mode = 'inner'
print('Starting new simulation...')

# initialize dataframe variables in global scope
sun_df = None
mercury_df = None
venus_df = None
earth_df = None
moon_df = None
mars_df = None
jupiter_df = None
saturn_df = None
uranus_df = None
neptune_df = None

# masses of the sun and planets
m_sun = 1.9891e30
m_earth = 5.9722e+24
m_mercury = 0.0553*m_earth
m_venus = 0.815*m_earth
m_mars = 0.1075*m_earth
m_moon = 0.0553*m_earth
m_jupiter = 317.8*m_earth
m_saturn = 95.2*m_earth
m_uranus = 14.6*m_earth
m_neptune = 17.2*m_earth

# initialize space object variables in global scope
sun = Space_Object('sun', m_sun, 0, 0, 0, 0, 0, 0)
mercury = Space_Object('mercury', m_mercury, 0, 0, 0, 0, 0, 0)
venus = Space_Object('venus', m_venus, 0, 0, 0, 0, 0, 0)
earth = Space_Object('earth', m_earth, 0, 0, 0, 0, 0, 0)
moon = Space_Object('moon', 10, 0, 0, 0, 0, 0, 0)
mars = Space_Object('mars', m_mars, 0, 0, 0, 0, 0, 0)
jupiter = Space_Object('jupiter', m_jupiter, 0, 0, 0, 0, 0, 0)
saturn = Space_Object('saturn', m_saturn, 0, 0, 0, 0, 0, 0)
uranus = Space_Object('uranus', m_uranus, 0, 0, 0, 0, 0, 0)
neptune = Space_Object('neptune', m_neptune, 0, 0, 0, 0, 0, 0)

# load the .pkl files into the dataframe variables 
loadPickles()
print('Data saved...')

#==============================================================================
# USER SETTINGS 
# define some constants 
T = 500 # time you want to run it for (days)
# starting_date = '06-19-1995'
YYYY = 2023
MM = 8
DD = 9
starting_date = datetime(YYYY, MM, DD, 0, 0, 0)
date_ctr = datetime(YYYY, MM, DD, 0, 0, 0)
t_delta_v = 130
delta_v = 5*SEC_PER_DAY

# start the instantiations of the objects with the dataframe data
getPositions(starting_date)
print('Positions initialized...')

me = Space_Craft('Space Craft', 10, 1.5e8, 0, 1e6, 0, 30*SEC_PER_DAY, 1e-1*SEC_PER_DAY, 0, 0, 0)

#==============================================================================

dt = 1 # time step in days 

date_list = [starting_date + timedelta(days=x) for x in np.arange(0, T, dt, dtype=float)]

# initialize position variables 
me_pos = np.zeros([len(date_list), 3])
sun_pos = np.zeros([len(date_list), 3])
earth_pos = np.zeros([len(date_list), 3])
mercury_pos = np.zeros([len(date_list), 3])
venus_pos = np.zeros([len(date_list), 3])
mars_pos = np.zeros([len(date_list), 3])
jupiter_pos = np.zeros([len(date_list), 3])
saturn_pos = np.zeros([len(date_list), 3])
uranus_pos = np.zeros([len(date_list), 3])
neptune_pos = np.zeros([len(date_list), 3])
moon_pos = np.zeros([len(date_list), 3])

# create plot
cspace = '#1F1D2A'
fig = plt.figure(figsize=(10,10), layout='tight', facecolor=cspace)
ax1 = fig.add_subplot(projection='3d', facecolor=cspace)
plotKep()

# solver 
print('Solving...')
t_span = np.array([0, T])
y0 = np.array(me.state)
n_ctr = 0
frame_ctr = 0
burn = False
rtol = 1e-5
atol = 1e-10

sol = solve_ivp(getDeriv, t_span, y0, t_eval=np.arange(0, T, dt), \
                method='RK23', rtol=rtol, atol=atol, max_step=0.05)

# plot space craft coordinates
ax1.scatter(sol.y[0,:], sol.y[1,:], sol.y[2,:], label='Space Craft', alpha=0.5, \
            color='cyan', marker='H', edgecolors='none')
ax1.set_xlabel('X Axis', color='white')
ax1.set_ylabel('Y Axis', color='white')
ax1.set_aspect('equal')

plotPlanets()

ax1.legend(['Keplerian Orbit 1', 'BURN', 'Keplerian Orbit 2', me.name, 'Sun', 'Mercury', 'Venus', 'Earth', 'Mars'])
ax1.set_title('Inner Planets', color='white')
print('Done!')




















'''
OLD SYMPLECTIC EULER CODE 
# Loop through time 
for i, date in enumerate(date_list):
    # get data of the positions of planets and update their instantiations 
    # turn getPositions.py into an update positions function that returns None 
    getPositions(date)
    # print(i)
    # update all accels for the current frame (only for spacecraft forces (eom))
    # me.setAcc(getGForces(sun, me) / me.mass / 1e3) # divide by 1000 again to convert to km/s^2
    
    # integrate 
    # integrateGrav()
    t_span = np.array([0, 86400])
    y0 = me.state
    sol = solve_ivp(getDeriv, t_span, y0, t_eval=[86400], method='RK45', max_step=600)
    
    me.setLoc(sol.y.flatten()[0:3])
    me.setVel(sol.y.flatten()[3:6])
    
    # record locations of the spacecraft and planets for plot in vectors 
    me_pos[i, :] = me.loc
    sun_pos[i, :] = sun.loc
    earth_pos[i, :] = earth.loc
    mercury_pos[i, :] = mercury.loc
    venus_pos[i, :] = venus.loc
    mars_pos[i, :] = mars.loc
    moon_pos[i, :] = moon.loc
    jupiter_pos[i, :] = jupiter.loc
    saturn_pos[i, :] = saturn.loc
    uranus_pos[i, :] = uranus.loc
    neptune_pos[i, :] = neptune.loc
    
    

# create plot of all bodies 
cspace = '#1F1D2A'
fig = plt.figure(figsize=(10,10), layout='tight', facecolor=cspace)
ax1 = fig.add_subplot(projection='3d', facecolor=cspace)
ax1.scatter(me_pos[:, 0], me_pos[:, 1], me_pos[:, 2])
ax1.scatter(sun_pos[:, 0], sun_pos[:, 1], sun_pos[:, 2])
ax1.scatter(earth_pos[:, 0], earth_pos[:, 1], earth_pos[:, 2])
ax1.scatter(mercury_pos[:, 0], mercury_pos[:, 1], mercury_pos[:, 2])
ax1.scatter(venus_pos[:, 0], venus_pos[:, 1], venus_pos[:, 2])
ax1.set_xlabel('XXX')
ax1.set_ylabel('YYY')
ax1.set_aspect('equal')

'''













