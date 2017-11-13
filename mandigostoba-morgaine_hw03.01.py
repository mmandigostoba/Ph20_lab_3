#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Ph 20 Problem Set #3
Morgaine Mandigo-Stoba
"""

import numpy as np
import matplotlib.pyplot as plt

# Part 1
# Problem #1
def expEuler(x0, v0, t, h):
    '''
    Inputs:
        x0 - the initial diaplacement
        v0 - the initial velocity
        t - the time to be integrated over
        h - the time step
        
    Outputs:
        x - array of the displacements at each time step
        v - array of the velocities at each time step
        ts - the array of times
    '''
    numSteps = int(t / h)
    x = np.zeros(numSteps + 1)
    v = np.zeros(numSteps + 1)
    ts = np.zeros(numSteps + 1)
    x[0] = x0
    v[0] = v0
    
    for i in range(numSteps):
        ts[i + 1] = ts[i] + h
        x[i + 1] = x[i] + (v[i] * h)
        v[i + 1] = v[i] - (x[i] * h)
    
    return(x, v, ts)
   
(x1, v1, ts1) = expEuler(1, 0, 12, 0.1)
plt.plot(ts1, x1, label='Displacement')
plt.plot(ts1, v1, label='Velocity')
plt.xlabel('Time (t)')
plt.title('Explicit Euler Method')
plt.legend(loc=3)
plt.savefig('exp_euler.eps')
plt.show()
    
# Problem 2
def eulerError(x, v, ts):
    '''
    Inputs:
        x - the appriximate displacements
        v - the approximate velocities
        ts - the time steps iterated over
        
    Outputs:
        errX - the error in the approximate displacements
        errV - the error in the approximate velocities
    '''
    realX = np.cos(ts)
    realV = - np.sin(ts)
    errX = realX - x
    errV = realV - v
    return(errX, errV)
  
(x1, v1, ts1) = expEuler(1, 0, 12, 0.1)
(errX1, errV1) = eulerError(x1, v1, ts1)

plt.plot(ts1, errX1, label='Displacement')
plt.plot(ts1, errV1, label='Velocity')
plt.xlabel('Time (t)')
plt.ylabel('Error')
plt.title('Explicit Euler Method Error')
plt.legend(loc=3)
plt.savefig('exp_euler_err.eps')
plt.show()

# Problem 3
hvals = np.array([.1, (.1 / 2), (.1 / 4), (.1 / 8), (.1 / 16)])
maxErr = np.zeros(len(hvals))
for i in range(len(hvals)):
    (x, v, ts) = expEuler(1, 0, 12, hvals[i])
    (errX, errV) = eulerError(x, v, ts)
    maxErr[i] = np.max(errX)

plt.plot(hvals, maxErr)
plt.xlabel('Time Step')
plt.ylabel('Maximum Error')
plt.title('Explicit Euler Method Maximum Error')
plt.savefig('exp_euler_max_err.eps')
plt.show()

# Problem 4
(x1, v1, ts1) = expEuler(1, 0, 12, 0.1)
energy1 = (x1**2) + (v1**2)

plt.plot(ts1, energy1)
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Explicit Euler Method Energy')
plt.savefig('exp_euler_energy.eps')
plt.show()

# Problem 5
def impEuler(x0, v0, t, h):
    '''
    Inputs:
        x0 - the initial displacement
        v0 - the initial velocity
        t - the time to be integrated over
        h - the time step
    
    Outputs:
        x - array of the displacements at each time step
        v - array of the velocities at each time step
        ts- the array of times
    '''
    numSteps = int(t / h)
    x = np.zeros(numSteps + 1)
    v = np.zeros(numSteps + 1)
    ts = np.zeros(numSteps + 1)
    x[0] = x0
    v[0] = v0
    
    for i in range(numSteps):
        ts[i + 1] = ts[i] + h
        x[i + 1] = (x[i] + (v[i] * h)) / (1 + (h**2))
        v[i + 1] = (v[i] - (x[i] * h)) / (1 + (h**2))
    
    return(x, v, ts)

(x2, v2, ts2) = impEuler(1, 0, 12, 0.1)
plt.plot(ts2, x2, label='Displacement')
plt.plot(ts2, v2, label='Velocity')
plt.xlabel('Time (t)')
plt.title('Implicit Euler Method')
plt.legend(loc=4)
plt.savefig('imp_euler.eps')
plt.show()

(errX2, errV2) = eulerError(x2, v2, ts2)

plt.plot(ts2, errX2, label='Displacement')
plt.plot(ts2, errV2, label='Velocity')
plt.xlabel('Time (t)')
plt.ylabel('Error')
plt.title('Implicit Euler Method Error')
plt.legend(loc=3)
plt.savefig('imp_euler_err.eps')
plt.show()

hvals = np.array([.1, (.1 / 2), (.1 / 4), (.1 / 8), (.1 / 16)])
maxErr = np.zeros(len(hvals))
for i in range(len(hvals)):
    (x, v, ts) = impEuler(1, 0, 12, hvals[i])
    (errX, errV) = eulerError(x, v, ts)
    maxErr[i] = np.max(errX)

plt.plot(hvals, maxErr)
plt.xlabel('Time Step')
plt.ylabel('Maximum Error')
plt.title('Implicit Euler Method Maximum Error')
plt.savefig('imp_euler_max_err.eps')
plt.show()

energy2 = (x2**2) + (v2**2)

plt.plot(ts2, energy2)
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Implicit Euler Method Energy')
plt.savefig('imp_euler_energy.eps')
plt.show()

# Part 2
# Problem 1
(x1, v1, ts1) = expEuler(1, 0, 12, 0.01)
(x2, v2, ts2) = impEuler(1, 0, 12, 0.01)

plt.plot(x1, v1, label='Explicit Euler Method')
plt.xlabel('Displacement')
plt.ylabel('Velocity')
plt.plot(x2, v2, label='Implicit Euler Method')
plt.legend(loc=1)
plt.title('Explicit and Implicit Euler Method Phase Space')
plt.savefig('exp_imp_phase.eps')
plt.show()

# Problem 2
def sympEuler(x0, v0, t, h):
    '''
    Inputs:
        x0 - the initial displacement
        v0 - the initial velocity
        t - the time to be integrated over
        h - the time step
    
    Outputs: 
        x - array of the displacements at each time step
        v - array of the velocities at each time step
        ts- the array of times
    '''
    numSteps = int(t / h)
    x = np.zeros(numSteps + 1)
    v = np.zeros(numSteps + 1)
    ts = np.zeros(numSteps + 1)
    x[0] = x0
    v[0] = v0
    
    for i in range(numSteps):
        ts[i + 1] = ts[i] + h
        x[i + 1] = x[i] + (v[i] * h)
        v[i + 1] = v[i] - (h * x[i + 1])
        
    return(x, v, ts)
    
(x3, v3, ts3) = sympEuler(1, 0, 12, 0.01)
plt.plot(x3, v3)
plt.xlabel('Displacement')
plt.ylabel('Velocity')
plt.title('Sympletic Euler Method Phase Space')
plt.savefig('symp_phase.eps')
plt.show()

# Problem 3
energy = (x3**2) + (v3**2)

plt.plot(ts3, energy)
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Sympletic Euler Method Energy vs. Time')
plt.savefig('symp_energy.eps')
plt.show()
    







    

    


