#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 18:03:52 2021

@author: Matthew Hutson
"""

from math import pi, sin, cos
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


seed = random.randint(0,1000)
seed = 426
# seed = 639
print('\nseed =', seed)
random.seed(seed)

size = 1000
horizon = round(.4*size)
radius = 5
numWaves = 4

    
def updatefig2b(frame, contents, size, waves, ball_state, ys):     
    # print(ball_state[1], ys[0])
    xs = np.arange(0,size,1)
    y_arrays = np.zeros([numWaves, size])
    for w in range(numWaves):  # wavelength, amplitude, offset, speed
        y_arrays[w] = -np.sin(xs / waves[w][0] + waves[w][2] + frame*[1,-1][w%2]/waves[w][3]) * waves[w][1] # + horizon       
    prev_ys = np.copy(ys)
    ys[:] = np.sum(y_arrays, axis=0) / numWaves * 2 + horizon # mean, amped
    # print(ys[0], prev_ys[0])
    # ys = np.full(size, horizon, dtype=float) # flat ground
    # ys = size-xs[:] # slanted ground
    contents[0].set_ydata(ys)
    
    ball_state[0] = (ball_state[0] + 1/30 * ball_state[2]) % size
    ball_state[1] = (ball_state[1] + 1/30 * ball_state[3])   
    x_int = int(ball_state[0])
    ball_state[1] = max(ball_state[1], ys[x_int]+radius)
    # print(round(ball_state[0]), round(ball_state[1]), round(ball_state[2]), round(ball_state[3]))
    if ball_state[1]-radius <= ys[x_int] and ball_state[3] < 0: # ball hits wave
        # print('bounce')
        # print('\nv before bounce', (ball_state[2]**2 + ball_state[3]**2)**.5)
        # print('xv =', ball_state[2])
        # print('yv =', ball_state[3])
        # print('y =', ball_state[1])
        slope = (ys[(x_int+1)%size] - ys[(x_int-1)%size]) / ((x_int+1)%size - (x_int-1)%size) # rise/run ratio
        # print('slope ratio', slope)
        slope = (np.arctan(slope) + pi/2) % (2*pi) # radians
        # print('slope rads', slope)
        traj = ball_state[3]/(ball_state[2]+.000001) # ball's trajectory as ratio
        # print('traj ratio', traj)
        flip = pi if ball_state[2]<0 else 0
        traj = (np.arctan(traj) + pi/2 + flip) % (2*pi) # radians 
        new_traj = (2*slope - traj) % (2*pi) # radians
        # print('old and new traj in rads:', traj, new_traj)
        velocity = (ball_state[2]**2 + ball_state[3]**2)**.5
        # velocity *= .9 # dampener
        ball_state[2] = velocity * sin(new_traj) 
        ball_state[3] = velocity * cos(new_traj) * -1
        # print('v after bounce', (ball_state[2]**2 + ball_state[3]**2)**.5)
        # print('xv =', ball_state[2])
        # print('yv =', ball_state[3])
        # print('x =', ball_state[0], 'y =', ball_state[1], 'ys[x_int]=', ys[x_int])
        # print('frame', frame)
        wave_yv = ys[x_int] - prev_ys[x_int] # treating it as a transverse wave
        # print('ys[x_int], prev_ys[x_int]', ys[x_int], prev_ys[x_int])
        # print('wave_yv', wave_yv)
        ball_state[3] += wave_yv * 30
    ball_state[3] -= 10.0 # gravity
    ball_state[2:4] *= .999 # air friction    
    # print(ball_state[3]) # y_vel 
    
    contents[1].set_data( [ball_state[0]], [ball_state[1]] )     
    return contents  # a list so doesn't need comma

def animate2b():
    waves = []
    for w in range(numWaves):
        wavelength = round( (size/(2*pi)) / random.choice([1,2,3,4]) )
        amplitude = round(size * random.uniform(0.02, 0.19)) * .5
        offset = round(size * random.random())
        speed = random.randrange(10,30) # bigger is slower
        # amplitude = 0 # flat
        # speed = 1000000 # stationary
        waves.append((wavelength, amplitude, offset, speed))      
    fig, subplot = plt.subplots() # creates a figure and a subplot
    xs = np.arange(size)
    line, = subplot.plot(xs, xs*0) # flat line as placeholder
    ball_state = np.asarray([size/2, size, random.randrange(-500,500), 0], dtype=float) # x1, y1, vx1, vy1
    ball, = plt.plot([],[],'ro')
    contents = [line, ball]    
    ys = np.zeros(size)
    plt.axis([0, size, 0, size])
    plt.title("Sales Projections")
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani = animation.FuncAnimation(fig, updatefig2b, frames=10000, fargs=(contents, size, waves, ball_state, ys), blit=True, interval=1, repeat=False)
    # ani.save('sales_projections.mp4', writer=writer) # doesn't work
    plt.show()


animate2b()

