#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 18:03:52 2021

@author: mhutson
"""

from math import pi, sin, cos
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from tqdm import tqdm
import cProfile

seed = random.randint(0,1000)
seed = 426
# seed = 639
print('\nseed =', seed)
random.seed(seed)

size = 1000
horizon = round(.4*size)
radius = 5
numWaves = 4


# def x_updateGrid(grid, size, frame):
#     frame = (frame + 1) % size
#     grid = np.zeros((size, size), dtype=int)
#     for x in range(size):
#         grid[round(-sin(x / wavelength + frame/size*50) * amplitude + horizon), x] = 1       
#     return grid


# def updateGrid(grid, size, waves, frame):
#     frame = (frame + 1) % size
#     grid = np.zeros((size, size), dtype=int)
#     y_lists = []
#     for w in range(len(waves)): # w = (wavelength, amplitude, offset, speed)
#         # speed = random.randrange(-30,30)
#         ys = [0] * size
#         for x in range(size):     
#             ys[x] = round(-sin(x / waves[w][0] + waves[w][2] + frame*[1,-1][w%2]/waves[w][3]) * waves[w][1] + horizon)
#             # ys[x] = round(-sin(x / waves[w][0] + waves[w][2] + frame/waves[w][3]) * waves[w][1] + horizon)
#         y_lists.append(ys)
#     ys = [0] * size                         
#     for x in range(size):
#         for w in range(numWaves):
#             ys[x] += y_lists[w][x]           
#         ys[x] = round(ys[x]/numWaves) * 2 - horizon
#         # ys[x] = round( ys[x] - (horizon * (numWaves-1)) )
#     for x in range(size):   
#         grid[ys[x], x] = 1              
#     return grid
# def updatefig1(frame, im, grid, size, waves):     
#     grid = updateGrid(grid, size, waves, frame)    
#     im.set_data(grid)
#     return im, # comma returns it as a tuple
# def animate1():
#     waves = []
#     for w in range(numWaves):
#         wavelength = round( (size/(2*pi)) / random.choice([1,2,3,4]) )
#         amplitude = round(size * random.uniform(0.02, 0.19))
#         offset = round(size * random.random())
#         speed = random.randrange(10,30) # bigger is slower
#         waves.append((wavelength, amplitude, offset, speed))    
#     grid = np.zeros((size, size), dtype=int)
#     grid[0,0]=1 # not sure why this is needed but it is
#     fig = plt.figure()
#     im = plt.imshow(grid, animated=True)
#     ani = animation.FuncAnimation(fig, updatefig1, frames=1000, fargs=(im, grid, size, waves,), blit=True, interval=1, repeat=True)
#     # ani.save("waves_movie.mp4")
#     # animation.FuncAnimation(fig, updatefig, frames=1000, fargs=(im, grid, size, waves,), blit=True, interval=1, repeat=True)
#     plt.show()
    
    
    
# def updatefig2(frame, line, size, waves):     
#     xs = np.arange(0,size,1)
#     y_arrays = np.zeros([numWaves, size])
#     for w in range(numWaves): 
#         # y_arrays[w] = -np.sin(xs / wavelength + offset) * amplitude + horizon # round?   
#         y_arrays[w] = -np.sin(xs / waves[w][0] + waves[w][2] + frame*[1,-1][w%2]/waves[w][3]) * waves[w][1] # + horizon       
#     # ys = np.sum(y_arrays, axis=0) - (horizon * (numWaves-1)) # sum
#     # ys = np.sum(y_arrays, axis=0) / numWaves # mean
#     # ys = (np.sum(y_arrays, axis=0) - (horizon*(numWaves))) * 2 / numWaves + horizon # mean, amped
#     ys = np.sum(y_arrays, axis=0) / numWaves * 2 + horizon # mean, amped
#     line.set_ydata(ys)
#     return line,  
# def animate2():
#     waves = []
#     for w in range(numWaves):
#         wavelength = round( (size/(2*pi)) / random.choice([1,2,3,4]) )
#         amplitude = round(size * random.uniform(0.02, 0.19))
#         offset = round(size * random.random())
#         speed = random.randrange(10,30) # bigger is slower
#         waves.append((wavelength, amplitude, offset, speed))      
#     fig, ax = plt.subplots() # creates a figure and a subplot
#     xs = np.arange(0,size,1)
#     line, = ax.plot(xs, xs*0) # flat line
    
#     ani = animation.FuncAnimation(fig, updatefig2, frames=1000, fargs=(line, size, waves,), blit=True, interval=1, repeat=False)
#     plt.axis([0, size, size, 0])
#     plt.show()
    
    
    
def updatefig2b(frame, contents, size, waves, ball_state, ys):     
    # print(ball_state[1], ys[0])
    xs = np.arange(0,size,1)
    y_arrays = np.zeros([numWaves, size])
    for w in range(numWaves):  # wavelength, amplitude, offset, speed
        y_arrays[w] = -np.sin(xs / waves[w][0] + waves[w][2] + frame*[1,-1][w%2]/waves[w][3]) * waves[w][1] # + horizon       
    # prev_ys = ys[:]
    prev_ys = np.copy(ys)
    ys[:] = np.sum(y_arrays, axis=0) / numWaves * 2 + horizon # mean, amped
    # print(ys[0], prev_ys[0])
    # ys = np.full(size, horizon, dtype=float) # flat ground
    # ys = size-xs[:] # slanted ground
    contents[0].set_ydata(ys)
    
    ball_state[0] = (ball_state[0] + 1/30 * ball_state[2]) % size
    ball_state[1] = (ball_state[1] + 1/30 * ball_state[3])   
    # x_int = int(round(ball_state[0]))
    x_int = int(ball_state[0])
    ball_state[1] = max(ball_state[1], ys[x_int]+radius)
    # print(round(ball_state[0]), round(ball_state[1]), round(ball_state[2]), round(ball_state[3]))
    # if 720 < frame < 800: 
    #     print(frame, round(ball_state[1]), ys[int(round(ball_state[0]))] )
    if ball_state[1]-radius <= ys[x_int] and ball_state[3] < 0: # ball_state[4] == 1:
        # print('bounce')
        # print('\nv before bounce', (ball_state[2]**2 + ball_state[3]**2)**.5)
        # print('xv =', ball_state[2])
        # print('yv =', ball_state[3])
        # print('y =', ball_state[1])
        # ball_state[4] = 0 # not airborne
        # ball_state[5] = 0 # reset airtime
        slope = (ys[(x_int+1)%size] - ys[(x_int-1)%size]) / ((x_int+1)%size - (x_int-1)%size) # ratio
        # print('slope ratio', slope)
        slope = (np.arctan(slope) + pi/2) % (2*pi) # radians
        # print('slope rads', slope)
        traj = ball_state[3]/(ball_state[2]+.000001) # ratio
        # print('traj ratio', traj)
        flip = pi if ball_state[2]<0 else 0
        traj = (np.arctan(traj) + pi/2 + flip) % (2*pi) # radians 
        new_traj = (2*slope - traj) % (2*pi) # radians
        # print('old and new traj in rads:', traj, new_traj)
        velocity = (ball_state[2]**2 + ball_state[3]**2)**.5
        # velocity *= .9 # dampener
        # ball_state[2] = velocity * cos(new_traj) # maybe
        # ball_state[3] = velocity * sin(new_traj) # maybe
        ball_state[2] = velocity * sin(new_traj) # maybe
        ball_state[3] = velocity * cos(new_traj) * -1# maybe
        # print('v after bounce', (ball_state[2]**2 + ball_state[3]**2)**.5)
        # print('xv =', ball_state[2])
        # print('yv =', ball_state[3])
        # print('x =', ball_state[0], 'y =', ball_state[1], 'ys[x_int]=', ys[x_int])
        # print('frame', frame)
        wave_yv = ys[x_int] - prev_ys[x_int]
        # print('ys[x_int], prev_ys[x_int]', ys[x_int], prev_ys[x_int])
        # print('wave_yv', wave_yv)
        ball_state[3] += wave_yv * 30
    # else:
    #     ball_state[4] = 1 # airborne
    #     ball_state[5] += 1 # increase airtime 
    # ball_state[3] -= ball_state[5] / 10.0 # gravity # fix this?
    # ball_state[3] -= frame / 10.0 # gravity
    ball_state[3] -= 10.0 # gravity
    ball_state[2:4] *= .999 # air friction
    
    # print(ball_state[3]) # y_vel
    # print('airtime', ball_state[5])
    # contents[1].set_data([frame%size], [ys[frame%size]])  
    contents[1].set_data( [ball_state[0]], [ball_state[1]] )  
    
    # if ball_state[1] < 0:
    #     print('too high')
    #     frame = 1000 # doesn't stop animation
    return contents  # list so don't need comma

def animate2b():
    waves = []
    for w in range(numWaves):
        wavelength = round( (size/(2*pi)) / random.choice([1,2,3,4]) )
        amplitude = round(size * random.uniform(0.02, 0.19)) * .5
        offset = round(size * random.random())
        speed = random.randrange(10,30) # bigger is slower
        # amplitude = 0
        # speed = 50
        waves.append((wavelength, amplitude, offset, speed))      
    fig, subplot = plt.subplots() # creates a figure and a subplot
    # xs = np.arange(0,size,1)
    xs = np.arange(size)
    line, = subplot.plot(xs, xs*0) # flat line
    ball_state = np.asarray([size/2, size, random.randrange(-500,500), 0, 1, 0], dtype=float) # x1, y1, vx1, vy1, airborne, airtime
    ball, = plt.plot([],[],'ro')
    contents = [line, ball]    
    ys = np.zeros(size)
    plt.axis([0, size, 0, size])
    plt.title("Sales Projections")
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # writer = animation.FFMpegFileWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani = animation.FuncAnimation(fig, updatefig2b, frames=10000, fargs=(contents, size, waves, ball_state, ys), blit=True, interval=1, repeat=False)
    # ani.save('sales_projections.mp4', writer=writer)
    # ani.save('sales_projections.mp4')
    plt.show()



# def plot1(gridplot=False, lineplot=True):
    
#     if gridplot: grid = np.zeros((size, size), dtype=int)
     
#     # for x in range(size):
#     #     grid[round(-sin(x / wavelength) * amplitude + horizon), x] = 1
        
#     y_lists = []
#     for w in range(numWaves):
#         ys = [0] * size
#         wavelength = round( (size/(2*pi)) / random.choice([1,2,3,4]) )
#         amplitude = round(size * random.uniform(0.02, 0.19))
#         offset = round(size * random.random())
#         for x in range(size):     
#             ys[x] = round(-sin(x / wavelength + offset) * amplitude + horizon)  
#         y_lists.append(ys)
#     ys = [0] * size                
         
#     for x in range(size):
#         for w in range(numWaves):
#             ys[x] += y_lists[w][x]           
#         # ys[x] = round(ys[x]/numWaves) # mean
#         ys[x] = round( ys[x] - (horizon * (numWaves-1)) ) # sum
        
#     ''' plot as grid: '''
#     if gridplot:
#         for x in range(size):   
#             grid[ys[x], x] = 1
#         plt.imshow(grid)
    
#     ''' plot as line: '''
#     if lineplot:
#         xs = [i for i in range(size)]
#         # ys = [size-i for i in ys]
#         plt.plot(xs, ys)
#         plt.axis([0, size, size, 0])
#         plt.show()


# def plot2():    
#     # xs = np.linspace(0,size-1,size)
#     xs = np.arange(0,size,1)
#     y_arrays = np.zeros([numWaves, size])
#     for w in range(numWaves):
#         wavelength = round( (size/(2*pi)) / random.choice([1,2,3,4]) )
#         amplitude = round(size * random.uniform(0.02, 0.19))
#         offset = round(size * random.random())    
#         y_arrays[w] = -np.sin(xs / wavelength + offset) * amplitude + horizon # round?   
#     ys = np.sum(y_arrays, axis=0) - (horizon * (numWaves-1))
#     plt.plot(xs, ys)
#     plt.axis([0, size, size, 0])
#     plt.show()
        
# animate()
animate2b()
# cProfile.run('plot()')
# plot(0,1)
# plot2()

# import time
# start = time.time() 
# plot1(0,1)
# end = time.time()
# print('line', end-start) 
# # start = time.time() 
# # plot1(1,0)
# # end = time.time()
# # print('grid', end-start)
# start = time.time() 
# plot1(0,1)
# end = time.time()
# print('line', end-start) # fastest. about .005
# start = time.time() 
# plot2()
# end = time.time()
# print('plot2', end-start)

