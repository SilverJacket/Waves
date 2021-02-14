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
# import matplotlib.patches as patches
# from scipy.misc import derivative


seed = random.randint(0,1000)
# seed = 460
print('\nseed =', seed)
random.seed(seed)
 
width = 2000
size = (width, int(width/2))
horizon = round(.4*size[1])
radius = 6
numWaves = 4
frames, max_frames = 0, 2000 # 5000
time_buffer_factor = 10 # 10
force, strength = 0, 100 # 100
gravity = 50 # 10
speed = 1/30 # 1/30
dampener = .99 # .99
air_friction = .98 # .999 or .98
score, right_pressed, left_pressed, paused = 0, False, False, True

# def init():
#     # line, = subplot.plot(xs, xs*0) # flat line as placeholder
#     # return line,
#     return []
    
def updatefig(frame, contents, size, waves, ball_state, f):   
    global frames, paused
    # print('frame', frame, 'frames', frames)
    if not paused:
        frames += 1
        prev_f = np.copy(f)
        f[0] = '0' # start building the wave function as a string
        # for w in range(numWaves+1):  # wavelength, amplitude, offset, speed
        for w in range(numWaves):
            # f[0] += "+ -__import__('numpy').sin(x/{} + {}) * {}".format(waves[w][0], waves[w][2] + frame*[1,-1][w%2]/waves[w][3], waves[w][1]) # add each sin wave
            f[0] += f"+ -__import__('numpy').sin(x/{waves[w][0]} + {waves[w][2] + frame*[1,-1][w%2]/waves[w][3]}) * {waves[w][1]}" # add each sin wave
        # f[0] = "(" + f[0] + ") / {} * 2 + {}".format(numWaves, horizon)     # find the average, amplify the result, and move it up
        f[0] = "(" + f[0] + f") / {numWaves} * 2 + {horizon}"               # find the average, amplify the result, and move it up
        contents[0].set_ydata(eval(f[0], {'x': np.arange(0,size[0],1)}))    # evaluate the final wave function        
        # ft = np.array([ "-__import__('numpy').sin(x/{} + {}) * {}".format(waves[numWaves][0], waves[numWaves][2] + frame*[1,-1][w%2]/waves[numWaves][3], waves[numWaves][1]) ]) # sin wave for target 
        ft = np.array([f"-__import__('numpy').sin(x/{waves[numWaves][0]} + {waves[numWaves][2] + frame*[1,-1][w%2]/waves[numWaves][3]}) * {waves[numWaves][1]}"]) # sin wave for target
        
        print_status = False   
        # prev_ball_state = ball_state[:]
        ball_state[0] = (ball_state[0] + speed * ball_state[2]) % size[0]   # update ball's x position
        ball_state[1] = (ball_state[1] + speed * ball_state[3])             # update ball's y position
        line_y = eval(f[0], {'x':ball_state[0]})    # find the wave's y position under the ball
        if ball_state[1]-radius <= line_y + 0:      # if the ball hits the wave
            ball_state[1] = line_y + radius # don't let the ball go underground
            # ''' fix velocity?: '''
            # ball_state[2], ball_state[3] = ball_state[0]-prev_ball_state[0], ball_state[1]-prev_ball_state[1]      
            # if ball_state[1]-radius <= ys[x_int] and ball_state[3] < 0: # ball hits wave
            if print_status: print('bounce')
            if print_status: print('\nv before bounce', (ball_state[2]**2 + ball_state[3]**2)**.5)
            if print_status: print('xv =', ball_state[2])
            if print_status: print('yv =', ball_state[3])
            if print_status: print('y =', ball_state[1])
            
            ''' get wave's slope under ball '''
            y1, y2 = eval(f[0], {'x':ball_state[0]-1}), eval(f[0], {'x':ball_state[0]+1})
            slope = (y2-y1) / ((ball_state[0]+1)%size[0] - (ball_state[0]-1)%size[0]) # get the wave's slope under the ball           
            # def w(x):
            #     return eval(f[0]);
            # slope2 = derivative(w, ball_state[0])                 # slower by ~10x, or 10e-4 sec
            # print(w(ball_state[0]), '\n', slope, '\n', slope2)    # equivalent 
            # if print_status: print('slope ratio', slope)
            slope = (np.arctan(slope) + pi/2) % (2*pi)              # radians
            # if print_status: print('slope rads', slope)
            
            ''' change ball's trajectory '''
            traj = ball_state[3]/(ball_state[2]+.000001) # ball's trajectory as ratio
            # if print_status: print('traj ratio', traj)
            flip = pi if ball_state[2]<0 else 0
            traj = (np.arctan(traj) + pi/2 + flip) % (2*pi) # radians 
            new_traj = (2*slope - traj) % (2*pi) # radians
            if print_status: print('old and new traj in rads:', traj, new_traj)
            velocity = (ball_state[2]**2 + ball_state[3]**2)**.5
            velocity *= dampener 
            ball_state[2] = velocity * sin(new_traj)        # calculate x velocity after bounce
            ball_state[3] = velocity * cos(new_traj) * -1   # calculate y velocity after bounce
            if print_status: print('v after bounce', (ball_state[2]**2 + ball_state[3]**2)**.5)
            if print_status: print('xv =', ball_state[2])
            if print_status: print('yv =', ball_state[3])
            if print_status: print('frame', frame)
            
            ''' further change ball's velocity '''
            wave_yv = line_y - eval(prev_f[0], {'x': ball_state[0]})    # wave's y velocity under ball; treating it as a transverse wave
            if print_status: print('wave_yv', wave_yv)           
            ball_state[3] += wave_yv / speed    # wave vertically boosts or dampens ball   
            # ball_state[2] += force
            ball_state[2] += force * sin(slope)         # maybe # left and right arrows push ball parallel to slope
            ball_state[3] += force * cos(slope) * -1    # maybe # left and right arrows push ball parallel to slope
        contents[1].set_data( [ball_state[0]], [ball_state[1]] )  
        ball_state[3] -= gravity 
        ball_state[2:4] *= air_friction # air friction .999 good
        
        right_string = "Right Pressed" if right_pressed else ""
        left_string = "Left Pressed   " if left_pressed else "                      "
        contents[3].set_text(left_string + right_string)
        
        global score
        contents[5].set_text("Score: " + str(score))
        contents[2].set_radius(100) 
        contents[2].set_edgecolor('green')
        new_x = - (frame/1) % size[0]
        new_y = size[1]*.6 + eval(ft[0], {'x':(contents[2].center[0]+size[0]/2)%size[0]})
        contents[2].center = (new_x, new_y) # new_center
        # if contents[5].contains_point( (ball_state[0],ball_state[1]), radius=radius): doesn't work
        if ((ball_state[0]-contents[2].center[0])**2 + (ball_state[1]-contents[2].center[1])**2)**.5 < radius + contents[2].radius \
            and frames < max_frames:   # if ball inside target and game not over
                contents[2].set_edgecolor('red')
                score += 1
                
        if frames < max_frames:
            contents[4].set_text("Time: " + str(max_frames - frames - 1))
        else: # game over, man
            contents[6].set_text("Score: " + str(score))
            contents[2].set_edgecolor('green')
            
    else:
        contents[6].set_text("Welcome to Rollerball!")
        contents[4].set_text("Press spacebar to begin.")
    
    return contents  # contents is a list so doesn't need comma

def play():
    waves = []
    # force = 0
    for w in range(numWaves):
        wavelength = round( (size[0]/(2*pi)) / random.choice([1,2,3,4]) )
        amplitude = round(size[1] * random.uniform(0.02, 0.19)) * 1 # .5
        offset = round(size[0] * random.random())
        # wave_speed = random.randrange(10,30) # bigger is slower
        wave_speed = 500 # slow
        # wave_speed = 1000000 # stationary
        # amplitude = 0 # flat
        waves.append((wavelength, amplitude, offset, wave_speed))
    waves.append((round((size[0]/(2*pi))/3), round(size[1]*.15), 0, wave_speed)) # for the target

    fig, subplot = plt.subplots(figsize=(10,5)) # creates a figure and a subplot
    subplot.set_aspect('equal')
    xs = np.arange(size[0])
    line, = subplot.plot(xs, xs*0) # flat line as placeholder
    ball_state = np.asarray([size[0]/2, size[1], random.randrange(-500,500), 0], dtype=float) # x1, y1, vx1, vy1
    ball, = plt.plot([],[],'ro')
    key_text = subplot.text(20, 20, '', fontsize=10)
    # right_key_text = subplot.text(width-100, 20, '', fontsize=10)
    time_text = subplot.text(20, size[1]-50, '', fontsize=10)
    score_text = subplot.text(20, size[1]-100, '', fontsize=10)
    # center_text = subplot.text(size[0]/2, size[1]/2, "Welcome to Rollerball\nPress space to begin.", ha='center', c='red', fontsize=40)
    center_text = subplot.text(size[0]/2, size[1]/2, '', ha='center', c='red', fontsize=40)
    target = plt.Circle((random.randrange(size[0]), .6*size[1]), radius=0, ec='green', fill=False) # target 
    # print(dir(target))
    # print(dir(score_text))
    plt.gca().add_patch(target)
    contents = [line, ball, target, key_text, time_text, score_text, center_text] 
    # ys = np.zeros(size[0])
    # f = '0'
    f = np.array(['0'], dtype=object)
    plt.axis([0, size[0], 0, size[1]])   
    subplot.get_xaxis().set_ticks([])
    subplot.get_yaxis().set_ticks([])  
    plt.xlabel("Use left and right keys to hit the green target.")
    plt.title("Rollerball")
        
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    global max_frames
    ani = animation.FuncAnimation(fig, updatefig, frames=max_frames*time_buffer_factor, fargs=(contents, size, waves, ball_state, f), blit=True, interval=30, repeat=False)
    ani.event_source.stop()
    # ani.save('sales_projections.mp4', writer=writer) # doesn't work
       
    def on_key1(event):
        global force, right_pressed, left_pressed, paused
        # nonlocal ani
        # strength = 100
        if event.key == 'right':
            # print('you pressed', event.key, event.xdata, event.ydata)
            force = strength
            right_pressed = True
        if event.key == 'left':
            # print('you pressed', event.key, event.xdata, event.ydata)
            force = -strength
            left_pressed = True
        if event.key == ' ':
            # print('space pressed')
            # if paused: ani.event_source.start() # doesn't work
            # else: ani.event_source.stop()
            paused = [True, False][paused]
            center_text.set_text("")
    cid = fig.canvas.mpl_connect('key_press_event', on_key1)
    def on_key_release1(event):
        global force, right_pressed, left_pressed
        if event.key == 'right':
            # print('you released', event.key, event.xdata, event.ydata)
            right_pressed = False
            force = 0
        if event.key ==  'left':
            # print('you released', event.key, event.xdata, event.ydata)
            left_pressed = False
            force = 0
    cid = fig.canvas.mpl_connect('key_release_event', on_key_release1)    
    
    plt.show()


play()

