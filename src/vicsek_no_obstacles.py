# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 11:03:33 2026

@author: eighd
"""

"""
This code runs the Vicsek model without obstacles
"""

import numpy as np
from vicsek_models import VicsekModel
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button


# Parameters to run Vicsek model
N = 50;    # number of agents
L = 5.0;    # length of square domain
c = 0.1;    # speed of agents
R = 0.5;    # interaction radius
eta = 0.1;  # noise strength


# Set up the plot for real-time animation
def animate(frame, scat, quiv, eta_slider, R_slider):  
    global vicsek
    
    eta_val = float(eta_slider.val)
    R_val = float(R_slider.val)

    vicsek.noise_strength = eta_val
    for a in vicsek.agents:
        a.noise_strength = eta_val
    vicsek.intrad = R_val

    # Only evolve if not just reset
    if not getattr(vicsek, "reset_flag", False):
        vicsek.evolve()
    else:
        vicsek.reset_flag = False  # skip just one step after reset
    
    # update scatter + quiver
    x_positions = [agent.x for agent in vicsek.agents]
    y_positions = [agent.y for agent in vicsek.agents]
    velocities = np.array([[np.cos(agent.angle), np.sin(agent.angle)] for agent in vicsek.agents]) * vicsek.speed  
    
    scat.set_offsets(np.c_[x_positions, y_positions])
    quiv.set_offsets(np.c_[x_positions, y_positions])
    quiv.set_UVC(velocities[:, 0], velocities[:, 1])
    
    return scat, quiv


# Reset function
def reset_simulation(event):
    global vicsek
    
    for i, a in enumerate(vicsek.agents):
        a.x = np.random.uniform(-vicsek.domain_length/2, vicsek.domain_length/2)
        a.y = np.random.uniform(-vicsek.domain_length/2, vicsek.domain_length/2)
        a.angle = np.random.uniform(0, 2*np.pi)
        
        # Update the internal arrays
        vicsek.headings[i] = a.angle
        vicsek.prev_headings[i] = a.angle
        vicsek.positions[i] = [a.x, a.y]
        vicsek.prev_positions[i] = [a.x, a.y]

    vicsek.reset_flag = True  # flag to skip one evolve update

    # Update scatter + quiver
    x_positions = [a.x for a in vicsek.agents]
    y_positions = [a.y for a in vicsek.agents]
    velocities = np.array([[np.cos(a.angle), np.sin(a.angle)] for a in vicsek.agents]) * vicsek.speed

    scat.set_offsets(np.c_[x_positions, y_positions])
    quiv.set_offsets(np.c_[x_positions, y_positions])
    quiv.set_UVC(velocities[:, 0], velocities[:, 1])

    fig.canvas.draw_idle()


# Create a Vicsek model and run the simulation
vicsek = VicsekModel(N, L, c, R, eta)
vicsek.reset_flag = False

# Create the figure and axis for the plot
fig, ax = plt.subplots(figsize=(9, 6))

# --- NEW: leave room at bottom for the slider ---
plt.subplots_adjust(bottom=0.28)

ax.set_xlim(-L/2, L/2)
ax.set_ylim(-L/2, L/2)
ax.set_aspect('equal')
ax.set_xticks([-L/2, 0, L/2])
ax.set_yticks([-L/2, 0, L/2])

# Create the scatter plot with initial positions
initial_x = [agent.x for agent in vicsek.agents]
initial_y = [agent.y for agent in vicsek.agents]
scat = ax.scatter(initial_x, initial_y, s=50, color='blue', marker='o', facecolors='none')

initial_velocities = np.array([[np.cos(agent.angle), np.sin(agent.angle)] for agent in vicsek.agents]) * vicsek.speed

'''
quiv = ax.quiver(initial_x, initial_y,
                 initial_velocities[:, 0], initial_velocities[:, 1],
                 angles='uv', scale_units='xy', scale=0.5, color='red')
'''
quiv = ax.quiver(
    initial_x, initial_y,
    initial_velocities[:, 0], initial_velocities[:, 1],
    angles='uv',
    scale_units='inches',   # or 'dots'
    scale=0.75,
    color='red'
)

# --- NEW: slider UI ---
slider_eta = fig.add_axes([0.15, 0.08, 0.7, 0.04])  # [left, bottom, width, height]
eta_slider = Slider(
    ax=slider_eta,
    label="noise η",
    valmin=0.0,
    valmax=1.0,
    valinit=vicsek.noise_strength,
    valstep=0.01
)

slider_R = fig.add_axes([0.15, 0.01, 0.7, 0.04])  # [left, bottom, width, height]
R_slider = Slider(
    ax=slider_R,
    label="interaction R",
    valmin=0.0,
    valmax=L,
    valinit=vicsek.intrad,
    valstep=0.01
)


button_ax = fig.add_axes([0.4, 0.15, 0.2, 0.05])  # [left, bottom, width, height]
reset_button = Button(button_ax, "Reset", color="lightgray", hovercolor="0.9")

reset_button.on_clicked(reset_simulation)

# (Optional) show current eta in the plot title, updated live
title = ax.set_title(f"Vicsek model (η={eta_slider.val:.2f}, R={R_slider.val:.2f})")

def on_slider_change(val):
    title.set_text(f"Vicsek model (η={eta_slider.val:.2f}, R={R_slider.val:.2f})")
    fig.canvas.draw_idle()

eta_slider.on_changed(on_slider_change)
R_slider.on_changed(on_slider_change)

# Set up the animation
ani = FuncAnimation(
    fig,
    animate,
    frames=None,                 # run continuously
    fargs=(scat, quiv, eta_slider, R_slider),
    interval=50,
    blit=False
)

# Display the plot
plt.show()