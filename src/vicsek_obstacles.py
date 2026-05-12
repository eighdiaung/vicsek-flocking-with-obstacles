# -*- coding: utf-8 -*-
"""
Created on Tue May 12 09:41:57 2026

@author: eighd
"""

"""
This code runs the Vicsek model with circular obstacles.
"""

import numpy as np
from vicsek_models import VicsekModel
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle


# Parameters to run Vicsek model
N = 50      # number of agents
L = 5.0     # length of square domain
c = 0.1     # speed of agents
R = 0.5     # interaction radius
eta = 0.1   # noise strength

# Obstacles: each row is [x_center, y_center, radius]
# Keep this as a 2D array, even for one obstacle.
obstacles = np.array([
    [0.0, 0.0, 0.45],
    [-1.25, 1.0, 0.35],
    [1.25, -1.0, 0.35],
])


def reset_simulation(event):
    global vicsek

    # Recreate the model so agents are initialized outside obstacles
    vicsek = VicsekModel(
        num_agents=N,
        domain_length=L,
        speed=c,
        intrad=float(R_slider.val),
        noise_strength=float(eta_slider.val),
        obstacles=obstacles
    )
    vicsek.reset_flag = True

    update_plot()
    fig.canvas.draw_idle()


def update_plot():
    x_positions = [agent.x for agent in vicsek.agents]
    y_positions = [agent.y for agent in vicsek.agents]

    velocities = np.array([
        [np.cos(agent.angle), np.sin(agent.angle)]
        for agent in vicsek.agents
    ]) * vicsek.speed

    scat.set_offsets(np.c_[x_positions, y_positions])
    quiv.set_offsets(np.c_[x_positions, y_positions])
    quiv.set_UVC(velocities[:, 0], velocities[:, 1])


def animate(frame, scat, quiv, eta_slider, R_slider):
    global vicsek

    eta_val = float(eta_slider.val)
    R_val = float(R_slider.val)

    vicsek.noise_strength = eta_val
    for a in vicsek.agents:
        a.noise_strength = eta_val

    vicsek.intrad = R_val

    if not getattr(vicsek, "reset_flag", False):
        vicsek.evolve()
    else:
        vicsek.reset_flag = False

    update_plot()

    return scat, quiv


# Create a Vicsek model and run the simulation
vicsek = VicsekModel(
    num_agents=N,
    domain_length=L,
    speed=c,
    intrad=R,
    noise_strength=eta,
    obstacles=obstacles
)
vicsek.reset_flag = False


# Create figure and axis
fig, ax = plt.subplots(figsize=(9, 6))
plt.subplots_adjust(bottom=0.28)

ax.set_xlim(-L / 2, L / 2)
ax.set_ylim(-L / 2, L / 2)
ax.set_aspect("equal")
ax.set_xticks([-L / 2, 0, L / 2])
ax.set_yticks([-L / 2, 0, L / 2])


# Draw obstacles
for obs in obstacles:
    circle = Circle(
        (obs[0], obs[1]),
        obs[2],
        facecolor="gray",
        edgecolor="black",
        alpha=0.5
    )
    ax.add_patch(circle)


# Initial scatter and quiver
initial_x = [agent.x for agent in vicsek.agents]
initial_y = [agent.y for agent in vicsek.agents]

scat = ax.scatter(
    initial_x,
    initial_y,
    s=50,
    color="blue",
    marker="o",
    facecolors="none"
)

initial_velocities = np.array([
    [np.cos(agent.angle), np.sin(agent.angle)]
    for agent in vicsek.agents
]) * vicsek.speed

quiv = ax.quiver(
    initial_x,
    initial_y,
    initial_velocities[:, 0],
    initial_velocities[:, 1],
    angles="uv",
    scale_units="inches",
    scale=0.75,
    color="red"
)


# Slider UI
slider_eta = fig.add_axes([0.15, 0.08, 0.7, 0.04])
eta_slider = Slider(
    ax=slider_eta,
    label="noise η",
    valmin=0.0,
    valmax=1.0,
    valinit=vicsek.noise_strength,
    valstep=0.01
)

slider_R = fig.add_axes([0.15, 0.01, 0.7, 0.04])
R_slider = Slider(
    ax=slider_R,
    label="interaction R",
    valmin=0.0,
    valmax=L,
    valinit=vicsek.intrad,
    valstep=0.01
)


# Reset button
button_ax = fig.add_axes([0.4, 0.15, 0.2, 0.05])
reset_button = Button(button_ax, "Reset", color="lightgray", hovercolor="0.9")
reset_button.on_clicked(reset_simulation)


# Title
title = ax.set_title(
    f"Vicsek model with obstacles (η={eta_slider.val:.2f}, R={R_slider.val:.2f})"
)


def on_slider_change(val):
    title.set_text(
        f"Vicsek model with obstacles (η={eta_slider.val:.2f}, R={R_slider.val:.2f})"
    )
    fig.canvas.draw_idle()


eta_slider.on_changed(on_slider_change)
R_slider.on_changed(on_slider_change)


# Animation
ani = FuncAnimation(
    fig,
    animate,
    frames=None,
    fargs=(scat, quiv, eta_slider, R_slider),
    interval=50,
    blit=False
)

plt.show()