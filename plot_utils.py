import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(height, width, setup_dict, obstacles, trajectory, thrusts, gate_width=.01):
    """
    Plot quadrotor trajectory with obstacles and thrust vectors.

    Parameters:
    -----------
    height : float
        Canvas height
    width : float
        Canvas width
    setup_dict : dict
        Numerical parameters with keys: m, g, r, I, T
        - m: mass
        - g: gravity
        - r: length (rotor arm length)
        - I: moment of inertia
        - T: time horizon
    obstacles : list of dict
        Gate-shaped obstacles, each with keys:
        - x: x-coordinate of obstacle
        - uh: upper part height
        - bh: bottom part height
        Gap height = height - uh - bh
    trajectory : ndarray
        Quadrotor trajectory in shape (time, 3) with [q1, q2, q3]
        - q1: x position
        - q2: y position
        - q3: angle
    thrusts : ndarray
        Thrust trajectory in shape (time, 2) with [u1, u2]
        - u1, u2: thrust forces at left and right rotors

    Returns:
    --------
    None (displays plot)
    """
    r = setup_dict['r']

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot canvas boundary
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.plot([0, width, width, 0, 0], [0, 0, height, height, 0], 'k-', linewidth=2)

    # Draw obstacles
    for obs in obstacles:
        x = obs['x']
        uh = obs['uh']
        bh = obs['bh']

        # Draw lower obstacle
        ax.fill_between([x - gate_width, x + gate_width], 0, bh, color='gray', alpha=0.5)

        # Draw upper obstacle
        ax.fill_between([x - gate_width, x + gate_width], height - uh, height, color='gray', alpha=0.5)

    # Plot trajectory and thrust
    for i, (state, thrust) in enumerate(zip(trajectory, thrusts)):
        q1t, q2t, q3t = state
        u1, u2 = thrust

        # Calculate rotor endpoints based on angle q3
        c3t = np.cos(q3t)
        s3t = np.sin(q3t)
        endpoints = np.array([
            [q1t - r * c3t, q2t - r * s3t],
            [q1t + r * c3t, q2t + r * s3t]
        ])

        # Plot quadrotor body and center
        ax.scatter(q1t, q2t, c='b', s=20)
        ax.plot(*endpoints.T, c='b', linewidth=1)

        # Plot thrust vectors
        d = endpoints[1] - endpoints[0]
        v = np.array([-d[1], d[0]]) / 100

        for uj, ei in zip([u1, u2], endpoints):
            force_start = ei
            force_end = ei + v * uj
            ax.plot([force_start[0], force_end[0]],
                   [force_start[1], force_end[1]],
                   c='r', linewidth=1, alpha=0.6)

    ax.set_xlabel('x (q1)')
    ax.set_ylabel('y (q2)')
    ax.set_title('Quadrotor Trajectory with Obstacles')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
