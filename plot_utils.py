import numpy as np
import matplotlib.pyplot as plt


def trajectory_decomposition(x0, xN, obstacles, r, tol=1e-9):
    """
    Decompose the x-axis trajectory between x0 and xN into segments around obstacle gates.

    The segmentation follows the pattern:
      [x0, obs_x0 - r], [obs_x0 - r, obs_x0 + r], [obs_x0 + r, obs_x1 - r], ..., [obs_xN + r, xN]

    Parameters
    ----------
    x0 : float
        Start x-coordinate.
    xN : float
        Final x-coordinate.
    obstacles : list
        List of obstacles. Each obstacle can be either a dict containing key 'x'
        (as used in plot_trajectory) or a numeric x-coordinate directly.
    r : float
        Quadrotor radius. Each gate interval is centered at obs_x and has half-width r.
    tol : float
        Small tolerance used to drop/merge vanishing intervals.

    Returns
    -------
    segments : list
        Ordered list of segment dictionaries in the direction from x0 to xN. Each
        dictionary has keys:
          - 'x_start': segment start coordinate
          - 'x_end': segment end coordinate
          - 'kind': 'free' for free-space segments, 'gate' for through-gate segments

    Notes
    -----
    - Obstacles whose gate intervals [obs_x - r, obs_x + r] do not intersect the
      interval [min(x0, xN), max(x0, xN)] are ignored.
    - Overlapping/adjacent gate intervals are merged into a single 'gate' segment.

    Example
    -------
    >>> trajectory_decomposition(0.0, 10.0, [{'x':3.0}, {'x':7.0}], 0.2)
    [ {'x_start':0.0, 'x_end':2.8, 'kind':'free'},
      {'x_start':2.8, 'x_end':3.2, 'kind':'gate'},
      {'x_start':3.2, 'x_end':6.8, 'kind':'free'},
      {'x_start':6.8, 'x_end':7.2, 'kind':'gate'},
      {'x_start':7.2, 'x_end':10.0, 'kind':'free'} ]
    """

    # Extract obstacle x coordinates (support dicts or numeric list)
    xs = []
    if not obstacles:
        return [{'x_start': float(x0), 'x_end': float(xN), 'kind': 'free'}]

    for obs in obstacles:
        if isinstance(obs, dict):
            if 'x' in obs:
                xs.append(float(obs['x']))
        else:
            # assume numeric
            try:
                xs.append(float(obs))
            except Exception:
                # skip invalid entries
                continue

    # If there are no valid obstacles, return the whole range as one free segment
    if len(xs) == 0:
        return [{'x_start': float(x0), 'x_end': float(xN), 'kind': 'free'}]

    # Work in a transformed coordinate so that we always move in increasing order
    direction = 1.0 if xN >= x0 else -1.0
    def T(x):
        return direction * x

    tx0 = T(x0)
    txN = T(xN)

    # Filter obstacles whose gate intervals intersect the domain [tx0, txN]
    txs = []
    for x in xs:
        tx = T(x)
        if tx + r >= tx0 - tol and tx - r <= txN + tol:
            txs.append(tx)

    if len(txs) == 0:
        return [{'x_start': float(x0), 'x_end': float(xN), 'kind': 'free'}]

    txs.sort()

    # Build gate intervals clipped to [tx0, txN]
    intervals = []
    for tx in txs:
        l = max(tx - r, tx0)
        rr = min(tx + r, txN)
        if rr - l > tol:
            intervals.append((l, rr))

    # Merge overlapping/adjacent intervals
    merged = []
    for l, rr in intervals:
        if not merged:
            merged.append([l, rr])
        else:
            prev_l, prev_r = merged[-1]
            if l <= prev_r + tol:
                # overlap: extend
                merged[-1][1] = max(prev_r, rr)
            else:
                merged.append([l, rr])

    # Build final segments between tx0 and txN
    segments = []
    cur = tx0
    for l, rr in merged:
        if l - cur > tol:
            segments.append({'x_start': direction * cur, 'x_end': direction * l, 'kind': 'free'})
        segments.append({'x_start': direction * l, 'x_end': direction * rr, 'kind': 'gate'})
        cur = rr

    if txN - cur > tol:
        segments.append({'x_start': direction * cur, 'x_end': direction * txN, 'kind': 'free'})

    # If there are no segments due to clipping, return the full range
    if len(segments) == 0:
        return [{'x_start': float(x0), 'x_end': float(xN), 'kind': 'free'}]

    return segments


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


def animate_uq(u, q, dt=0.01, interval=None, draw=True, return_anim=False, save_path=None, figsize=(10,6), title=None, angle_col=-1):
    """
    Create an inline animation of control inputs u1,u2 and angle q in a Jupyter notebook.

    Parameters
    ----------
    u1, u2, q : array-like
        1D sequences of equal length representing control inputs and angle over time.
    dt : float
        Time step between samples (seconds). Used to create the time axis if interval
        is not provided. Default 0.01.
    interval : float or None
        Frame interval in milliseconds. If None, computed from dt as dt*1000.
    draw : bool
        If True, the function will try to display the animation inline (using
        HTML embedding). If False the animation object is returned (or saved if
        save_path is provided).
    return_anim : bool
        If True the matplotlib.animation.FuncAnimation object is also returned.
    save_path : str or None
        If provided, attempts to save the animation to this file (mp4 or gif
        depending on extension). Requires appropriate writers installed.
    figsize : tuple
        Figure size for the animation.
    title : str or None
        Optional title for the figure.

    Returns
    -------
    If return_anim is True: (anim, html_or_none)
    Else: html_or_none or None

    Notes
    -----
    This utility is intended for use in Jupyter notebooks. It attempts to use
    FuncAnimation.to_jshtml() for inline display; if that fails and save_path is
    provided it will try to save to disk (ffmpeg/gifsicle may be required).
    """
    import numpy as _np
    import matplotlib.pyplot as _plt
    from matplotlib import animation as _animation

    u_arr = _np.asarray(u)
    q_arr = _np.asarray(q)

    # Parse control array u (support shapes (N,2) or (2,N) or larger where last two cols are controls)
    if u_arr.ndim != 2:
        raise ValueError('u must be a 2D array with two control signals per sample')

    if u_arr.shape[1] == 2:
        u1 = u_arr[:, 0]
        u2 = u_arr[:, 1]
    elif u_arr.shape[0] == 2:
        u1 = u_arr[0, :]
        u2 = u_arr[1, :]
    elif u_arr.shape[1] >= 2:
        # fallback: use last two columns
        u1 = u_arr[:, -2]
        u2 = u_arr[:, -1]
    else:
        raise ValueError('Control array shape not recognized; expected one axis of length 2')

    # Parse angle q: accept 1D or 2D arrays; if 2D select a column (angle_col)
    if q_arr.ndim == 1:
        q1d = q_arr
    elif q_arr.ndim == 2:
        col = int(angle_col)
        if col < 0:
            col = q_arr.shape[1] + col
        if col < 0 or col >= q_arr.shape[1]:
            col = q_arr.shape[1] - 1
        q1d = q_arr[:, col]
    else:
        q1d = q_arr.ravel()

    u1 = _np.asarray(u1)
    u2 = _np.asarray(u2)
    q = _np.asarray(q1d)

    N = len(u1)
    if len(u2) != N or len(q) != N:
        raise ValueError("u1, u2, q must have the same length")

    if interval is None:
        interval = float(dt) * 1000.0

    t = _np.arange(N) * float(dt)

    fig, (ax0, ax1) = _plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Top: controls
    ax0.set_title('Controls (u1, u2)')
    ax0.set_ylabel('Thrust')
    ax0.grid(True, alpha=0.3)
    line_u1_full, = ax0.plot(t, u1, color='tab:blue', alpha=0.2)
    line_u2_full, = ax0.plot(t, u2, color='tab:orange', alpha=0.2)
    line_u1, = ax0.plot([], [], color='tab:blue')
    line_u2, = ax0.plot([], [], color='tab:orange')
    marker_u1, = ax0.plot([], [], 'o', color='tab:blue', markersize=6)
    marker_u2, = ax0.plot([], [], 'o', color='tab:orange', markersize=6)
    vline0 = ax0.axvline(0, color='k', linestyle='--', alpha=0.6)

    # Bottom: angle q
    ax1.set_title('Angle q')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('q (rad)')
    ax1.grid(True, alpha=0.3)
    line_q_full, = ax1.plot(t, q, color='tab:green', alpha=0.2)
    line_q, = ax1.plot([], [], color='tab:green')
    marker_q, = ax1.plot([], [], 'o', color='tab:green', markersize=6)
    vline1 = ax1.axvline(0, color='k', linestyle='--', alpha=0.6)

    if title is not None:
        fig.suptitle(title)

    ax0.set_xlim(t[0], t[-1])
    # Set reasonable y-limits with small padding
    def _pad_limits(arr, pad=0.1):
        mn, mx = float(_np.min(arr)), float(_np.max(arr))
        if mx == mn:
            mx = mn + 1.0
        d = (mx - mn) * pad
        return mn - d, mx + d

    ax0.set_ylim(*_pad_limits(_np.concatenate([u1, u2])))
    ax1.set_ylim(*_pad_limits(q))

    def init():
        line_u1.set_data([], [])
        line_u2.set_data([], [])
        marker_u1.set_data([], [])
        marker_u2.set_data([], [])
        vline0.set_xdata([t[0], t[0]])
        line_q.set_data([], [])
        marker_q.set_data([], [])
        vline1.set_xdata([t[0], t[0]])
        return (line_u1, line_u2, marker_u1, marker_u2, vline0, line_q, marker_q, vline1)

    def update(i):
        ti = t[i]
        line_u1.set_data(t[:i+1], u1[:i+1])
        line_u2.set_data(t[:i+1], u2[:i+1])
        marker_u1.set_data(t[i], u1[i])
        marker_u2.set_data(t[i], u2[i])
        vline0.set_xdata(ti)

        line_q.set_data(t[:i+1], q[:i+1])
        marker_q.set_data(t[i], q[i])
        vline1.set_xdata(ti)

        return (line_u1, line_u2, marker_u1, marker_u2, vline0, line_q, marker_q, vline1)

    anim = _animation.FuncAnimation(fig, update, frames=N, init_func=init, interval=interval, blit=True)

    html_obj = None
    if save_path is not None:
        try:
            anim.save(save_path)
        except Exception:
            # ignore save errors, user can handle
            pass

    if draw:
        try:
            # Prefer JS animation for wide compatibility
            from IPython.display import HTML as _HTML, display as _display
            html_obj = _HTML(anim.to_jshtml())
            _display(html_obj)
        except Exception:
            try:
                from IPython.display import HTML as _HTML, display as _display
                html_obj = _HTML(anim.to_html5_video())
                _display(html_obj)
            except Exception:
                # Last resort: show static figure
                _plt.show()

    if return_anim:
        return anim, html_obj
    else:
        return html_obj

import numpy as _np
import matplotlib.pyplot as _plt
from matplotlib import animation as _animation

def animate_trajectory(height, width, setup_dict, obstacles, trajectory, thrusts=None, gate_width=.01,
                       dt=0.01, interval=None, draw=True, return_anim=False, save_path=None,
                       figsize=(10,8), title=None, trail=True):
    """
    Animate the quadrotor physical trajectory (positions, orientation, thrusts) over time.

    Parameters
    ----------
    height, width : float
        Canvas dimensions (same as plot_trajectory).
    setup_dict : dict
        Contains 'r' (arm length) at minimum.
    obstacles : list of dict
        Obstacles as in plot_trajectory (each with 'x','uh','bh').
    trajectory : array-like (N,3)
        Sequence of states [q1, q2, q3] where q1=x, q2=y, q3=angle.
    thrusts : array-like (N,2) or None
        Rotor thrusts [u1,u2] at each time. If None zeros are used.
    gate_width, dt, interval, draw, return_anim, save_path, figsize, title, trail : see animate_uq

    Returns
    -------
    If return_anim is True: (anim, html_or_none) else html_or_none or None
    """

    traj = _np.asarray(trajectory)
    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError('trajectory must be (N,3) or (N,2+) array with q1,q2,q3')

    N = traj.shape[0]
    if thrusts is None:
        thrusts = _np.zeros((N, 2))
    thrusts = _np.asarray(thrusts)
    if thrusts.shape[0] != N:
        raise ValueError('thrusts must have same length as trajectory')

    if interval is None:
        interval = float(dt) * 1000.0

    r = float(setup_dict.get('r', 0.2))

    t = _np.arange(N) * float(dt)

    fig, ax = _plt.subplots(figsize=figsize)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.plot([0, width, width, 0, 0], [0, 0, height, height, 0], 'k-', linewidth=2)

    # Draw static obstacles
    for obs in obstacles:
        x = obs['x']
        uh = obs['uh']
        bh = obs['bh']
        ax.fill_between([x - gate_width, x + gate_width], 0, bh, color='gray', alpha=0.5)
        ax.fill_between([x - gate_width, x + gate_width], height - uh, height, color='gray', alpha=0.5)

    # Dynamic artists
    center_scatter = ax.scatter([], [], c='b', s=50)
    body_line, = ax.plot([], [], c='b', linewidth=2)
    rotor1_line, = ax.plot([], [], c='b', linewidth=1)
    rotor2_line, = ax.plot([], [], c='b', linewidth=1)
    thrust1_line, = ax.plot([], [], c='r', linewidth=1, alpha=0.5)
    thrust2_line, = ax.plot([], [], c='r', linewidth=1, alpha=0.5)
    if trail:
        trail_line, = ax.plot([], [], c='b', linewidth=1, alpha=0.6)
    else:
        trail_line = None

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        center_scatter.set_offsets(_np.empty((0, 2)))
        body_line.set_data([], [])
        rotor1_line.set_data([], [])
        rotor2_line.set_data([], [])
        thrust1_line.set_data([], [])
        thrust2_line.set_data([], [])
        if trail_line is not None:
            trail_line.set_data([], [])
        time_text.set_text('')
        if trail_line is not None:
            return (center_scatter, body_line, rotor1_line, rotor2_line, thrust1_line, thrust2_line, trail_line, time_text)
        else:
            return (center_scatter, body_line, rotor1_line, rotor2_line, thrust1_line, thrust2_line, time_text)

    def update(i):
        q1t = float(traj[i, 0])
        q2t = float(traj[i, 1])
        q3t = float(traj[i, 2]) if traj.shape[1] > 2 else 0.0

        # rotor endpoints
        c3t = _np.cos(q3t)
        s3t = _np.sin(q3t)
        ep0 = _np.array([q1t - r * c3t, q2t - r * s3t])
        ep1 = _np.array([q1t + r * c3t, q2t + r * s3t])

        # update body and rotors
        body_line.set_data([ep0[0], ep1[0]], [ep0[1], ep1[1]])
        rotor1_line.set_data([ep0[0]], [ep0[1]])
        rotor2_line.set_data([ep1[0]], [ep1[1]])
        center_scatter.set_offsets([[q1t, q2t]])

        # thrust vectors
        d = ep1 - ep0
        v = _np.array([-d[1], d[0]]) / 20.0
        u1 = float(thrusts[i, 0])
        u2 = float(thrusts[i, 1])
        f1_end = ep0 + v * np.exp(u1/4)
        f2_end = ep1 + v * np.exp(u2/4)
        thrust1_line.set_data([ep0[0], f1_end[0]], [ep0[1], f1_end[1]])
        thrust2_line.set_data([ep1[0], f2_end[0]], [ep1[1], f2_end[1]])

        # trail
        if trail_line is not None:
            trail_line.set_data(traj[:i+1, 0], traj[:i+1, 1])

        time_text.set_text(f't={i}\n')

        if trail_line is not None:
            return (center_scatter, body_line, rotor1_line, rotor2_line, thrust1_line, thrust2_line, trail_line, time_text)
        else:
            return (center_scatter, body_line, rotor1_line, rotor2_line, thrust1_line, thrust2_line, time_text)

    anim = _animation.FuncAnimation(fig, update, frames=N, init_func=init, interval=interval, blit=True)

    html_obj = None
    if save_path is not None:
        try:
            anim.save(save_path)
        except Exception:
            pass

    if draw:
        try:
            from IPython.display import HTML as _HTML, display as _display
            html_obj = _HTML(anim.to_jshtml())
            _display(html_obj)
        except Exception:
            try:
                from IPython.display import HTML as _HTML, display as _display
                html_obj = _HTML(anim.to_html5_video())
                _display(html_obj)
            except Exception:
                _plt.show()

    if return_anim:
        return anim, html_obj
    else:
        return html_obj
