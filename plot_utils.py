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
