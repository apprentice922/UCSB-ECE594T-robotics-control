"""
Optimize a planar quadrotor trajectory using cvxpy with IPOPT (nonlinear solver).
`
This script follows the setup in optim_test.ipynb but uses the full nonlinear
discrete dynamics in quadrotor_dynamics.f_discrete and solves the resulting
nonconvex NLP with IPOPT via cvxpy: prob.solve(nlp=True, solver=cp.IPOPT).

Requirements:
- cvxpy built with IPOPT support (cyipopt installed and cvxpy from a recent
  branch that includes IPOPT interface). If IPOPT is not available, the script
  will raise an informative error.

Usage: run as a script. It will print solver status and final trajectory and
controls.
"""
import numpy as np
import cvxpy as cp
from quadrotor_dynamics import f_discrete, m, g, I, r
from plot_utils import trajectory_decomposition


def solve_quadrotor(x0 = np.array([0.25, 0.0, 0.5, 0.0, 0.0, 0.0]),
                    xT = np.array([2, 0.0, 1.5, 0.0, 0.0, 0.0]),
                    obstacles = None,
                    N=100, u_max=10.0,
                    border_x = np.array([0, 4]),
                    border_y = np.array([0,2]),
                    gate_y_margin=0.05,
                    h_min=5e-5, h_max=5e-1):
    """
    Solve the quadrotor trajectory with segmentation across obstacle gates.

    This version uses a single global control sequence (u1,u2) of length N and
    treats the time step h as a cvxpy decision variable bounded by [h_min,h_max].
    The total number of intervals is N and each segment consumes a slice of the
    global control sequence; segments are assigned roughly equal counts (at
    least 2 intervals each).
    """

    # Prepare obstacles input for decomposition: allow None or list of dicts
    obs_for_decomp = [] if obstacles is None else obstacles

    # Get segments along x using the quadrotor radius r
    segments = trajectory_decomposition(float(x0[0]), float(xT[0]), obs_for_decomp, r)

    # Simple equal allocation of the N intervals across segments
    n_seg = len(segments)
    print(segments)
    if N < 2 * n_seg:
        raise ValueError("N must be at least 2 * number_of_segments")
    base = N // n_seg
    counts = [base] * n_seg
    rem = N - base * n_seg
    for i in range(rem):
        counts[i] += 1

    # Global control sequences
    u1 = cp.Variable(N)
    u2 = cp.Variable(N)

    # Time step h as a decision variable
    h = cp.Variable()

    constraints = []
    # bounds on h to keep problem well posed
    constraints += [h >= float(h_min), h <= float(h_max)]

    # Build per-segment state variables and dynamics using slices of global u
    x_segs = []
    u1_segs = []; u2_segs = []
    idx_offset = 0
    obs_idx = 0
    for si, seg in enumerate(segments):
        Ni = int(counts[si])
        x_seg = cp.Variable((6, Ni + 1))
        x_segs.append(x_seg)

        u1_seg = cp.Variable(N); u2_seg = cp.Variable(N)
        u1_segs.append(u1_seg); u2_segs.append(u2_seg)

        # x bounds for this segment (handle direction)
        xmin = min(seg['x_start'], seg['x_end'])
        xmax = max(seg['x_start'], seg['x_end'])
        constraints += [x_seg[0, :] >= float(xmin), x_seg[0, :] <= float(xmax)]

        # y (q2) bounds: free segments -> canvas height; gate segments -> gate opening
        if seg['kind'] == 'free':
            constraints += [x_seg[2, :] >= float(border_y[0]), x_seg[2, :] <= float(border_y[1])]
        else:
            gate_ymin = obstacles[obs_idx]['bh'] + gate_y_margin
            gate_ymax = border_y[1] - obstacles[obs_idx]['uh'] - gate_y_margin

            constraints += [x_seg[2, :] >= float(gate_ymin), x_seg[2, :] <= float(gate_ymax)]
            obs_idx += 1

        # Dynamics within segment using global u slices
        # physical model transition
        for k in range(Ni):
            global_idx = idx_offset + k
            xk = x_seg[:, k]
            uk = cp.hstack([u1[global_idx], u2[global_idx]])
            x_next = f_discrete(xk, uk, h)
            constraints += [x_seg[:, k + 1] == x_next]

        # Continuity with previous segment
        if si == 0:
            constraints += [x_seg[:, 0] == x0]
        else:
            constraints += [x_seg[:, 0] == x_segs[si - 1][:, -1]]

        # Final segment must match xT at its last state
        if si == len(segments) - 1:
            constraints += [x_seg[0, -1] == float(xT[0])]
        idx_offset += Ni

    # Global control bounds
    constraints += [u1 >= 0, u1 <= u_max]
    constraints += [u2 >= 0, u2 <= u_max]

    # Objective: minimize total control effort across global sequences
    objective = cp.Minimize(cp.sum_squares(u1) + cp.sum_squares(u2))

    problem = cp.Problem(objective, constraints)

    print("Solving with IPOPT (nlp=True, solver=cp.IPOPT). This may take a while...")
    try:
        problem.solve(nlp=True, solver=cp.IPOPT, verbose=True, tol=1e-4)
    except Exception as e:
        raise RuntimeError(
            "IPOPT solve failed. Ensure cyipopt and a compatible cvxpy build are installed. "
            f"Original error: {e}"
        )

    print(f"Status: {problem.status}, Objective: {problem.value}")

    # Collect concatenated trajectories
    x_traj = np.hstack([xs.value for xs in x_segs]) if len(x_segs) > 0 else np.zeros((6, 0))
    u1_traj = u1.value if u1.value is not None else np.zeros(N)
    u2_traj = u2.value if u2.value is not None else np.zeros(N)
    h_val = h.value if h.value is not None else None
    return x_traj, u1_traj, u2_traj, h_val, problem


if __name__ == '__main__':
    cost = 0
    x_traj = np.array([]); u1_traj = np.array([]); u2_traj = np.array([])

    _x_traj, _u1_traj, _u2_traj, _h_val, problem = solve_quadrotor(xT = np.array([3, 0.0, 1.5, 0.0, 0.0, 0.0]), 
        N=200, u_max=10.0, 
        obstacles=[{'x': .75, 'uh': .2, 'bh': .6}, {'x': 2, 'uh': 1.25, 'bh': 0}])
    #[{'x': .75, 'uh': .2, 'bh': .6}, {'x': 2, 'uh': 1.25, 'bh': 0}]
    x_traj = _x_traj; u1_traj = _u1_traj; u2_traj = _u2_traj;
    cost += problem.value

    #_x_traj, _u1_traj, _u2_traj, problem = solve_quadrotor(x0=_x_traj[:, -1], xT=[2], N=200, h=0.01, u_max=10.0, obstacles_height=[1.25, 0])
    #x_traj = np.hstack((x_traj, _x_traj)); u1_traj = np.hstack((u1_traj, _u1_traj)); u2_traj = np.hstack((u2_traj, _u2_traj));
    #cost += problem.value


    print("x shape:", x_traj.shape)
    print("u1 shape:", u1_traj.shape)
    print("u2 shape:", u2_traj.shape)
