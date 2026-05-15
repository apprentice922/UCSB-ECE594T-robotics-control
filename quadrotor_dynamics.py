import numpy as np
import cvxpy as cp

# System parameters
m = 1.0      # mass
g = 10.0     # gravity
I = 0.1      # moment of inertia
r = 0.2      # arm length

def f(x, u):
    """
    Continuous-time quadrotor dynamics: xdot = f(x, u)

    State: x = [q1, v1, q2, v2, q3, w]
    Control: u = [u1, u2]

    Args:
        x: state (6,) numpy array
        u: control (2,) numpy array

    Returns:
        xdot: state derivative (6,) numpy array
    """
    # Unpack state and input. Works for numpy arrays or cvxpy Expressions/Variables
    q1, v1, q2, v2, q3, w = x
    u1, u2 = u

    # Use cvxpy nonlinear atoms so expressions work when x/u are cvxpy variables
    xdot = cp.hstack([
        v1,
        -(1.0 / m) * cp.sin(q3) * (u1 + u2),
        v2,
        (1.0 / m) * cp.cos(q3) * (u1 + u2) - g,
        w,
        (r / I) * (u2 - u1)
    ])

    return xdot


def f_discrete(x, u, h):
    """
    Discrete-time dynamics using explicit Euler: x_next = x + h*f(x, u)

    Args:
        x: state (6,) numpy array
        u: control (2,) numpy array
        h: time step (float)

    Returns:
        x_next: next state (6,) numpy array
    """
    # Explicit Euler integration. Returns a cvxpy expression when x/u are cvxpy objects
    return x + h * f(x, u)
