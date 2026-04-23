## Docs:
- Overleaf: [link](https://www.overleaf.com/project/69e14af9f0dffb3cf79c497b)
- Google docs: [link]()

## Tentative Project: Planar quadrator control with checkpoints, minimize time
(Add any helpful information!) 
Demo: 

    =>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>
              ||    ||    ||
      __.__         ||    ||   (end)
     (start)  ||    ||          
              ||          ||
    =>=>=>=>=>=>=>=>=>=>=>=>=>=>=>=>
    


### Breakdown
- Physical model (2D for simplicity) with dynamics:
    - Drone (quatrator in HW1)
- Task: let robot patrol around a given route, and it must pass few checkpoints (imagine Formula 1)
- Objective: minimize effort ($\sum_{k=0}^{NT}u_k^2$)
- Constraints: 
    - maximum angular and its speed
    - maximum thrust
    - no collision (not deviate from the design path too much)
- Reference (not necessary)

### Dynamics

### Manipulator equation

Plannar quadrator:

$$
\begin{align}
m\ddot q_1(t) = -\sin(q_3(t))\big(u_1(t)+u_2(t)\big) \\
m\ddot q_2(t) + mg = \cos(q_3(t))\big(u_1(t)+u_2(t)\big) \\
I\ddot q_3(t) = r\big(u_2(t)-u_1(t)\big) \\
\end{align}
$$

By $\frac{(1)}{(2)}$, $q_3$ can be acquired: $q_3 = -tan^{-1}\frac{m\ddot{q_1}}{m\ddot{q_2}+mg}$

### Solution

Control from one point to another, assume the start and end angular and its speed are $0$, $x,y$ speed are also $0$. So that:
$$
\begin{align*}
q_1 &= f(t) = a_7t^7 + a_6t^6+ a_5t^5 + a_4t^4+a_3t^3+a_2t^2+a_1t + a_0, \\
q_2 &= g(t) = b_3t^3 + b_2t^2 + b_1t + b_0 \\
\end{align*}
$$

For $q_1$:

$$
\begin{align*}
f(0) = x_0, \\
f'(0) = x'_0, \\
f''(0) = 0, \\
f^{3}(0) = 0,\\
f(T) = x_T, \\
f'(T) = x'_T, \\
f''(T) = 0, \\
f^{3}(T) = 0,\\
\end{align*}
$$

For $q_2$:
$$
\begin{align*}
g(0) = y_0,\\
g'(0) = y'_0,\\
g(T) = y_T,\\
g'(T) = y'_T,\\
\end{align*}
$$