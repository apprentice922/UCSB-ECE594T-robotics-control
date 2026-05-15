## Docs:
- Overleaf: [link](https://www.overleaf.com/project/69e14af9f0dffb3cf79c497b)
- Slide: [link](https://docs.google.com/presentation/d/1n8pPTcRT5xGPVQExOGrhLRYNoHsAOd-rifjKl72mJAA/edit?usp=sharing)

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

By $$\frac{(1)}{(2)}$$, $$q_3$$ can be acquired: $$q_3 = -tan^{-1}\frac{m\ddot{q_1}}{m\ddot{q_2}+mg}$$

### Solution

Control from one point to another

$$
\begin{align*}
q &= a_7t^7 + a_6t^6+ a_5t^5 + a_4t^4+a_3t^3+a_2t^2+a_1t + a_0
\end{align*}
$$

Rest Part are in [proposal](proposal/ECE594T_project_proposal.pdf)
