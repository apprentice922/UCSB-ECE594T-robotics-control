## Docs:
- Overleaf: [link](https://www.overleaf.com/project/69e14af9f0dffb3cf79c497b)
- Slide: [link](https://docs.google.com/presentation/d/1n8pPTcRT5xGPVQExOGrhLRYNoHsAOd-rifjKl72mJAA/edit?usp=sharing) / [link](https://docs.google.com/presentation/d/19Y5HKb4kWksdJhwh4jv9UD1ZJLGkIPckAD13lv50kEk/edit?slide=id.g4702670f3a_2_74#slide=id.g4702670f3a_2_74)

## Planar quadrator control with checkpoints, minimize efforts
Demo:

![control trajectory](results/test1.gif)
    


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

### To Do:
- [x] dynamic function
- [x] environment visualization
- [x] trajectory optimization
    - [x] first-order approximation for equality constraints
    - [x] IPOPT solver
- [ ] trajectory optimization in avoiding obstacles
    - [x] minimize efforts by decomposing obstacle validation constraints
    - [ ] visulation of thrust polishment
    - [ ] minimize time consumption by MICP
    - [ ] Use soft constraints (penaly) to solve the problem  

Rest Part are in [proposal](proposal/ECE594T_project_proposal.pdf)(outdated)
