\text{Animation 1. A video of the final cart-pole swing-up control problem.}
Our goal is to inject energy into the system in a controlled way such that it possesses an amount of energy corresponding to the unstable fixed point. To do this, we first need to linearize part of the dynamics so that we are working only with a subset of the nonlinear dynamics. The linearization of part of the dynamics of a system is appropriately called partial feedback linearization (PFL). There are two distinct types of PFL: collocated and non-collocated. We employ collocated PFL if we want to linearize the dynamics of actuated variables, and employ non-collocated PFL when we want to linearize the dynamics of unactuated variables. Collocated PFL yields the following equations (the same as those obtained in Underactuated Robotics when parameters are equal to 1):
\ddot{\theta} = -\ddot{x}\cos\theta - \sin\theta,

\ddot{x}(2-\cos^2\theta) - \sin\theta\cos\theta - \dot{\theta}^2\sin\theta = f_x.
If we apply control in the form of
f_x = \ddot{x}^{des}(2-\cos^2\theta) - \sin\theta\cos\theta - \dot{\theta}^2\sin\theta,
then we get the partially feedback linearized equations
\ddot{x} = \ddot{x}^{des},

\ddot{\theta} = -\ddot{x}^{des}\cos\theta - \sin\theta.
Let’s rewrite these equations such that u represents our control input.

\ddot{x} = u,

\ddot{\theta} = -u\cos\theta - \sin\theta.
We are interested in controlling the pendulum to the unstable fixed point and to do this we need to introduce the idea of desired energy. A lone pendulum, not on a cart but rigidly affixed to the ceiling, has energy corresponding to
E(q,\dot{q}) = \frac{1}{2}\dot{\theta}^2 - \cos\theta.
This is just the kinetic energy minus the potential energy when, again, all of the parameters are equal to one. The energy at the fixed point (θ=π) is

E(q,\dot{q}) = 1.
A suitable controller that injects energy into the system such that the error in the desired energy and the actual energy is zero is
u = k\dot{\theta}\cos\theta\,(E^{d}(q,\dot{q}) - E(q,\dot{q})).
However, we need to make sure the cart is regulated in some way so we superpose a PD controller for the cart and get
u = k_E\dot{\theta}\cos\theta\,(E^{d}(q,\dot{q}) - E(q,\dot{q})) - k_p x - k_d\dot{x}.
I chose k_E = 8, k_p = 0.5, k_d = 0.5 for the controller employed in the above animation. See [Underactuated Robotics](http://underactuated.mit.edu/index.html) for some details concerning how to ensure the energy is bounded and will go to zero. Here is a plot of the system trajectory in the phase space of the pendulum.


The code for solving this problem can be found at [Cartpole Stabilization](https://github.com/blakerbuchanan/controlsProblems.git) in the cartpoleStabilization folder. Note that all of my implementations are in the Julia programming language.

# LQR:

**State vector:** $x =\begin{bmatrix}x\\[2pt] \dot x\\[2pt] \theta\\[2pt] \dot\theta\end{bmatrix}\in\mathbb R^{4}$          $u = f_x\;\;(\text{horizontal force on the cart})$

**Base Equation:** $\dot{x}=\mathbf{A}x+\mathbf{B}u$

$M=m_c+m_p$

from textbook: $A=\begin{bmatrix}0 & 1 & 0 & 0\\[4pt]0 & 0 & -\dfrac{m_p}{m_c}\,g & 0\\[10pt]0 & 0 & 0 & 1\\[4pt]0 & 0 & \dfrac{M}{l\,m_c}\,g & 0\end{bmatrix}$$B=\begin{bmatrix}0\\[6pt]\dfrac{1}{m_c}\\[10pt]0\\[6pt]-\dfrac{1}{l\,m_c}\end{bmatrix}$

https://underactuated.mit.edu/acrobot.html

https://chatgpt.com/share/6862e12d-2128-800a-988f-33c1ec9633b3

- that chat shows how we derived A,B from this:

![image.png](attachment:4a409d9b-87d8-4d67-96da-dd1ea904ed8c:image.png)

**Cost Matrices**:

$Q = \begin{bmatrix}1.0 & 0 & 0 & 0 \\0 & 0.1 & 0 & 0 \\0 & 0 & 100.0 & 0 \\0 & 0 & 0 & 1.0\end{bmatrix} \\[1em]$Heavy penalty on angle, less on position

$R = \begin{bmatrix}0.01\end{bmatrix}$ Reduced control cost for more aggressive control

from wikipedia: $A^TP+PA-PBR^{-1}B^TP+Q=0$

we solve for P using scipy: 

```python
P = solve_continuous_are(A, B, Q, R)
```

$K=R^{-1}B^TP$

### Control:

$\theta_{error}=((\theta+\pi)\%(2*\pi))-\pi$

$x_{error}=\begin{bmatrix}x\\[2pt] \dot x\\[2pt] \theta_{error}\\[2pt] \dot\theta\end{bmatrix}$

$u=-Kx_{error}$