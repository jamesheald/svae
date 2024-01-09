import numpy as np
# np.random.seed(0)

T = 50

z_dim = 2
u_dim = 2

goal = np.random.randn(z_dim)
A = np.array(((0, -1),(1, 0)))
B = np.random.randn(z_dim, u_dim) * 1e-2
Q = 1 * np.eye(z_dim)
R = 1e-3 * np.eye(u_dim)

import control 
K, S, E = control.dlqr(A, B, Q, R)

# P = np.empty((z_dim, z_dim, T + 1))
# Qf = Q
# P[:,:,-1] = Qf
# for i in range(T, 0, -1):
#     P[:,:,i-1] = Q + A.T @ P[:,:,i] @ A - (A.T @ P[:,:,i] @ B) @ np.linalg.inv(R + B.T @ P[:,:,i] @ B) @ (B.T @ P[:,:,i] @ A)      

# K = np.empty((u_dim, z_dim, T))
# for i in range(T):
#     K[:,:,i] = np.linalg.inv(R + B.T @ P[:,:, i + 1] @ B) @ B.T @ P[:,:, i + 1] @ A

# non-zero setpoint, discrete time case
# https://websites.umich.edu/~dsbaero/library/Optimaloutputfeedbackfornon-zerosetpointregulation_thediscrete-timecase.pdf
# "Optimal output feedback for non-zero set point regulation: the discrete-time case"
z = np.zeros((T + 1, z_dim))
u_bar = np.linalg.solve(B, (np.eye(z_dim) - A) @ goal)
z[0, :] = np.ones(2)
for t in range(T):
    # u = - K[:,:,0] @ (z[t, :] - goal)

    # u = - K @ (z[t, :] - goal)
    # z[t + 1, :] = A @ z[t, :] + B @ (u + u_bar)

    A_opt = A - B @ K
    b_opt = B @ (K @ goal + u_bar)
    z[t + 1, :] = A_opt @ z[t, :] + b_opt

# A @ x + B @ (u + u_bar)
# A @ x + B @ u + B @ u_bar
# A x - B @ K @ (x - x_bar) + B @ u_bar
# A x - B @ K @ x + B @ K @ x_bar + B @ u_bar
# (A - B @ K) @ x + B @ (K @ x_bar + u_bar)

from matplotlib import pyplot as plt
plt.plot(z[:, 0], z[:, 1], '-o')
plt.plot(goal[0], goal[1], 'r+')
plt.show(block = False)

# non-zero setpoint, continuous time case
# "Lecture notes on LQR/LQG controller design" by Jo˜ao P. Hespanha, section 4.6 "Optimal set-point"
# https://staff.uz.zgora.pl/wpaszke/materialy/kss/lqrnotes.pdf

breakpoint()

# multiplicative noise in lqr [optional related reading:Todorov and Jordan, nips 2003]