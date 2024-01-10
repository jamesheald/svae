import numpy as np
from scipy.linalg import solve_discrete_are, solve

x_dim = 2
u_dim = 1
A = np.array(((0, -1),(1, 0)))
B = np.ones((x_dim, u_dim))
b = np.ones((x_dim,))

A_new = np.zeros((x_dim + 1, x_dim + 1))
A_new[:x_dim, :x_dim] = A
A_new[:x_dim, -1] = b * 0
A_new[-1, -1] = 1

B_new = np.zeros((x_dim + 1, u_dim))
B_new[:x_dim, :] = B
B_new[-1, :] = 0

Q = np.eye(x_dim + 1)
R = np.eye(u_dim)

S = np.zeros((x_dim, u_dim))
breakpoint()
X = solve_discrete_are(A, B,np.eye(x_dim), R, e=None, s=S)
X = solve_discrete_are(A_new, B_new, Q, R, e=None, s=S)
K = solve(B.T @ X @ B + R, B.T @ X @ A + S.T)







# np.random.seed(0)
import numpy as np

T = 50

z_dim = 2
u_dim = 1

goal = np.random.randn(z_dim)
A = np.array(((0, -1),(1, 0)))
B = np.random.randn(z_dim, u_dim) * 1e-2
Q = 1 * np.eye(z_dim)
R = 1e-3 * np.eye(u_dim)

# import control 
# K, S, E = control.dlqr(A, B, Q, R)

# control.dlqr implements the below 3 lines of code but with a zillion checks etc
from scipy.linalg import solve_discrete_are, solve
S = np.zeros((Q.shape[0], R.shape[1]))
X = solve_discrete_are(A, B, Q, R, e=None, s=S)
breakpoint()
K = solve(B.T @ X @ B + R, B.T @ X @ A + S.T)

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
breakpoint()
u_bar = np.linalg.solve(B, (np.eye(z_dim) - A) @ goal)
z[0, :] = np.ones(2)
for t in range(T):
    A_opt = A - B @ K
    b_opt = B @ (K @ goal + u_bar)
    z[t + 1, :] = A_opt @ z[t, :] + b_opt
    # u = - K[:,:,0] @ (z[t, :] - goal)
    # u = - K @ (z[t, :] - goal)
    # z[t + 1, :] = A @ z[t, :] + B @ (u + u_bar)

# A @ x + B @ (u + u_bar)
# A @ x + B @ u + B @ u_bar
# A x - B @ K @ (x - x_bar) + B @ u_bar
# A x - B @ K @ x + B @ K @ x_bar + B @ u_bar
# (A - B @ K) @ x + B @ (K @ x_bar + u_bar)

from matplotlib import pyplot as plt
plt.plot(z[:, 0], z[:, 1], '-o')
plt.plot(goal[0], goal[1], 'r+')
plt.show()

# non-zero setpoint, continuous time case
# "Lecture notes on LQR/LQG controller design" by Jo˜ao P. Hespanha, section 4.6 "Optimal set-point"
# https://staff.uz.zgora.pl/wpaszke/materialy/kss/lqrnotes.pdf

breakpoint()

# multiplicative noise in lqr [optional related reading:Todorov and Jordan, nips 2003]