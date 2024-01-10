# np.random.seed(0)
import numpy as np

K_version = 3

T = 50

z_dim = 2
u_dim = 1

goal = np.random.randn(z_dim)
A = np.array(((0, -1),(1, 0)))
B = np.random.randn(z_dim, u_dim) * 1e-2
Q = 1 * np.eye(z_dim)
R = 1e-3 * np.eye(u_dim)

if u_dim == 1:
    goal = (np.linalg.solve(A - np.eye(z_dim), B) * np.random.randn() * 1e4).squeeze()
else:
    goal = np.random.randn(z_dim) * 1e1

if K_version == 1:
    import control 
    K, S, E = control.dlqr(A, B, Q, R)

if K_version == 2: # nb solve_discrete_are not available in jax
    # control.dlqr implements the below 2 lines of code but with a zillion checks etc
    from scipy.linalg import solve_discrete_are, solve
    X = solve_discrete_are(A, B, Q, R)
    K = solve(B.T @ X @ B + R, B.T @ X @ A)

if K_version == 3:
    # P = np.empty((z_dim, z_dim, T + 1))
    # Qf = Q
    # P[:,:,-1] = Qf
    # for i in range(T, 0, -1):
    #     P[:,:,i-1] = Q + A.T @ P[:,:,i] @ A - (A.T @ P[:,:,i] @ B) @ np.linalg.inv(R + B.T @ P[:,:,i] @ B) @ (B.T @ P[:,:,i] @ A)      

    # # K = np.empty((u_dim, z_dim, T))
    # # for i in range(T):
    # #     K[:,:,i] = np.linalg.inv(R + B.T @ P[:,:, i + 1] @ B) @ B.T @ P[:,:, i + 1] @ A

    # # use first K
    # K = np.linalg.inv(R + B.T @ P[:,:, 1] @ B) @ B.T @ P[:,:, 1] @ A

    from functools import partial
    from jax.lax import while_loop 
    from jax.numpy.linalg import solve, norm
    def cond_fun(x, eps = 1e-6):

        P, delta_P_norm, cnt = x

        return delta_P_norm > eps

    def get_previous_P(x, A, B, Q, R):

        P, delta_P_norm, cnt = x

        prev_P = Q + A.T @ P @ A - (A.T @ P @ B) @ solve(R + B.T @ P @ B, B.T @ P @ A)

        return prev_P, norm(P - prev_P), cnt + 1

    def get_optimal_feedback_gain(A, B, Q, R):

        init_val = Q, 1e3, 0
        P, _, cnt = while_loop(cond_fun, partial(get_previous_P, A=A, B=B, Q=Q, R=R), init_val) # iterate until P converges
        K = solve(R + B.T @ P @ B, B.T @ P @ A)

        print(cnt)

        return K

    K = get_optimal_feedback_gain(A, B, Q, R)

# non-zero setpoint, discrete time case
# https://websites.umich.edu/~dsbaero/library/Optimaloutputfeedbackfornon-zerosetpointregulation_thediscrete-timecase.pdf
# "Optimal output feedback for non-zero set point regulation: the discrete-time case"
z = np.zeros((T + 1, z_dim))
# u_bar = np.linalg.solve(B, (np.eye(z_dim) - A) @ goal)
(u_bar, _, _, _) = np.linalg.lstsq(B, (np.eye(z_dim) - A) @ goal)
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