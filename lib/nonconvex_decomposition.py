import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import polytope


import numpy as np
from typing import List, Tuple, Optional

class NonConvexDecomposition:
    """
    Build disjunctive convex sets for 'safe ∩ (outside each unsafe polytope)'.

    Inputs:
      - Safe sets: list of H-reps (F, f). These are AND'ed (stacked).
      - Unsafe sets: list of H-reps (A, b) describing INSIDE of convex obstacles (unsafe).
                     For each obstacle k with rows i, we create facet-wise outside sets:
                         A_i x >= b_i (+ eps)
                     which in <= form becomes:
                         (-A_i) x <= -(b_i + eps)

    Output:
      - F_comb: List[List[(F_i, f_i)]]
        For each obstacle k: a list of facet-wise convex sets (F_i, f_i).
        In a MICP, you typically introduce binaries per obstacle to enforce
        'OR over i' while AND-ing safe constraints and all obstacles' disjunctions.
    """

    def __init__(self, eps: float = 0.0):
        """
        Parameters
        ----------
        eps : float
            Safety margin added to b when building 'outside' constraints A_i x >= b_i + eps.
        """
        self.eps = float(eps)


    def stack_safe(self, safe_sets):
        """
        Stack all safe constraints into a single H-rep (F, f).
        """
        if safe_sets is None or len(safe_sets) == 0:
            return None
        else:
            F_list, f_list = [], []
            for (F, f) in safe_sets:
                F_list.append(np.asarray(F, dtype=float))
                f_list.append(np.asarray(f, dtype=float).reshape(-1))
            F_safe = np.vstack(F_list)
            f_safe = np.hstack(f_list)
            return F_safe, f_safe

    def build(self, safe_sets = None, unsafe_sets = None):
        """
        Parameters
        ----------
        safe_sets  : List[(F, f)]
            List of safe convex sets in H-rep (F x <= f). These are AND'ed (stacked).
        unsafe_sets: List[(A, b)]
            List of unsafe convex sets (inside regions) in H-rep (A x <= b).

        Returns
        -------
        F_comb : List[List[(F_i, f_i)]]
            For each obstacle k, a list of facet-wise convex sets (F_i, f_i) where
            the first row encodes the 'outside' of facet i:
                (A_k[i]) x <= (b_k[i] + eps)
            followed by all stacked safe constraints.
        """
        # Stack all safe constraints once 
        if safe_sets is not None:
            safe_stacked = self.stack_safe(safe_sets)
            Fx_list, fx_list = safe_stacked

        if safe_stacked is not None and unsafe_sets is not None:
            Fx_list = []
            fx_list = []
            for idx_obs, (A, b) in enumerate(unsafe_sets):
                A = np.asarray(A, dtype=float)
                b = np.asarray(b, dtype=float).reshape(-1)

                m, d = A.shape
                Fx_list_ = []
                fx_list_ = []
                for i in range(m):
                    # Build 'outside' half-space for facet i in <= form
                    H_A = A[i : i + 1, :]                   # Half- space shape (1, d)
                    h_b = np.array([(b[i] + self.eps)])     # Half- space shape (1,)

                    if safe_stacked is not None:
                        F_safe, f_safe = safe_stacked
                        if F_safe.shape[1] != d:
                            raise ValueError(
                                f"Dim mismatch: safe dim {F_safe.shape[1]} vs obstacle dim {d}"
                            )
                        F_i = np.vstack([H_A, F_safe])
                        f_i = np.hstack([h_b, f_safe])
                    else:
                        F_i, f_i = H_A, h_b

                    Fx_list_.append(F_i)
                    fx_list_.append(f_i)
                Fx_list.append(Fx_list_)
                fx_list.append(fx_list_)

        return Fx_list, fx_list

if __name__ == "__main__":
    
    ### Velocity constraints
    vx_max = 0.05  # Maximum velocity in x-direction (m/s)
    vy_max = 0.05  # Maximum velocity in y-direction (m/s)
    vel_vertices = np.array([[vx_max, vy_max], [vx_max, -vy_max], [-vx_max, -vy_max], [-vx_max, vy_max]])


    #### Acceleration constraints
    ax_max = 1.0  # Maximum acceleration in x-direction (m/s^2)
    ay_max = 1.0  # Maximum acceleration in y-direction (m/s^2)
    accel_vertices = np.array([[ax_max, ay_max], [ax_max, -ay_max], [-ax_max, -ay_max], [-ax_max, ay_max]])


    ### In X-Y coordinate ###
    Workspace = np.array(([[178,  47],
        [704,  47],
        [704, 817],
        [178, 817]]))


    ## Convex Unsafe Region: Like vertices 
    # Obs1 =  [array([[568, 429],
    Obs1 = np.array([[568, 429],
                    [703, 429],
                    [703, 816],
                    [568, 816]])

    Obs2 = np.array([[308, 181],
                    [438, 181],
                    [438, 691],
                    [308, 691]]) 

    Obs3 = np.array([[437,  47],
                    [568,  47],
                    [568, 309],
                    [437, 309]])

    # Convex Inequality Constraints: Ax <= b
    w_poly = poly = polytope.qhull(Workspace)
    Obs = [Obs1, Obs2, Obs3]
    obs_poly = [polytope.qhull(box) for box in Obs]
    vel_poly = polytope.qhull(vel_vertices)
    accel_poly = polytope.qhull(accel_vertices)

    wA = w_poly.A
    wb = w_poly.b

    ## velocity convex constraint
    ## 0, 0 at x-y coordinate only vx, vy is constrained.
    F_v = np.hstack([np.zeros((vel_poly.A.shape[0], 2)), vel_poly.A])
    f_v = vel_poly.b

    # Sate Convex Inequality Constraints: Ax <= b
    F_w = np.hstack([wA, np.zeros((wA.shape[0], 2))])
    f_w = wb

    # Obstacle unsafe region
    ## For each obstacle polygon convex unsafe set in x-y coor, 0, 0, added for velocity
    Tx = []
    for obs in obs_poly:
        F = np.hstack([obs.A, np.zeros((obs.A.shape[0], 2))])
        fvec = obs.b
        Tx.append((F, fvec))

    F_obs = Tx.copy()


    safe_state_sets = [
    (F_w, f_w),
    (F_v, f_v),
    ]

    # 2) Unsafe sets (inside of obstacles) as (A, b) with A x <= b
    unsafe_state_sets = []
    for obs in F_obs:
        unsafe_state_sets.append((np.asarray(obs[0], dtype=float), np.asarray(obs[1], dtype=float)))

    # 3) Build facet-wise disjunctions with an optional margin (eps)
    nonconvex_decomp = NonConvexDecomposition(eps=0.0)
    Fx_list, fx_list = nonconvex_decomp.build(safe_state_sets, unsafe_state_sets)

    Gu = accel_poly.A
    gu = accel_poly.b