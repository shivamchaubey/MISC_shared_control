# constraints_helper.py
import numpy as np
import polytope
from typing import List, Tuple
from nonconvex_decomposition import NonConvexDecomposition  # your class with .build()

Array = np.ndarray
HRep = Tuple[Array, Array]  # (F, f) meaning F x <= f


def xywh_to_vertices(xywh: Array) -> Array:
    x, y, w, h = np.asarray(xywh, dtype=float).ravel()
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=float)


def verts_to_hrep(vertices: Array) -> HRep:
    P = polytope.qhull(np.asarray(vertices, dtype=float))
    return np.asarray(P.A, dtype=float), np.asarray(P.b, dtype=float)


class ConstraintHelper:
    """
    Build state and input constraints for state layout [x, y, vx, vy].

    - Safe state sets are AND'ed (stacked).
    - Unsafe convex obstacles are decomposed into facet-wise 'outside' disjuncts.
      NonConvexDecomposition(eps).build(...) returns Fx_list, fx_list.
    - Acceleration constraints (Gu, gu) are built directly in u-space from vertices.
    """

    def __init__(self, eps: float = 0.0):
        self.eps = float(eps)
        self._pos_dim = 2
        self._vel_dim = 2
        self._state_dim = self._pos_dim + self._vel_dim

    # ---- individual builders ----
    def workspace_constraints(self, workspace_vertices: Array) -> HRep:
        A, b = verts_to_hrep(workspace_vertices)          # A[x,y] <= b
        F = np.hstack([A, np.zeros((A.shape[0], self._vel_dim))])  # lift to [x,y,vx,vy]
        return F, b

    def velocity_constraints(self, vel_vertices: Array) -> HRep:
        A, b = verts_to_hrep(vel_vertices)                # A[vx,vy] <= b
        F = np.hstack([np.zeros((A.shape[0], self._pos_dim)), A])  # lift to [x,y,vx,vy]
        return F, b

    def obstacle_constraints(self, obstacle_vertices_list: List[Array]) -> List[HRep]:
        out = []
        for V in obstacle_vertices_list:
            A, b = verts_to_hrep(V)                       # A[x,y] <= b  (inside unsafe)
            F = np.hstack([A, np.zeros((A.shape[0], self._vel_dim))])  # lift
            out.append((F, b))
        return out

    def acceleration_constraints(self, accel_vertices: Array) -> HRep:
        # u-space constraints (no lifting), Gu*u <= gu with u=[ax, ay]
        return verts_to_hrep(accel_vertices)

    # ---- main helper ----
    def construct_state_constraints(
        self,
        workspace_vertices: Array,
        obstacle_vertices_list: List[Array],
        vel_vertices: Array,
        accel_vertices: Array,
    ):
        """
        Returns
        -------
        Fx_list, fx_list : lists for disjunctive state constraints (per obstacle, per facet)
        Gu, gu           : input-space (acceleration) constraints
        """
        F_w, f_w = self.workspace_constraints(workspace_vertices)
        F_v, f_v = self.velocity_constraints(vel_vertices)
        safe_state_sets = [(F_w, f_w), (F_v, f_v)]

        unsafe_state_sets = self.obstacle_constraints(obstacle_vertices_list)

        decomp = NonConvexDecomposition(eps=self.eps)
        Fx_list, fx_list = decomp.build(safe_state_sets, unsafe_state_sets)

        Gu, gu = self.acceleration_constraints(accel_vertices)
        return Fx_list, fx_list, Gu, gu


if __name__ == "__main__":

    helper = ConstraintHelper(eps=0.0)

    Workspace = np.array([[178, 47], [704, 47], [704, 817], [178, 817]], dtype=float)
    Obs1 = np.array([[568, 429], [703, 429], [703, 816], [568, 816]], dtype=float)
    Obs2 = np.array([[308, 181], [438, 181], [438, 691], [308, 691]], dtype=float)
    Obs3 = np.array([[437,  47], [568,  47], [568, 309], [437, 309]], dtype=float)
    vel_vertices   = np.array([[0.05, 0.05], [0.05, -0.05], [-0.05, -0.05], [-0.05, 0.05]], dtype=float)
    accel_vertices = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]], dtype=float)

    Fx_list, fx_list, Gu, gu = helper.construct_state_constraints(
        workspace_vertices=Workspace,
        obstacle_vertices_list=[Obs1, Obs2, Obs3],
        vel_vertices=vel_vertices,
        accel_vertices=accel_vertices,
    )
    print("Fx_list:", Fx_list)
    print("fx_list:", fx_list)
    print("Gu:", Gu)
    print("gu:", gu)