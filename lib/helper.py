import numpy as np
import polytope

def load_cis_npz(path):
    """Load CIS .npz and return a handy dict with Python-native types."""
    data = np.load(path, allow_pickle=True)

    out = {
        # compact CIS: nested list of [H|f] arrays (each is shape (m_i, n+1))
        # split forms, if you want them
        # system and constraints
        "A":  data["A"],
        "B":  data["B"],
        "Cx_dis_list": data["Cx_dis_list"].tolist(),
        "cx_dis_list": data["cx_dis_list"].tolist(),
        "Cx": data["Cx"],
        "cx": data["cx"], 
        "Fx": data["Fx"],
        "fx": data["fx"],
        "Fx_hs_list": data["Fx_hs_list"].tolist(),
        "fx_hs_list": data["fx_hs_list"].tolist(),
        "Gu": data["Gu"],
        "gu": data["gu"],
        "E":  data["E"],
        "Gw": data["Gw"],
        "Fw": data["Fw"],
        "fw": data["fw"],  # alias

        # config/meta (stored as a Python dict)
        "L": data["L"].item(),
        "T": data["T"].item(),
        "implicit": bool(data["implicit"]),
    }
    return out

# ---------- polytope projection helpers ----------
def _project_poly_H(F, f, axes=(0, 1)):
    """
    Project H-rep {x | F x <= f} to 2D axes and return (Fp, fp).
    Uses 1-based dims for tulip-control/polytope compatibility.
    """
    P = polytope.Polytope(np.asarray(F, float), np.asarray(f, float).reshape(-1))
    dims = (np.array(axes, int) + 1).tolist()
    Pp = P.project(dims)
    return np.asarray(Pp.A, float), np.asarray(Pp.b, float).reshape(-1)

def _project_Fx_list(Fx_list, fx_list, axes=(0, 1)):
    Fp_list, fp_list = [], []
    for F, f in zip(Fx_list, fx_list):
        Fp, fp = _project_poly_H(F, f, axes=axes)
        Fp_list.append(Fp); fp_list.append(fp)
    return Fp_list, fp_list
# -------------------------------------------------


#--------- For handeling polytope -------
from typing import List, Tuple
Array = np.ndarray
HRep = Tuple[Array, Array]  # (F, f) meaning F x <= f

def xywh_to_vertices(xywh: Array) -> Array:
    x, y, w, h = np.asarray(xywh, dtype=float).ravel()
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=float)


def verts_to_hrep(vertices: Array) -> HRep:
    P = polytope.qhull(np.asarray(vertices, dtype=float))
    return np.asarray(P.A, dtype=float), np.asarray(P.b, dtype=float)

# ---- only for circular robot (quickest) ----
def inflate_obstacle_poly(A: np.ndarray, b: np.ndarray, r: float):
    """
    Inflate obstacle P={x|Ax<=b} by a disk of radius r: A x <= b + r||a_i||_2.
    """
    row_norms = np.linalg.norm(np.asarray(A, float), axis=1)
    return A, np.asarray(b, float).reshape(-1) + r * row_norms

def deflate_workspace_poly(A: np.ndarray, b: np.ndarray, r: float):
    """
    Deflate workspace P={x|Ax<=b} by a disk of radius r: A x <= b - r||a_i||_2.
    """
    row_norms = np.linalg.norm(np.asarray(A, float), axis=1)
    return A, np.asarray(b, float).reshape(-1) - r * row_norms

#--------------------------------------------------

# to check if [[], [], []] has some values
def assert_has_any_entries(blocks):
    if (not blocks) or all(isinstance(b, list) and not b for b in blocks):
        raise ValueError(
            "CIS computation failed: check your constraints, discretized model, or disturbances."
        )


# def extract_nonconvex_CIS(safe_set):
#     print("Extracting non-convex CIS ........")
#     nonconvex_set = safe_set[:-1]
#     C_list = []
#     c_list = []
#     for i in range(len(nonconvex_set)):
#         C_i = []
#         c_i = []
#         for j in range(len(nonconvex_set[i])):

#             Al = nonconvex_set[i][j]['Al'][0]
#             Au = nonconvex_set[i][j]['Au'][0]
#             bl = nonconvex_set[i][j]['bl'][0].squeeze()
#             bu = nonconvex_set[i][j]['bu'][0].squeeze()
#             print (Al.shape, bl.shape, Au.shape, bu.shape)
#             if Al.shape[1] > 0:
#                 C_i.append(Al)
#                 c_i.append(bl)
#             if Au.shape[1] > 0:
#                 C_i.append(Au)
#                 c_i.append(bu)

#         C_list.append(C_i)
#         c_list.append(c_i)

#     return C_list, c_list

# def extract_convex_CIS(safe_set):
#     print ("Extracting convex CIS .........")
#     idx = len(safe_set) - 1
#     convex_set = safe_set[idx].copy()
#     Cx = convex_set['H_struct'][0]
#     cx_ = convex_set['f_struct'][0].squeeze()
#     return Cx, cx_