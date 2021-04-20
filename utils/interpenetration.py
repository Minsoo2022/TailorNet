import numpy as np

import scipy.sparse as sp
from scipy.sparse import vstack, csr_matrix
from scipy.sparse.linalg import spsolve
import trimesh

from psbody.mesh import Mesh
from psbody.mesh.geometry.vert_normals import VertNormals
from psbody.mesh.geometry.tri_normals import TriNormals
from psbody.mesh.search import AabbTree
from utils.diffusion_smoothing import numpy_laplacian_uniform as laplacian


def get_nearest_points_and_normals(vert, base_verts, base_faces):
    """For each vertex of `vert`, find nearest surface points on
    base mesh (`base_verts`, `base_faces`).
    """
    # vert : (9723, 3), base_verts : (27554, 3), base_faces : (55104, 3)
    fn = TriNormals(v=base_verts, f=base_faces).reshape((-1, 3))
    vn = VertNormals(v=base_verts, f=base_faces).reshape((-1, 3))

    tree = AabbTree(Mesh(v=base_verts, f=base_faces))
    nearest_tri, nearest_part, nearest_point = tree.nearest(vert, nearest_part=True)
    nearest_tri = nearest_tri.ravel().astype(np.long)
    nearest_part = nearest_part.ravel().astype(np.long)

    nearest_normals = np.zeros_like(vert)

    # nearest_part tells you whether the closest point in triangle abc is
    # in the interior (0), on an edge (ab:1,bc:2,ca:3), or a vertex (a:4,b:5,c:6)
    cl_tri_idxs = np.nonzero(nearest_part == 0)[0].astype(np.int)
    cl_vrt_idxs = np.nonzero(nearest_part > 3)[0].astype(np.int)
    cl_edg_idxs = np.nonzero((nearest_part <= 3) & (nearest_part > 0))[0].astype(np.int)

    nt = nearest_tri[cl_tri_idxs]
    nearest_normals[cl_tri_idxs] = fn[nt]

    nt = nearest_tri[cl_vrt_idxs]
    npp = nearest_part[cl_vrt_idxs] - 4
    nearest_normals[cl_vrt_idxs] = vn[base_faces[nt, npp]]

    nt = nearest_tri[cl_edg_idxs]
    npp = nearest_part[cl_edg_idxs] - 1
    nearest_normals[cl_edg_idxs] += vn[base_faces[nt, npp]]
    npp = np.mod(nearest_part[cl_edg_idxs], 3)
    nearest_normals[cl_edg_idxs] += vn[base_faces[nt, npp]]

    nearest_normals = nearest_normals / (np.linalg.norm(nearest_normals, axis=-1, keepdims=True) + 1.e-10)

    return nearest_point, nearest_normals


def remove_interpenetration_fast_custom(mesh, base, L=None, thresh_dir=0, thresh_dist=0, inverse=False):
    """Deforms `mesh` to remove its interpenetration from `base`.
    This is posed as least square optimization problem which can be solved
    faster with sparse solver.
    """

    eps = 0.001
    ww = 2.0

    # pdb.set_trace()
    nverts = mesh.v.shape[0]

    if L is None:
        L = laplacian(mesh.v, mesh.f)

    nearest_points, nearest_normals = get_nearest_points_and_normals(mesh.v, base.v, base.f)

    # mesh_normal = VertNormals(v=mesh.v, f=mesh.f).reshape((-1, 3))

    direction = np.sign(np.sum((mesh.v - nearest_points) * nearest_normals,
                               axis=-1))  # when merging, mesh : body, nearest_points : garment

    check = np.sum((mesh.v - nearest_points) * nearest_normals, axis=-1)
    # indices_all=  np.where(check <= thresh_dir )[0]
    indices1 = np.where(check <= 0)[0]

    indices_threshold = ((nearest_points[indices1] - mesh.v[indices1]) ** 2).mean(axis=1) < thresh_dist
    indices1_new = np.array([index for i, index in enumerate(indices1) if indices_threshold[i]])

    indices2 = np.where((0 < check) & (check <= thresh_dir) == True)[0]

    indices_threshold = ((nearest_points[indices2] - mesh.v[indices2]) ** 2).mean(axis=1) < 0.5 * thresh_dist
    indices2_new = np.array([index for i, index in enumerate(indices2) if indices_threshold[i]])

    indices = np.sort(np.concatenate([indices1_new, indices2_new]))

    ##fixed##

    pentgt_points = nearest_points[indices] - mesh.v[indices]
    pentgt_points = nearest_points[indices] \
                    + eps * pentgt_points / np.expand_dims(0.0001 + np.linalg.norm(pentgt_points, axis=1), 1)
    tgt_points = mesh.v.copy()
    tgt_points[indices] = ww * pentgt_points

    rc = np.arange(nverts)
    data = np.ones(nverts)
    data[indices] *= ww
    I = csr_matrix((data, (rc, rc)), shape=(nverts, nverts))

    A = vstack([L, I])
    b = np.vstack((
        L.dot(mesh.v),
        tgt_points
    ))

    res = spsolve(A.T.dot(A), A.T.dot(b))
    mres = Mesh(v=res, f=mesh.f)
    # pdb.set_trace()
    return mres, indices

# def remove_interpenetration_fast_custom(mesh, base, L=None, threshold = 0, direction_threshold = 0, inverse=False):
#     """Deforms `mesh` to remove its interpenetration from `base`.
#     This is posed as least square optimization problem which can be solved
#     faster with sparse solver.
#     """
#
#     eps = 0.001
#     ww = 2.0
#     distance_threshold = threshold
#     nverts = mesh.v.shape[0]
#
#     if L is None:
#         L = laplacian(mesh.v, mesh.f)
#
#     # when merging, mesh : body, base: garment
#     nearest_points, nearest_normals = get_nearest_points_and_normals(mesh.v, base.v, base.f)
#     # when merging, nearest : garment
#     #direction_logit = np.sum((mesh.v - nearest_points) * nearest_normals, axis=-1)
#     direction_logit = np.sum((mesh.v - nearest_points) * nearest_normals, axis=-1) * (((mesh.v - nearest_points) ** 2).sum(axis=1) ** (1 / 2))
#
#     # when merging, nearest garment vertex에서 bdoy vertex 까지의 벡터와 garment vertext에서의 노말벡터의 각도인데 이게 보통은 음수야
#
#     if not inverse:
#         indices = np.where(direction_logit < direction_threshold)[0]
#     else:
#         indices = np.where(direction_logit > direction_threshold)[0]
#
#     indices_threshold = ((nearest_points[indices] - mesh.v[indices]) ** 2).mean(axis=1) < distance_threshold
#     indices = np.array([index for i, index in enumerate(indices) if indices_threshold[i]])
#
#
#     pentgt_points = nearest_points[indices] - mesh.v[indices]
#     pentgt_points = nearest_points[indices] \
#                     + eps * pentgt_points / np.expand_dims(0.0001 + np.linalg.norm(pentgt_points, axis=1), 1)
#     tgt_points = mesh.v.copy()
#     tgt_points[indices] = ww * pentgt_points
#
#     rc = np.arange(nverts)
#     data = np.ones(nverts)
#     data[indices] *= ww
#     I = csr_matrix((data, (rc, rc)), shape=(nverts, nverts))
#
#     A = vstack([L, I])
#     b = np.vstack((
#         L.dot(mesh.v),
#         tgt_points
#     ))
#
#     res = spsolve(A.T.dot(A), A.T.dot(b))
#     mres = Mesh(v=res, f=mesh.f)
#     return mres, indices

def remove_interpenetration_fast(mesh, base, L=None, inverse=False):
    """Deforms `mesh` to remove its interpenetration from `base`.
    This is posed as least square optimization problem which can be solved
    faster with sparse solver.
    """

    eps = 0.001
    ww = 2.0
    nverts = mesh.v.shape[0]

    if L is None:
        L = laplacian(mesh.v, mesh.f)

    nearest_points, nearest_normals = get_nearest_points_and_normals(mesh.v, base.v, base.f)
    direction = np.sign( np.sum((mesh.v - nearest_points) * nearest_normals, axis=-1) )
    if not inverse:
        indices = np.where(direction < 0)[0]
    else:
        indices = np.where(direction > 0)[0]
    pentgt_points = nearest_points[indices] - mesh.v[indices]
    pentgt_points = nearest_points[indices] \
                    + eps * pentgt_points / np.expand_dims(0.0001 + np.linalg.norm(pentgt_points, axis=1), 1)
    tgt_points = mesh.v.copy()
    tgt_points[indices] = ww * pentgt_points

    rc = np.arange(nverts)
    data = np.ones(nverts)
    data[indices] *= ww
    I = csr_matrix((data, (rc, rc)), shape=(nverts, nverts))

    A = vstack([L, I])
    b = np.vstack((
        L.dot(mesh.v),
        tgt_points
    ))

    res = spsolve(A.T.dot(A), A.T.dot(b))
    mres = Mesh(v=res, f=mesh.f)
    return mres

if __name__ == '__main__':
    import os
    ROOT = "/BS/cpatel/work/data/learn_anim/mixture_exp31/000_0/smooth_TShirtNoCoat/0990/"
    body = Mesh(filename=os.path.join(ROOT, "body_160.ply"))
    mesh = Mesh(filename=os.path.join(ROOT, "pred_160.ply"))

    mesh1 = remove_interpenetration_fast(mesh, body)
    mesh1.write_ply("/BS/cpatel/work/proccessed.ply")
    mesh.write_ply("/BS/cpatel/work/orig.ply")
    body.write_ply("/BS/cpatel/work/body.ply")

    # from psbody.mesh import MeshViewers
    # mvs = MeshViewers((1, 2))
    # mesh1.set_vertex_colors_from_weights(np.linalg.norm(mesh.v - mesh1.v, axis=1))
    # mesh.set_vertex_colors_from_weights(np.linalg.norm(mesh.v - mesh1.v, axis=1))
    # # mesh1.set_vertex_colors_from_weights(np.zeros(mesh.v.shape[0]))
    # # mesh.set_vertex_colors_from_weights(np.zeros(mesh.v.shape[0]))
    # mvs[0][0].set_static_meshes([mesh, body])
    # mvs[0][1].set_static_meshes([mesh1, body])
    # mesh1.show()

    import ipdb
    ipdb.set_trace()