import os
import argparse

from utils import obj_read, read_pfm, read_calib, read_normalset_calibrate
from renderer import Renderer
import numpy as np
import cv2
from pytorch3d.structures import Meshes
import torch
from glob import glob
import openmesh as om
import time


def main(args):
    # Read mesh data
    mesh = om.read_trimesh(args.filename)
    verts = mesh.points()
    faces = mesh.fv_indices()

    verts = torch.tensor(verts, device=args.device).float()
    faces = torch.tensor(faces, device=args.device)

    # Read calib and normal data
    R, T, K, names = read_calib(args.cablidir, args.ratio)
    pix_normal = read_normalset_calibrate(args.normaldir, names, R)
    img_size = [pix_normal.shape[1], pix_normal.shape[2]]

    pix_normal = torch.tensor(np.array(pix_normal), device=args.device).float()
    pix_normal = pix_normal.unsqueeze(-2)

    # Sampling data using interval
    pix_normal = pix_normal[::args.interval]
    R = torch.tensor(R).float()[::args.interval]
    T = torch.tensor(T).float()[::args.interval]
    K = torch.tensor(K).float()[::args.interval]
    names = names[::args.interval]

    # Create renderer for fragments
    renderer = Renderer(img_size, R, T, K)

    n_view = len(R)
    znear = verts[:, 2].min().item()
    zfar = verts[:, 2].max().item()
    mesh_batch = Meshes(verts=verts[None].repeat(n_view, 1, 1), faces=faces[None].repeat(n_view, 1, 1))
    fragments = renderer.get_fragments(mesh_batch, znear=znear, zfar=zfar)

    # Sample face normal from multi-view normal maps
    n_face = faces.shape[0]
    start_idx = torch.arange(n_view, device=args.device) * n_face
    pix2face = fragments.pix_to_face
    pix2face = pix2face - start_idx.reshape(-1, 1, 1, 1)
    valid_idx = pix2face > 0
    pix2face = pix2face[valid_idx]
    pix_normal = pix_normal[valid_idx]

    pix2face = pix2face[..., None].repeat(1, 3)

    # Use PyTorch's scatter_add function to compute sums and counts for each index
    index_sums = torch.zeros([n_face + 1, 3], dtype=pix_normal.dtype, device=args.device)
    index_counts = torch.zeros([n_face + 1, 3], dtype=torch.int32, device=args.device)

    index_sums = index_sums.scatter_add(0, pix2face, pix_normal)
    index_counts = index_counts.scatter_add(0, pix2face, torch.ones_like(pix2face, dtype=torch.int32))

    # Weighted average for face normal computation
    barycentric = fragments.bary_coords
    barycentric = barycentric[valid_idx]
    barycentric = (barycentric - 1/3).square().mean(-1).sqrt() # center is 1/3

    index_weights = torch.zeros([n_face + 1], dtype=pix_normal.dtype, device=args.device)
    index_weights = index_weights.scatter_add(0, pix2face[:, 0], barycentric)
    index_weights = index_weights.square()

    face_normal_new = index_sums / (index_weights[..., None].float() + 1e-8)
    face_normal_new = face_normal_new[:-1]
    face_normal_new = -torch.nn.functional.normalize(face_normal_new, dim=1) # minus for orientation align

    # Vertex updating using face normal guide and Laplacian regularization
    mesh_torch = Meshes(verts=verts[None], faces=faces[None])
    laplacian = mesh_torch.laplacian_packed()
    face_normal = mesh_torch.faces_normals_list()[0]

    face_valid_idx = (face_normal_new == 0).all(1)
    face_normal_new[face_valid_idx, :] = face_normal[face_valid_idx, :]
    face_normal_new = face_normal_new.detach()
    verts_new = verts.detach()

    fv_indices = torch.from_numpy(mesh.fv_indices()).long()
    vf_indices = torch.from_numpy(mesh.vf_indices()).long()

    v_adj_num = (vf_indices > -1).sum(-1, keepdim=True).to(args.device)
    v_adj_num = torch.clamp(v_adj_num, min=1)  # for isolated vertex
    face_normals = torch.cat((face_normal_new, torch.zeros((1, 3)).to(face_normal_new.dtype).to(args.device)))
    adj_face_normals = face_normals[vf_indices]

    for _ in range(args.n_iter):
        face_cent = verts_new[fv_indices].mean(1)
        v_cx = face_cent[vf_indices] - torch.unsqueeze(verts_new, 1)
        d_per_face = (adj_face_normals * v_cx).sum(-1, keepdim=True)
        v_per_face = adj_face_normals * d_per_face
        v_face_mean = v_per_face.sum(1) / v_adj_num
        verts_new = verts_new + v_face_mean + laplacian.mm(verts_new) * 0.01

    return verts_new.detach().cpu().numpy(), mesh.fv_indices()

# pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='2', help='gpu id')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--ratio', type=float, default=0.25, help='resize ratio')
    parser.add_argument('--n_iter', type=int, default=300, help='number of iteration')
    parser.add_argument('--interval', type=int, default=1, help='view interval')
    parser.add_argument('--filename', type=str, default='../../DB/BYroad/240109/Output_hjw/points_mvsnet/frame0005/3D_Scan/filtered_mesh_7.obj', help='obj name')
    parser.add_argument('--normaldir', type=str, default='../../DB/BYroad/240109/Output_hjw/normal/frame0005', help='normal image dir')
    parser.add_argument('--cablidir', type=str, default='../../DB/BYroad/240109/Output_hjw/cams/frame0005', help='calib file dir')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    tt = time.time()
    verts, faces = main(args)
    print(time.time() - tt)

    om.write_mesh('refine_hjw_7.obj', om.TriMesh(verts, faces))
