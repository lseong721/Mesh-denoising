import os
import argparse

from utils import obj_read, read_pfm, read_calib, read_normalset_calibrate, obj_read
from renderer import Renderer
import numpy as np
import cv2
from pytorch3d.structures import Meshes
import torch
from glob import glob
import openmesh as om
import time
from pytorch3d.ops import interpolate_face_attributes
import torch.nn.functional as functional
from tqdm import tqdm
from pytorch3d.loss import mesh_laplacian_smoothing

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
    pix_normal = torch.tensor(np.array(pix_normal), device=args.device).float()
    img_size = [pix_normal.shape[1], pix_normal.shape[2]]

    # Sampling data using interval
    pix_normal = pix_normal[::args.interval]
    R = torch.tensor(R).float()[::args.interval]
    T = torch.tensor(T).float()[::args.interval]
    K = torch.tensor(K).float()[::args.interval]
    names = names[::args.interval]

    # Create renderer for fragments
    renderer = Renderer(img_size, R, T, K)

    # Vertex update using optimization
    n_view = len(R)
    znear = verts[:, 2].min().item()
    zfar = verts[:, 2].max().item()
    mesh_batch = Meshes(verts=verts[None].repeat(n_view, 1, 1), faces=faces[None].repeat(n_view, 1, 1))
    deform_V = torch.full(verts.shape, 0.0, device=args.device, requires_grad=True)
    optim = torch.optim.RMSprop([deform_V], lr=0.1)
    mesh = Meshes(verts=verts[None].repeat(n_view, 1, 1), faces=faces[None].repeat(n_view, 1, 1))

    optim_bar = tqdm(range(args.n_iter), desc='Mehs refinement')
    for i in optim_bar:
        optim.zero_grad()
        mesh_update = mesh.offset_verts(deform_V.repeat(n_view, 1))

        # Fragment update
        if i % 10 == 0:
            fragments = renderer.get_fragments(mesh_batch, znear=znear, zfar=zfar)

        verts_normal = mesh_update.verts_normals_packed()
        faces_normal = verts_normal[mesh_update.faces_packed()]
        pix_normal_pred = -interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, faces_normal)
        pix_normal_pred = pix_normal_pred.squeeze(-2)
        pix_depth_pred = fragments.zbuf.squeeze(-2)
        mask = (pix_depth_pred > 0).float()

        loss_N = functional.mse_loss(pix_normal_pred * mask, pix_normal * mask)
        loss_L = mesh_laplacian_smoothing(mesh_update, method="uniform")

        loss = loss_N * 1 + loss_L * 0.001

        loss.backward()
        optim.step()

        optim_bar.set_description('loss N: %.08f | loss L: %.08f | step: %02d' % (loss_N.item(), loss_L.item(), i))

    verts_new = mesh_update.verts_list()[0].detach().cpu().numpy()
    return verts_new, faces.detach().cpu().numpy()

# pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='2', help='gpu id')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--ratio', type=float, default=0.25, help='resize ratio')
    parser.add_argument('--n_iter', type=int, default=20, help='number of iteration')
    parser.add_argument('--interval', type=int, default=1, help='view interval')
    parser.add_argument('--filename', type=str, default='../../DB/BYroad/240109/Output_hjw/points_mvsnet/frame0005/3D_Scan/filtered_mesh_8.obj', help='obj name')
    parser.add_argument('--normaldir', type=str, default='../../DB/BYroad/240109/Output_hjw/normal/frame0005', help='normal image dir')
    parser.add_argument('--cablidir', type=str, default='../../DB/BYroad/240109/Output_hjw/cams/frame0005', help='calib file dir')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    tt = time.time()
    verts, faces = main(args)
    print(time.time() - tt)

    om.write_mesh('refine_hjw_81.obj', om.TriMesh(verts, faces))
