import numpy as np
import torch
from PIL import Image
import trimesh
import networkx as nx
from pytorch3d.ops import knn_points, knn_gather
from PIL import ImageDraw
from time import time

from pixel3dmm import env_paths

def pad_to_3_channels(img):
    if img.shape[-1] == 3:
        return img
    elif img.shape[-1] == 1:
        return np.concatenate([img, np.zeros_like(img[..., :1]), np.zeros_like(img[..., :1])], axis=-1)
    elif img.shape[-1] == 2:
        return np.concatenate([img, np.zeros_like(img[..., :1])], axis=-1)
    else:
        raise ValueError('too many dimensions in prediction type!')


def uv_pred_to_mesh(output, mask, rgb_img, right_ear = None, left_ear = None):
    valid_verts = np.load(f'{env_paths.VALID_VERTS_UV_MESH}')

    m_test = trimesh.load(f'{env_paths.head_template_ply}', process=False)
    edges = m_test.edges_unique

    g = nx.from_edgelist(m_test.edges_unique)
    one_ring = [list(g[i].keys()) for i in range(len(m_test.vertices))]

    can_uv = torch.from_numpy(np.load(f'{env_paths.FLAME_UV_COORDS}')).cuda().unsqueeze(0).float()
    can_uv[..., 0] = (can_uv[..., 0] * -1) + 1
    can_uv[..., 1] = (can_uv[..., 1] * -1) + 1


    #valid_verts = valid_verts & (np.max(np.abs(can_uv.squeeze(0).detach().cpu().numpy()), axis=-1)<0.5)


    #valid_verts = valid_verts[np.in1d(valid_verts, np.nonzero((np.max(np.abs((2*can_uv-1).squeeze(0).detach().cpu().numpy()), axis=-1)<0.499))[0])]


    #img = (pad_to_3_channels(output[0, 0].permute(1, 2, 0).detach().cpu().float().numpy() + 1) / 2 * 255).astype(
    #    np.uint8)
    img = (((rgb_img[0, 0].cpu().numpy()))*255).astype(np.uint8)
    #Image.fromarray(img).show()
    #Image.fromarray(
    #    (pad_to_3_channels(output[0, 0].permute(1, 2, 0).detach().cpu().float().numpy() + 1) / 2 * 255).astype(
    #        np.uint8)).show()

    invalid_pred = output[0, 0].abs() >= 1.0 #0.5
    output[0, 0][invalid_pred] = 10
    gt_uv = (output[0, 0].permute(1, 2, 0) + 1) / 2
    gt_uv = gt_uv * mask[0, 0].unsqueeze(-1)
    gt_uv = gt_uv.reshape(1, -1, 2)
    knn_result = knn_points(can_uv, gt_uv)
    pixel_position_width = knn_result.idx % 512
    pixel_position_height = knn_result.idx // 512

    dists = knn_result.dists.clone()

    gt_2_verts = torch.cat([pixel_position_width, pixel_position_height], dim=-1)

    # dists = (dists - dists.min())
    # dists = dists / dists.max()+
    delta = 0.00005 #1
    max_dist = delta
    empty = img
    drawn_verts = []
    for i in range(pixel_position_height.shape[1]):
        if i not in valid_verts:
            continue
        if dists[0, i, 0] < delta:
            if can_uv[0, i, 0] < 0.5:
                empty[gt_2_verts[0, i, 1].item(), gt_2_verts[0, i, 0].item(), 0] = 255  # dists[0, i, 0]
            else:
                empty[gt_2_verts[0, i, 1].item(), gt_2_verts[0, i, 0].item(), 1] = 255  # dists[0, i, 0]
            drawn_verts.append(i)
    empty = (empty).astype(np.uint8)
    im = Image.fromarray(empty)
    draw = ImageDraw.Draw(im)

    for i in drawn_verts:
        for j in one_ring[i]:
            if j in drawn_verts:
                if torch.abs(gt_2_verts[0, i] - gt_2_verts[0, j]).sum(-1) > 35:
                    continue
                draw.line(
                    [
                        (
                            gt_2_verts[0, i, 0].int().item(),
                            gt_2_verts[0, i, 1].int().item()
                        ),
                        (
                            gt_2_verts[0, j, 0].int().item(),
                            gt_2_verts[0, j, 1].int().item()
                        )

                    ], fill=128)
    empty = np.array(im)
    #for i in range(pixel_position_height.shape[1]):
    #    if i not in valid_verts:
    #        continue
    #    if dists[0, i, 0] < delta:
    #        empty[gt_2_verts[0, i, 1].item(), gt_2_verts[0, i, 0].item(), 0] = int(255 * (dists[0, i, 0] / max_dist))
    #        drawn_verts.append(i)

    for i in range(pixel_position_height.shape[1]):
        if i not in valid_verts:
            continue
        if dists[0, i, 0] < delta:
            if can_uv[0, i, 0] < 0.5:
                empty[
                np.clip(gt_2_verts[0, i, 1].item()-1, 0, empty.shape[0]-1):np.clip(gt_2_verts[0, i, 1].item()+1, 0, empty.shape[0]-1),
                np.clip(gt_2_verts[0, i, 0].item()-1, 0, empty.shape[1]-1):np.clip(gt_2_verts[0, i, 0].item()+1, 0, empty.shape[1]-1),
                0] = 200  # dists[0, i, 0]
            else:
                empty[
                np.clip(gt_2_verts[0, i, 1].item() - 1, 0, empty.shape[0] - 1):np.clip(gt_2_verts[0, i, 1].item() + 1,
                                                                                       0, empty.shape[0] - 1),
                np.clip(gt_2_verts[0, i, 0].item() - 1, 0, empty.shape[1] - 1):np.clip(gt_2_verts[0, i, 0].item() + 1,
                                                                                       0, empty.shape[1] - 1),
                1] = 200  # dists[0, i, 0]
    return empty
