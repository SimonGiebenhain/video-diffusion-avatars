import os
import sys
import traceback

from math import ceil

import PIL.Image
import torch
import distinctipy
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import facer
import tyro

from pixel3dmm import env_paths

colors = distinctipy.get_colors(22, rng=0)


def viz_results(img, seq_classes, n_classes, suppress_plot = False):

    seg_img = np.zeros([img.shape[-2], img.shape[-1], 3])
    #distinctipy.color_swatch(colors)
    bad_indices = [
        0,  # background,
        1,  # neck
        # 2, skin
        3,  # cloth
        4,  # ear_r (images-space r)
        5,  # ear_l
        # 6 brow_r
        # 7 brow_l
        # 8,  # eye_r
        # 9,  # eye_l
        # 10 noise
        # 11 mouth
        # 12 lower_lip
        # 13 upper_lip
        14,  # hair,
        # 15, glasses
        16,  # ??
        17,  # earring_r
        18,  # ?
    ]
    bad_indices = []

    for i in range(n_classes):
        if i not in bad_indices:
            seg_img[seq_classes[0, :, :] == i] = np.array(colors[i])*255

    if not suppress_plot:
        plt.imshow(seg_img.astype(np.uint(8)))
        plt.show()
    return Image.fromarray(seg_img.astype(np.uint8))

def get_color_seg(img, seq_classes, n_classes):

    seg_img = np.zeros([img.shape[-2], img.shape[-1], 3])
    colors = distinctipy.get_colors(n_classes+1, rng=0)
    #distinctipy.color_swatch(colors)
    bad_indices = [
        0,  # background,
        1,  # neck
        # 2, skin
        3,  # cloth
        4,  # ear_r (images-space r)
        5,  # ear_l
        # 6 brow_r
        # 7 brow_l
        # 8,  # eye_r
        # 9,  # eye_l
        # 10 noise
        # 11 mouth
        # 12 lower_lip
        # 13 upper_lip
        14,  # hair,
        # 15, glasses
        16,  # ??
        17,  # earring_r
        18,  # ?
    ]

    for i in range(n_classes):
        if i not in bad_indices:
            seg_img[seq_classes[0, :, :] == i] = np.array(colors[i])*255


    return Image.fromarray(seg_img.astype(np.uint8))


def crop_gt_img(img, seq_classes, n_classes):

    seg_img = np.zeros([img.shape[-2], img.shape[-1], 3])
    colors = distinctipy.get_colors(n_classes+1, rng=0)
    #distinctipy.color_swatch(colors)
    bad_indices = [
        0,  # background,
        1,  # neck
        # 2, skin
        3,  # cloth
        4, #ear_r (images-space r)
        5, #ear_l
        # 6 brow_r
        # 7 brow_l
        #8,  # eye_r
        #9,  # eye_l
        # 10 noise
        # 11 mouth
        # 12 lower_lip
        # 13 upper_lip
        14,  # hair,
        # 15, glasses
        16,  # ??
        17,  # earring_r
        18,  # ?
    ]

    for i in range(n_classes):
        if i in bad_indices:
            img[seq_classes[0, :, :] == i] = 0


    #plt.imshow(img.astype(np.uint(8)))
    #plt.show()
    return img.astype(np.uint8)


device = 'cuda' if torch.cuda.is_available() else 'cpu'



face_detector = facer.face_detector('retinaface/mobilenet', device=device)
face_parser = facer.face_parser('farl/celebm/448', device=device)  # optional "farl/lapa/448"


def main(video_name : str):


    out = f'{env_paths.PREPROCESSED_DATA}/{video_name}'
    out_seg = f'{out}/seg_og/'
    out_seg_annot = f'{out}/seg_non_crop_annotations/'
    os.makedirs(out_seg, exist_ok=True)
    os.makedirs(out_seg_annot, exist_ok=True)
    folder = f'{out}/cropped/'





    frames = [f for f in os.listdir(folder) if f.endswith('.png') or f.endswith('.jpg')]

    frames.sort()

    if len(os.listdir(out_seg)) == len(frames):
        print(f'''
                        <<<<<<<< ALREADY COMPLETED SEGMENTATION FOR {video_name}, SKIPPING >>>>>>>>
                        ''')
        return

    #for file in frames:
    batch_size = 1

    for i in range(len(frames)//batch_size):
        image_stack = []
        frame_stack = []
        original_shapes = []
        for j in range(batch_size):
            file = frames[i * batch_size + j]

            if os.path.exists(f'{out_seg_annot}/color_{file}.png'):
                print('DONE')
                continue
            img = Image.open(f'{folder}/{file}')#.resize((512, 512))

            og_size = img.size

            image = facer.hwc2bchw(torch.from_numpy(np.array(img)[..., :3])).to(device=device)  # image: 1 x 3 x h x w
            image_stack.append(image)
            frame_stack.append(file[:-4])

        for batch_idx in range(ceil(len(image_stack)/batch_size)):
            image_batch = torch.cat(image_stack[batch_idx*batch_size:(batch_idx+1)*batch_size], dim=0)
            frame_idx_batch = frame_stack[batch_idx*batch_size:(batch_idx+1)*batch_size]
            og_shape_batch = original_shapes[batch_idx*batch_size:(batch_idx+1)*batch_size]

            #if True:
            try:
                with torch.inference_mode():
                    faces = face_detector(image_batch)
                    torch.cuda.empty_cache()
                    faces = face_parser(image_batch, faces, bbox_scale_factor=1.25)
                    torch.cuda.empty_cache()

                seg_logits = faces['seg']['logits']
                back_ground = torch.all(seg_logits == 0, dim=1, keepdim=True).detach().squeeze(1).cpu().numpy()
                seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
                seg_classes = seg_probs.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
                seg_classes[back_ground] = seg_probs.shape[1] + 1


                for _iidx in range(seg_probs.shape[0]):
                    frame = frame_idx_batch[_iidx]
                    iidx = faces['image_ids'][_iidx].item()
                    try:
                        I_color = viz_results(image_batch[iidx:iidx+1], seq_classes=seg_classes[_iidx:_iidx+1], n_classes=seg_probs.shape[1] + 1, suppress_plot=True)
                        I_color.save(f'{out_seg_annot}/color_{frame}.png')
                    except Exception as ex:
                        pass
                    I = Image.fromarray(seg_classes[_iidx])
                    I.save(f'{out_seg}/{frame}.png')
                torch.cuda.empty_cache()
            except Exception as exx:
                traceback.print_exc()
                continue


if __name__ == '__main__':

    tyro.cli(main)

