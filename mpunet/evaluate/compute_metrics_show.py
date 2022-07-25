import numpy as np
from mpunet.evaluate.metrics import *

from scipy.ndimage import label, generate_binary_structure

from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib
import os
import pandas as pd
from skimage.morphology import (square, rectangle, diamond, disk, cube,
                                octahedron, ball, octagon, star, binary_opening)
from skimage.morphology import (erosion, dilation, opening, closing, binary_closing,
                                binary_dilation, binary_erosion, binary_erosion,
                                area_closing,
                                white_tophat, remove_small_holes)


def extract_surface_morph(img, radius=1, bg_val=0):
    res = np.zeros(img.shape, dtype=np.uint8)

    footprint = ball(radius)

    for label in sorted(np.unique(img)):
        if label == bg_val:
            continue

        img_cur = (img == label).astype(np.uint8)

        labels_dilation = binary_dilation(img_cur, footprint).astype(np.uint8)
        labels_erosion = binary_erosion(img_cur, footprint).astype(np.uint8)

        labels_out = labels_dilation - labels_erosion

        res += (int(label) * labels_out).astype(np.uint8)

    res = res.astype('uint8')

    return res

def save_gap_regions(path, label, pred, mask, name, affine):

    if not os.path.exists(path):
        os.mkdir(path)

    label = label * mask
    pred = pred * mask

    path_pred = os.path.join(path, 'pred_'+name)
    path_label = os.path.join(path, 'label_'+name)


    nib.save(nib.Nifti1Image(label.astype(np.uint8)
                             , affine), path_label)

    nib.save(nib.Nifti1Image(pred.astype(np.uint8)
                             , affine), path_pred)


from scipy.ndimage import distance_transform_edt

def pv_distance(img, pair = ((1, 3), (2, 4))):

    min_dist = []

    for i, (femur, pelvis) in enumerate(pair):

        lf = img == femur
        lp = img == pelvis


        # dist_1 = distance_transform_edt(img != femur).astype(np.float32) * lp
        # dist_2 = distance_transform_edt(img != pelvis).astype(np.float32) * lf

        dist_1 = distance_transform_edt(img != femur).astype(np.float32)[lp]
        dist_2 = distance_transform_edt(img != pelvis).astype(np.float32)[lf]

        assert np.min(dist_1) == np.min(dist_2)

        min_dist.append(np.min(dist_1))

        del dist_1, dist_2

    return min_dist


def only_compute_distance(pred_path):

    pred_path = [i for i in os.listdir(pred_path)
                   if not (i.startswith('.') or not (i.endswith('nii') or i.endswith('gz')))]

    for j in sorted(pred_path):
        pred_path = os.path.join(pred_root, j)
        img_func = nib.load(pred_path)
        affine_2 = img_func.affine
        pred = img_func.get_fdata().astype(np.uint8)
        pred = np.squeeze(pred)

        min_dist_pred = pv_distance(pred, pair = ((1, 3), (2, 4), (3, 4), (3, 5), (4, 5)))
        print(min_dist_pred)


if __name__ == '__main__':


    label_root = '/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/validation_labels/val'

    pred_root = '/Users/px/GoogleDrive/MultiPlanarUNet/pred_traing_w_2/no_weight_2_2/ccd_symmetric'

    save_path = 'no_weight22.csv'

    assd = 0
    hausdorff_single = 0

    labels_path = [i for i in os.listdir(label_root)
                   if not (i.startswith('.') or not (i.endswith('nii') or i.endswith('gz')))]

    pred_path = [i for i in os.listdir(pred_root)
                   if not (i.startswith('.') or not (i.endswith('nii') or i.endswith('gz')))]

    # assert len(labels_path) == len(pred_path)

    dice_final = []
    dice_gap_final = []
    assd_final = []
    hausdorff_final = []

    for i, j in zip(sorted(labels_path), sorted(pred_path)):

        label_path = os.path.join(label_root, i)
        pred_path = os.path.join(pred_root, j)

        img_func = nib.load(label_path)
        affine_1 = img_func.affine
        label = img_func.get_fdata().astype(np.uint8)
        label = np.squeeze(label)
        # img = extract_surface_morph(img, radius=3, bg_val=0)

        img_func = nib.load(pred_path)
        affine_2 = img_func.affine
        pred = img_func.get_fdata().astype(np.uint8)
        pred = np.squeeze(pred)
        # img2 = extract_surface_morph(img2, radius=3, bg_val=0)

        if label.shape != pred.shape:
            print(f'cannot work on label {label.shape} and pred {pred.shape}')
            continue

        # min_dist_label = pv_distance(label)
        # min_dist_pred = pv_distance(pred)
        # print(min_dist_label)
        # print(min_dist_pred)

        dices = dice_all(label, pred, ignore_zero=False, )

        mean_dice = np.mean(dices)

        acc = np.sum((label == pred)) / np.prod(label.shape)


        mask = cal_gap(label, pred, dist=10,
                       use_predict=False
                       )

        acc_gap = np.sum((label == pred) * mask) / np.sum(mask)

        dice_gap = dice_all(label[mask], pred[mask],
                              # ignore_zero=False,
                              )
        mean_dice_gap = np.mean(dice_gap)


        ignore_sacrum = True

        if ignore_sacrum:
            label[label == 5] = 0
            pred[pred == 5] = 0


        hausdorff_list = hausdorff_distance_all(label, pred)

        hausdorff_single = np.mean(hausdorff_list)


        pred_surf = extract_surface_morph(pred, radius=3, bg_val=0)
        label_surf = extract_surface_morph(label, radius=3, bg_val=0)

        assd = assd_surface(pred_surf, label_surf)


        dice_final.append(mean_dice)
        dice_gap_final.append(mean_dice_gap)
        assd_final.append(assd * 100)
        hausdorff_final.append(hausdorff_single)

    dice_final = np.array(dice_final)
    dice_gap_final = np.array(dice_gap_final)
    assd_final = np.array(assd_final)
    hausdorff_final = np.array(hausdorff_single)

    data = {'dice_final': dice_final,
            'dice_gap_final': dice_gap_final,
            'assd_final': assd_final,
            'hausdorff_final': hausdorff_final,
            }

    df_marks = pd.DataFrame(data)
    df_marks.to_csv(save_path)

    print(round(mean_dice, 4), end='\t')
    print(round(mean_dice_gap, 4), end='\t')
    print(round(assd, 4), end='\t')
    print(round(hausdorff_single, 4))


    print(dice_gap, end='\t')
    print(hausdorff_list)
