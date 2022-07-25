import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import skimage
from skimage import filters
from PIL import Image

from skimage.morphology import (square, rectangle, diamond, disk, cube,
                                octahedron, ball, octagon, star, binary_opening)
from skimage.morphology import (erosion, dilation, opening, closing, binary_closing,
                                binary_dilation, binary_erosion,
                                area_closing,
                                white_tophat, remove_small_holes)

from scipy.ndimage import distance_transform_edt
from scipy.ndimage import distance_transform_cdt

from mpunet.postprocessing.connected_component_3D import *

from scipy import ndimage as ndi
import matplotlib.pyplot as plt

from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte


def reset_reanrange(labels, thresh=0.1):
    res = np.zeros(labels.shape)
    num_forge = np.sum(labels != 0)
    print(f'total num of forg = {num_forge}')

    labels_indices = sorted(np.unique(labels))
    re_aranged_ind = 1
    for i in labels_indices[1:]:
        cur_sum = np.sum(labels == i)

        print(f'total num of cur_sum = {cur_sum}')

        if cur_sum > num_forge * thresh:
            res[labels == i] = re_aranged_ind
            re_aranged_ind += 1

    return res


def watershed_w_surface(pred_img, grad_img=None, surf_img=None):

    if grad_img is None:
        grad_img = np.gradient(pred_img)
        grad_img = np.array(grad_img) ** 2
        grad_img = np.sum(grad_img, axis=0)
        grad_img = np.sqrt(grad_img)

        # grad_img = ndi.gaussian_gradient_magnitude(pred_img, sigma=2)

    binary_img_fg = pred_img > 0.7
    binary_img_fg = connected_component_3D(binary_img_fg, connectivity=26, portion_foreground=0.01)

    binary_img_bg = pred_img > 0.3

    del pred_img

    opening = binary_opening(binary_img_fg, ball(5))
    closing = binary_closing(binary_img_bg, ball(3))
    dilate_close = binary_dilation(closing, ball(3))

    sure_fg = binary_erosion(opening, ball(3))

    sure_fg = connected_component_3D(sure_fg, connectivity=26, portion_foreground=0.01) > 0


    ni_img = nib.Nifti1Image(sure_fg.astype(np.uint8)
                             , affine)
    nib.save(ni_img, 'sure_fg.nii.gz')

    sure_bg = 1 - dilate_close  # sure background area

    distances = distance_transform_edt(1 - dilate_close)

    sure_bg = watershed(distances, watershed_line=True)
    sure_bg = sure_bg == 0

    # ni_img = nib.Nifti1Image(sure_bg.astype(np.uint8)
    #                          , affine)
    # nib.save(ni_img, 'sure_bg.nii.gz')

    # from skimage.morphology import skeletonize
    # sure_bg = skeletonize(sure_bg)

    unknown = 1 - np.logical_or(sure_bg, sure_fg)

    markers, num_labels = label(sure_fg)
    unique_class_before = np.unique(markers)

    markers = reset_reanrange(markers, 0.01)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    markers[sure_bg == 1] = 1
    # Now, mark the region of unknown with zero
    markers[unknown == 1] = 0

    combined = grad_img * 0.35 + surf_img * 0.65



    # pred_img
    res = watershed(grad_img, markers) - 1
    #
    # ni_img = nib.Nifti1Image(res.astype(np.float32)
    #                          , affine)
    # nib.save(ni_img, 'watershed.nii.gz')


    return res

def test():
    image = img_as_ubyte(data.eagle())

    # denoise image
    denoised = rank.median(image, disk(2))

    # find continuous region (low gradient -
    # where less than 10 for this image) --> markers
    # disk(5) is used here to get a more smooth image
    markers = rank.gradient(denoised, disk(5)) < 10
    markers = ndi.label(markers)[0]

    # local gradient (disk(2) is used to keep edges thin)
    gradient = rank.gradient(denoised, disk(2))

    # process the watershed
    labels = watershed(gradient, ((markers == 487) + 1).astype(np.int32))

    plt.imshow(labels)

    Mf = binary_opening(img, ball(5))
    Mf = binary_erosion(Mf, ball(2))
    dialted = binary_erosion(img, ball(3))


if __name__ == '__main__':

    root_path = '/Users/px/Downloads/predictions_no_arg/nii_files_task_0'
    pred_grads_root = os.path.join(root_path, 'pred_grads')

    surface_root = os.path.join(root_path, 'surface')

    save_root = os.path.join(root_path, 'watershed')

    if not os.path.exists(save_root):
        os.mkdir(save_root)

    for i in os.listdir(root_path):
        if i.startswith('.') or not (i.endswith('nii') or i.endswith('gz')):
            continue

        surf_path = os.path.join(surface_root, i)
        pred_grads_path = os.path.join(pred_grads_root, i)
        pred_path = os.path.join(root_path, i)

        # img_func = nib.load(pred_grads_path)
        # img_grad = img_func.get_fdata().astype(np.float32)
        # img_grad = np.squeeze(img_grad)

        # pred_path = '/Users/px/Downloads/Patient_25_0_PRED.nii.gz'

        img_func = nib.load(pred_path)
        pred_img = img_func.get_fdata().astype(np.float32)
        pred_img = np.squeeze(pred_img)

        img_func = nib.load(surf_path)
        affine = img_func.affine
        surf_img = img_func.get_fdata().astype(np.float32)
        # surf = np.squeeze(surf == 2)

        save_path = os.path.join(save_root, i)
        res = watershed_w_surface(pred_img, grad_img=None, surf_img=surf_img)

        ni_img = nib.Nifti1Image(res.astype(np.float32)
                                 , affine)
        nib.save(ni_img, save_path)

    '''
    # label_path = '/Users/px/Downloads/predictions_no_arg/nii_files/Patient_11_0_PRED.nii.gz'

    binary_path = '/Users/px/Downloads/predictions_no_arg/Patient_11_0_binary.nii.gz'

    surface_path = '/Users/px/Downloads/jaw_0_115_cropped/surface/Patient_11_0.15mm_cropped.nii.gz'

    img_path = '/Users/px/Downloads/jaw_0_115_cropped/images/Patient_11_0.15mm_cropped.nii.gz'

    grad_path = '/Users/px/Downloads/predictions_no_arg/nii_files/Patient_11_0_grad.nii.gz'

    save_path = '/Users/px/Downloads/predictions_no_arg/Patient_11_0_PRED_comb.nii.gz'

    save_path = '/Users/px/Downloads/distance.nii.gz'

    save_path = '/Users/px/Downloads/predictions_no_arg/Patient_11_0_PRED_water_2.nii.gz'

    # save_path = '/Users/px/Downloads/predictions_no_arg/nii_files/Patient_7_0_PRED_tooth_simple_ccd.nii.gz'

    opening = binary_opening(binary_img, ball(3))
    closing = binary_closing(binary_img, ball(3))

    sure_fg = binary_erosion(opening, ball(5))

    sure_bg = 1 - binary_closing(closing, ball(3))  # sure background area

    from scipy.ndimage import distance_transform_edt
    dist_transform = distance_transform_edt(opening)

    # dist_transform = (dist_transform - np.min(dist_transform))/(np.max(dist_transform)-np.min(dist_transform))

    # Finding sure foreground area
    sure_fg = (dist_transform > (0.2 * np.max(dist_transform))) * 1

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = 1 - np.logical_or(sure_bg, sure_fg)

    from scipy.ndimage import label

    markers, num_labels = label(sure_fg)

    markers = reset_reanrange(markers, 0.01)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    markers[sure_bg == 1] = 1
    # Now, mark the region of unknown with zero
    markers[unknown == 1] = 0

    res = watershed(img, markers) - 1

    ni_img = nib.Nifti1Image((sure_bg == 0).astype(np.uint8)
                             , affine)
    nib.save(ni_img, save_path)

    thresh = filters.threshold_otsu(img)

    ni_img = nib.Nifti1Image((1 - sure_bg).astype(np.uint8)
                             , affine)
    nib.save(ni_img, save_path)

    np.unique(res)
    np.unique(filtered_markers)

    mask_pred = (res > 1)

    cca_fg, _p = mark_cca(labels)

    labeled_img, _p = mark_cca(markers, _p)

    markers, num_labels = label(B)
    np.unique(markers)

    markers = reset_reanrange(markers, 0.01)

    ni_img = nib.Nifti1Image((res).astype(np.uint8)
                             , affine)
    nib.save(ni_img, save_path)

    # img = cv2.imread(path)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    sure_bg = cv2.dilate(opening, kernel, iterations=5)  # sure background area

    # Perform the distance transform algorithm
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    # Normalize the distance image for range = {0.0, 1.0}
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

    # Finding sure foreground area
    ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    labels = np.copy(markers)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)

    markers = markers - 1

    mask_pred = (markers > 0) + 0

    # cv2.imwrite("C:\\Users\\Peidi Xu\\Desktop\\train_potato\\" + img_name, mask_pred)

    cca_fg, _p = mark_cca(labels)

    labeled_img, _p = mark_cca(markers, _p)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    combined = (labeled_img / 800 + img / 255)
    # =============================================================================
    #     plt.imshow((labeled_img>0) * img)
    #     plt.imshow(combined)
    #     plt.imshow(labeled_img)
    #
    # =============================================================================
    combined = np.clip(combined, 0, 1)

    '''

def OTSU_segment(img, th_begin=0, th_end=256, interval=1):
    max_g = 0
    thresh_opt = 0

    for threshold in range(th_begin, th_end, interval):
        fore_mask = img > threshold
        back_mask = img <= threshold

        fore_num = np.sum(fore_mask)
        back_num = np.sum(back_mask)

        if 0 == fore_num or 0 == back_num:
            continue

        w0 = float(fore_num) / img.size
        u0 = float(np.sum(img * fore_mask)) / fore_num
        w1 = float(back_num) / img.size
        u1 = float(np.sum(img * back_mask)) / back_num
        # intra-class variance
        g = w0 * w1 * (u0 - u1) ** 2

        if g > max_g:
            max_g = g
            thresh_opt = threshold
        if threshold % 10 == 0:
            print(f'current best thresh = {thresh_opt}')
    return thresh_opt


def mark_cca(labels, permute_fun=None):
    # Map component labels to hue val

    n_label = np.max(labels)

    if permute_fun == None:
        permute = np.random.permutation(np.arange(1, n_label + 1))
        permute = np.hstack([0, permute])
        permute_fun = lambda t: permute[t]
        permute_fun = np.vectorize(permute_fun)
    else:
        permute_fun = permute_fun
    label_hue = np.uint8(179 * permute_fun(labels) / n_label)

    # label_hue = np.uint8(179*labels/n_label)
    blank_ch = 255 * np.ones_like(label_hue)

    # =============================================================================
    #     mask = labels>0
    #
    #     label_r = np.uint8(255*labels/np.max(labels))
    #     label_g = np.uint8(255*(np.max(labels)-labels)*mask/np.max(labels))
    #     label_b = np.uint8(255*(labels)/np.max(labels))
    #
    #     labeled_img = cv2.merge([label_r, label_g, label_b])
    #
    #     return labeled_img
    #
    # =============================================================================
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2RGB)

    # set bg label to black
    labeled_img[label_hue == 0] = 0
    return labeled_img, permute_fun
