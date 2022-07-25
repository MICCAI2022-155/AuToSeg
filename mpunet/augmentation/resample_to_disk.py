seed = 42  # for reproducibility


import nibabel as nib
import nibabel.processing
from skimage.transform import rescale


from aug_save_to_disk import *

def rescale():
    path = '/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/vessels/train/images/cropped_RAT_03_reshaped.nii.gz'
    input_img = nib.load(path)
    output_path = '/Users/px/Downloads/a.nii.gz'

    voxel_before = input_img.header.get_zooms()[0]

    resampled_img = nib.processing.resample_to_output(input_img, voxel_before * 2, order=1)
    # resampled_img = nib.processing.smooth_image(input_img, 0.5)
    # resampled_img = nib.processing.resample_from_to(input_img, 1)
    nib.save(resampled_img, output_path)

def save_to_disk(img_before, lab_before, img_after, lab_after, voxel_after=0.15):

    for input_path, output_path in zip([img_before, lab_before], [img_after, lab_after]):
        input_img = nib.load(input_path)

        resampled_img = nib.processing.smooth_image(input_img, 0.5)

        # resampled_img = nib.processing.resample_to_output(input_img, voxel_after,
        #                                                   cval=0,order=0,
        #                                                   mode='nearest')

        # data = input_img.get_fdata()
        # data = rescale(data, 0.5,
        #                preserve_range=True).astype(np.float32)
        # affine_func = input_img.affine
        # resampled_img = nib.Nifti1Image(data, affine_func)

        nib.save(resampled_img, output_path)


def blur_save_to_disk(img_before, lab_before, img_after, lab_after, smooth_level=0.5):

    for i, (input_path, output_path) in enumerate(zip([img_before, lab_before], [img_after, lab_after])):
        input_img = nib.load(input_path)

        if i == 0:
            resampled_img = nib.processing.smooth_image(input_img, smooth_level)
        else:
            resampled_img = input_img
        nib.save(resampled_img, output_path)


if __name__ == '__main__':

    dataset_dir = '/Users/px/Downloads/training_same_FOV'

    dataset_dir = '/Users/px/Downloads/jaw_0_115_cropped'

    dataset_dir = '/Users/px/Downloads/jaw_0_115_cropped_upper'

    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')

    aug_root = os.path.join(dataset_dir, 'aug')
    aug_image_root = os.path.join(aug_root, 'images')
    aug_label_root = os.path.join(aug_root, 'labels')

    safe_make(aug_root)
    safe_make(aug_image_root)
    safe_make(aug_label_root)

    image_names = os.listdir(images_dir)
    label_names = os.listdir(images_dir)
    assert len(image_names) == len(label_names)

    train_15_list = ['Patient_4_0.15mm_cropped.nii.gz', 'Patient_5_0.15mm_cropped.nii.gz',
                     'Patient_15_0.15mm_cropped.nii.gz', 'Patient_25_0.15mm_cropped.nii.gz'
                     ]
    train_15_list = ['Patient_5_0.15mm_cropped.nii.gz',
                     ]
    train_20_list = ['Patient_15_0.15mm_cropped.nii.gz', 'Patient_26_0.15mm_cropped.nii.gz',
                     ]
    train_25_list = ['Patient_25_0.15mm_cropped.nii.gz',
                     ]


    # train_25_list = ['Patient_5_0.15mm_cropped.nii.gz']
    # train_20_list = ['Patient_5_0.15mm_cropped.nii.gz']

    for i in os.listdir(images_dir):
        img_before = os.path.join(os.path.join(images_dir, i))
        lab_before = os.path.join(os.path.join(labels_dir, i))

        # if '15mm' in i or '25mm' in i or '0.2mm' in i:
        if i in train_15_list or i in train_20_list or i in train_25_list:
            voxel_after = 0.3
            voxel_after = 'blurred'
            img_after = os.path.join(os.path.join(aug_image_root, str(voxel_after)+i))
            lab_after = os.path.join(os.path.join(aug_label_root, str(voxel_after)+i))

            # save_to_disk(img_before, lab_before, img_after, lab_after, voxel_after)

            blur_save_to_disk(img_before, lab_before, img_after, lab_after, smooth_level=0.5)

            if i in train_15_list or i in train_20_list:
                voxel_after = 'blurred_7_'
                img_after = os.path.join(os.path.join(aug_image_root, str(voxel_after) + i))
                lab_after = os.path.join(os.path.join(aug_label_root, str(voxel_after) + i))

                blur_save_to_disk(img_before, lab_before, img_after, lab_after, smooth_level=0.7)

            # if '15mm' in i or '0.2mm' in i:
            #     voxel_after = 0.5
            #     img_after = os.path.join(os.path.join(aug_image_root, str(voxel_after)+i))
            #     lab_after = os.path.join(os.path.join(aug_label_root, str(voxel_after)+i))
            #
            #     save_to_disk(img_before, lab_before, img_after, lab_after, voxel_after)