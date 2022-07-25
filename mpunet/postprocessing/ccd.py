from scipy.ndimage import label
import numpy as np

def ccd_largest_part(img):
    res = np.zeros(img.shape)

    for i in np.unique(img):
        if i == 0: continue

        labels_out, num_labels = label((img == i).astype('uint8'))

        lab_list, lab_count = np.unique(labels_out, return_counts=True)

        if lab_list[0] == 0:
            lab_list = lab_list[1:]
            lab_count = lab_count[1:]

        largest_ind = np.argmax(lab_count)
        lab = lab_list[largest_ind]

        res += (i * (labels_out == lab)).astype(np.uint8)

    res = res.astype('uint8')

    return res
