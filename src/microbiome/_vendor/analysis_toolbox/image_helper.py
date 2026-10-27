# Vendored from analysis_toolbox (image_helper.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


import numpy as np


def scale_to_uint8(arr, mode="simple"):

    if mode == "simple":
        arr_zeroed = arr - np.nanmin(arr)
        out = ((arr_zeroed / np.nanmax(arr_zeroed)) * 255).astype(np.uint8)
    elif mode == "cv":
        import cv2

        out = cv2.normalize(
            src=arr,
            dst=None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )

    return out


def apply_clahe(arr, **kwargs):
    import cv2

    arr_uint8 = scale_to_uint8(arr, mode="cv")

    # The declaration of CLAHE
    # clipLimit -> Threshold for contrast limiting
    clahe = cv2.createCLAHE(**kwargs)
    arr_uint8_clahe = clahe.apply(arr_uint8)

    # convert back to original range
    arr_clahe = ((arr_uint8_clahe / 255) * np.nanmax(arr)) + np.nanmin(arr)

    return arr_clahe
