"""Helper Function for Visualizing Orthogonal Slices of MRI Volumes."""

import matplotlib.pyplot as plt
import numpy as np


def orthogonal_slices(volume: np.ndarray):
    z, y, x = (n // 2 for n in volume.shape)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, plane, title in zip(
        axes, (volume[:, :, x], volume[:, y, :], volume[z, :, :]), ("Sagittal", "Coronal", "Axial")
    ):
        axis.imshow(plane.T, cmap="gray", origin="lower")
        axis.set(title=title)
        axis.axis("off")
    return fig
