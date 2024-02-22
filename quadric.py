import torch
import numpy as np
from data import Data


__all__ = ["DiffusedQuadrics"]


class DiffusedQuadrics:
    """
    # Re-implementation of Variational Shape Reconstruction via
    # Quadric Error Metrics https://doi.org/10.1145/3588432.3591529.
    Another script exists named variational.py implementing the
    partitioning.

    Parameters
    ----------
    :param k: int
        Number of neighbors to search for
    """

    _IN_TYPE = Data
    _OUT_TYPE = Data

    def __init__(self, k=25):
        self.k = k

    def _process(self, data):
        if data.pos.device != "cuda:1":
            data.pos = data.pos.to("cuda:1")
            data.normal = data.normal.to("cuda:1")
            data.neighbor_index = data.neighbor_index.to("cuda:1")
            data.neighbor_distance = data.neighbor_distance.to("cuda:1")
        # Equation 4 except we compute sqrt(a)
        support_areas = data.neighbor_distance.sum(dim=1) / (self.k * np.sqrt(2))
        support_areas = support_areas.view(-1, 1).to("cuda:1")
        row_by_row_dot_product = (-data.normal * data.pos).sum(dim=1).view(-1, 1)
        # Equation 3 except we don't actually compute the plane quadrics and
        # instead we just store (n_x, n_y, n_z, -n.p_T)
        plane_quadrics = support_areas * (
            torch.cat((data.normal, row_by_row_dot_product), dim=1)
        )

        # All plane quadrics
        U_flat = torch.einsum(
            "ij,ik->ijk",
            plane_quadrics,
            plane_quadrics,
        )
        width, length = data.neighbor_index.shape
        result = U_flat[data.neighbor_index.reshape(-1, 1), :, :].reshape(
            width, length, U_flat.shape[1], U_flat.shape[2]
        )

        # data.diffused_q is divided in two to address cuda:1 OOM errors during QEM computation
        data.diffused_q = result.sum(dim=1)
        return data
