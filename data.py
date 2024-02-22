import torch
import numpy as np
import laspy
from knn import knn_1
from typing import List
from pgeof import pgeof


def sizes_to_pointers(sizes: torch.LongTensor):
    """Convert a tensor of sizes into the corresponding pointers. This
    is a trivial but often-required operation.
    """
    assert sizes.dim() == 1
    assert sizes.dtype == torch.long
    zero = torch.zeros(1, device=sizes.device, dtype=torch.long)
    return torch.cat((zero, sizes)).cumsum(dim=0)


def normalize_positions(pos: torch.Tensor) -> torch.Tensor:
    pos[:, 0] = (pos[:, 0] - pos[:, 0].min()) / (pos[:, 0].max() - pos[:, 0].min())
    pos[:, 1] = (pos[:, 1] - pos[:, 1].min()) / (pos[:, 1].max() - pos[:, 1].min())
    pos[:, 2] = (pos[:, 2] - pos[:, 2].min()) / (pos[:, 2].max() - pos[:, 2].min())
    return pos


class Data:
    """Holder for point cloud features."""

    def __init__(self, las: laspy.LasData,
                 keys: List[str],
                 ) -> None:
        t = [
            torch.tensor(las[ax], dtype=torch.float32, device="cpu") / 3.28084
            for ax in ["x", "y", "z"]
        ]
        self.pos = normalize_positions(torch.stack(t, dim=-1))
        self.intensity = torch.tensor(las["intensity"].astype(float))
        self.classification = las["classification"]
        self.estimate_keys(keys=keys)

    def knn(self) -> None:
        self.neighbor_index, self.neighbor_distance = knn_1(self.pos, k=25)
        return None

    def estimate_keys(
        self,
        keys: List[str],
        k_min=5,
        k_step=-1,
        k_min_search=25,
    ) -> None:
        self.knn()

        if self.pos is not None:
            # Prepare data for numpy boost interface. Note: we add each
            # point to its own neighborhood before computation
            xyz = self.pos.cpu().numpy()
            nn = torch.cat(
                (
                    torch.arange(xyz.shape[0], device="cpu").view(-1, 1),
                    self.neighbor_index,
                ),
                dim=1,
            )
            k = nn.shape[1]

            # Check for missing neighbors (indicated by -1 indices)
            n_missing = (nn < 0).sum(dim=1)
            if (n_missing > 0).any():
                sizes = k - n_missing
                nn = nn[nn >= 0]
                nn_ptr = sizes_to_pointers(sizes.cpu())  # type: ignore
            else:
                nn = nn.flatten().cpu()
                nn_ptr = torch.arange(xyz.shape[0] + 1) * k
            nn = nn.numpy().astype("uint32")
            nn_ptr = nn_ptr.numpy().astype("uint32")

            # Make sure array are contiguous before moving to C++
            xyz = np.ascontiguousarray(xyz)
            nn = np.ascontiguousarray(nn)
            nn_ptr = np.ascontiguousarray(nn_ptr)

            # C++ geometric features computation on CPU
            f = pgeof(
                xyz,
                nn,
                nn_ptr,
                k_min=k_min,
                k_step=k_step,
                k_min_search=k_min_search,
                verbose=False,
            )
            f = torch.from_numpy(f.astype("float32"))
            self.pos = self.pos.to("cuda:1")
            self.neighbor_distance = self.neighbor_distance.to("cuda:1")
            self.neighbor_index = self.neighbor_index.to("cuda:1")
            # Keep only required features
            if "linearity" in keys:
                self.linearity = f[:, 0].view(-1, 1).to("cuda:1")

            if "planarity" in keys:
                self.planarity = f[:, 1].view(-1, 1).to("cuda:1")

            if "scattering" in keys:
                self.scattering = f[:, 2].view(-1, 1).to("cuda:1")

            # Heuristic to increase importance of verticality in
            # partition
            if "verticality" in keys:
                self.verticality = f[:, 3].view(-1, 1).to("cuda:1")
                self.verticality *= 2

            if "curvature" in keys:
                self.curvature = f[:, 10].view(-1, 1).to("cuda:1")

            if "length" in keys:
                self.length = f[:, 7].view(-1, 1).to("cuda:1")

            if "surface" in keys:
                self.surface = f[:, 8].view(-1, 1).to("cuda:1")

            if "volume" in keys:
                self.volume = f[:, 9].view(-1, 1).to("cuda:1")

            # As a way to "stabilize" the normals' orientation, we
            # choose to express them as oriented in the z+ half-space
            if "normal" in keys:
                self.normal = f[:, 4:7].view(-1, 3).to("cuda:1")
                self.normal[self.normal[:, 2] < 0] *= -1
        else:
            raise ValueError("self.pos is None.")
