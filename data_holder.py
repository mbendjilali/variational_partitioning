import torch
import numpy as np
import laspy
from pathlib import Path
from knn import knn_1
from pgeof import pgeof
from random import sample as random_sample


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


class DataHolder:
    """Holder for point cloud features."""

    def __init__(self, las: laspy.LasData) -> None:
        if las is not None:
            t = [
                torch.tensor(las[ax], dtype=torch.float32, device="cuda:0") / 3.28084
                for ax in ["x", "y", "z"]
            ]
            self.pos = normalize_positions(torch.stack(t, dim=-1))
            self.intensity = torch.tensor(las["intensity"].astype(float))
            self.classification = las["classification"]

            self.neighbor_distance: torch.Tensor
            self.neighbor_index: torch.Tensor
            self.plane_quadrics: torch.Tensor

    def resample(self, number_of_points: int) -> None:
        idx = random_sample(range(self.pos.shape[0]), number_of_points)
        self.pos = self.pos[idx]
        self.intensity = self.intensity[idx]
        self.classification = self.classification[idx]

    def knn(self) -> None:
        self.neighbor_index, self.neighbor_distance = knn_1(self.pos, k=25)
        return None

    def estimate_normals(
        self,
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
                    torch.arange(xyz.shape[0], device="cuda:0").view(-1, 1),
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

            normal = f[:, 4:7].view(-1, 3).to("cuda:0")
            normal[normal[:, 2] < 0] *= -1
            self.normal = normal
        else:
            raise ValueError("self.pos is None.")

    def estimate_plane_quadrics(self) -> None:
        self.estimate_normals()

        # Equation 4 except we compute sqrt(a)
        support_areas = self.neighbor_distance.sum(dim=1) / (25 * np.sqrt(2))
        support_areas = support_areas.view(-1, 1)

        row_by_row_dot_product = (-self.normal * self.pos).sum(dim=1).view(-1, 1)
        # Equation 3 except we don't actually compute the plane quadrics and
        # instead we just store (n_x, n_y, n_z, -n.p_T)
        quadrics = torch.cat((self.normal, row_by_row_dot_product), dim=1)
        self.plane_quadrics = support_areas * quadrics
        return None

    def compute_diffused_quadric(
        self,
    ) -> torch.Tensor:
        # All plane quadrics
        U_flat = torch.einsum(
            "ij,ik->ijk",
            self.plane_quadrics,
            self.plane_quadrics,
        )
        width, length = self.neighbor_index.shape
        result = U_flat[self.neighbor_index.reshape(-1, 1), :, :].reshape(
            width, length, U_flat.shape[1], U_flat.shape[2]
        )
        return result.sum(dim=1)

    def to_las(self, output_path: Path) -> None:
        lasheader = laspy.LasHeader(version="1.4", point_format=3)
        lasheader.add_extra_dim(
            laspy.ExtraBytesParams(
                name="normal_x",
                type=np.float64,  # type: ignore
            )
        )
        lasheader.add_extra_dim(
            laspy.ExtraBytesParams(
                name="normal_y",
                type=np.float64,  # type: ignore
            )
        )
        lasheader.add_extra_dim(
            laspy.ExtraBytesParams(
                name="normal_z",
                type=np.float64,  # type: ignore
            )
        )
        lasdata = laspy.LasData(lasheader)
        lasdata.x = self.pos[:, 0].cpu().numpy()
        lasdata.y = self.pos[:, 1].cpu().numpy()
        lasdata.z = self.pos[:, 2].cpu().numpy()
        lasdata.normal_x = self.normal[:, 0].cpu().numpy()
        lasdata.normal_y = self.normal[:, 1].cpu().numpy()
        lasdata.normal_z = self.normal[:, 2].cpu().numpy()
        lasdata.intensity = self.intensity
        lasdata.classification = self.classification
        lasdata.write(str(output_path))
