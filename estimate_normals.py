import torch
import numpy as np
import laspy
from pathlib import Path
from knn import knn_1
from pgeof import pgeof
from tqdm import tqdm
import click


def sizes_to_pointers(sizes: torch.LongTensor):
    """Convert a tensor of sizes into the corresponding pointers. This
    is a trivial but often-required operation.
    """
    assert sizes.dim() == 1
    assert sizes.dtype == torch.long
    zero = torch.zeros(1, device=sizes.device, dtype=torch.long)
    return torch.cat((zero, sizes)).cumsum(dim=0)


class Data:
    """Holder for point cloud features."""

    def __init__(self, las: laspy.LasData) -> None:
        if las is not None:
            t = [torch.FloatTensor(las[ax]) / 3.28084 for ax in ["x", "y", "z"]]
            self.pos = torch.stack(t, dim=-1)
            self.intensity = torch.FloatTensor(las["intensity"].astype(float))
            self.classification = las["classification"]

            # To compute using pgeof in PointFeatures

            self.neighbor_distance: torch.Tensor
            self.neighbor_index: torch.Tensor
            self.normal: torch.Tensor

    def knn(self) -> None:
        self.neighbor_index, self.neighbor_distance = knn_1(self.pos, k=25)
        return None


class NormalEstimation:
    def __init__(
        self,
        data: Data,
        k_min=5,
        k_step=-1,
        k_min_search=25,
    ):
        self.k_min = k_min
        self.k_step = k_step
        self.k_min_search = k_min_search
        self.data: Data = self._process(data=data)

    def _process(self, data: Data) -> Data:
        if data.pos is not None:
            # Prepare data for numpy boost interface. Note: we add each
            # point to its own neighborhood before computation
            device = data.pos.device
            xyz = data.pos.cpu().numpy()
            nn = torch.cat((torch.arange(xyz.shape[0]).view(-1, 1), data.neighbor_index), dim=1)
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
                k_min=self.k_min,
                k_step=self.k_step,
                k_min_search=self.k_min_search,
                verbose=False,
            )
            f = torch.from_numpy(f.astype("float32"))

            data.normal = f[:, 4:7].view(-1, 3).to(device)
            data.normal[data.normal[:, 2] < 0] *= -1

        return data


def estimate_normals_from_path(input_path: Path) -> Data:
    if input_path.is_file():
        lasfile = laspy.read(str(input_path))
    else:
        raise ValueError(f"{str(input_path)} doesn't exist.")
    las_data = Data(lasfile)
    las_data.knn()
    estimated_normals = NormalEstimation(las_data).data
    return estimated_normals


def data_to_las(
    data: Data,
    outpath: Path,
) -> None:
    lasheader = laspy.LasHeader(version="1.4", point_format=3)

    lasheader.add_extra_dim(
        laspy.ExtraBytesParams(
            name="normal_x",
            type=np.float32,
        ))
    lasheader.add_extra_dim(
        laspy.ExtraBytesParams(
            name="normal_y",
            type=np.float32,
        ))
    lasheader.add_extra_dim(
        laspy.ExtraBytesParams(
            name="normal_z",
            type=np.float32,
        ))

    lasdata = laspy.LasData(lasheader)
    lasdata.x = data.pos[:, 0].numpy()
    lasdata.y = data.pos[:, 1].numpy()
    lasdata.z = data.pos[:, 2].numpy()
    lasdata.normal_x = data.normal[:, 0].numpy()
    lasdata.normal_y = data.normal[:, 1].numpy()
    lasdata.normal_z = data.normal[:, 2].numpy()
    lasdata.classification = data.classification

    lasdata.write(str(outpath))


@click.command()
@click.option("--in",
              "input_dir",
              help="Input directory containing las files.",
              required=True,
              type=click.Path(exists=True),)
@click.option("--out",
              "output_dir",
              help="Output directory that will contain normals.",
              required=True,
              type=click.Path(exists=True),)
def main(input_dir: Path,
         output_dir: Path,
         ) -> None:
    for input_path in input_dir.iterdir():
        estimated_normals = estimate_normals_from_path(input_path)
        output_path = output_dir / input_path.name
        data_to_las(data=estimated_normals, outpath=output_path)


if __name__ == "__main__":
    main()
