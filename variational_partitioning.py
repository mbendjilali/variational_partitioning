"""
# Re-implementation of Variational Shape Reconstruction via
# Quadric Error Metrics https://doi.org/10.1145/3588432.3591529.
"""


import torch
import numpy as np
import laspy
from pathlib import Path
import pandas as pd
from typing import List
from data_holder import DataHolder
from random import sample
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


def normalize_positions(pos: torch.Tensor) -> torch.Tensor:
    pos[:, 0] = (pos[:, 0] - pos[:, 0].min()) # / maximum_offset
    pos[:, 1] = (pos[:, 1] - pos[:, 1].min()) # / maximum_offset
    pos[:, 2] = (pos[:, 2] - pos[:, 2].min()) # / maximum_offset
    return pos


class VariationalParitioning:
    def __init__(
        self,
        data: DataHolder,
        lambda_l2: float = 1e-5,
        nb_of_gen: List[int] = [16, 64, 128],
    ) -> None:
        self.data = data
        self.lambda_l2 = lambda_l2
        self.nb_of_gen = nb_of_gen
        self.qem = -torch.ones((self.data.pos.shape[0], 3))
        self.cluster_class = -torch.ones((self.data.pos.shape[0], 3))

        self.generators = {
            0: sample(range(self.data.pos.shape[0]), 4),
            1: sample(range(self.data.pos.shape[0]), 4),
            2: sample(range(self.data.pos.shape[0]), 4),
        }

    def compute_cost(  # In Development
        self,
        gen: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the QEM and total cost of the addition of one point
        to any generator within a particular generation.
        """

        # Homogeneous coordinates of the generators.
        h_coords = self.data.pos[self.generators[gen]]
        h_coords = torch.cat(
            (h_coords, torch.ones((h_coords.shape[0], 1), device="cuda:0")),
            dim=1,
        )

        # Diffused quadric of the point.
        diffused_q = self.data.compute_diffused_quadric()

        # Point position set to the right format for pairwise distances.
        pos = self.data.pos.unsqueeze(1)
        gen_pos = self.data.pos[self.generators[gen]].unsqueeze(0)

        # Computation of the QEM.
        qem = torch.sum(
            h_coords * ((diffused_q @ (h_coords.t())).transpose(2, 1)), dim=2
        )

        # L2 cost for Voronoï-like regularity.
        l2_cost = torch.sum((pos - gen_pos) ** 2, dim=2)
        total_cost = qem + self.lambda_l2 * l2_cost
        return total_cost, qem

    def make_clusters(self, max_iter=16) -> None:
        """
        TODO comment.
        """
        for gen in list(self.generators.keys()):
            for _ in range(max_iter):
                total_cost, qem = self.compute_cost(gen=gen)
                # Find the indices of the minimum elements
                # in each row of total_cost
                total_cost_flat = torch.argmin(total_cost, dim=1)

                # Convert the flattened indices to 2D indices
                row_indices = torch.arange(total_cost.shape[0], device="cuda:0").view(
                    -1, 1
                )
                t_c_f_idx = torch.cat([row_indices, total_cost_flat.view(-1, 1)], dim=1)

                self.cluster_class[:, gen] = total_cost_flat
                self.qem[:, gen] = qem[t_c_f_idx[:, 0], t_c_f_idx[:, 1]]

                df = pd.DataFrame(
                    torch.cat((self.cluster_class, self.qem), dim=1)  # type: ignore
                )
                new_generators = []
                highest_qems = []
                for cluster_index in range(len(self.generators[gen])):
                    cluster_qems = df[df[gen] == cluster_index][gen + 3]
                    if cluster_qems.empty:
                        continue
                    lowest_qem_id = cluster_qems.idxmin()
                    new_generators.append(cluster_qems.idxmax())
                    highest_qems.append(cluster_qems.max())
                    self.generators[gen][cluster_index] = lowest_qem_id  # type: ignore

                next_nb_of_gen = (
                    len(self.generators[gen]) + len(new_generators) // 2 + 1
                )
                if len(self.generators[gen]) == self.nb_of_gen[gen]:
                    continue
                elif (
                    next_nb_of_gen <= self.nb_of_gen[gen]
                ):  # Fill up the number of generators until reached.
                    self.generators[gen] += sample(
                        new_generators, len(new_generators) // 2 + 1
                    )  # We can control the pace at which we introduce new generators.
                else:
                    self.generators[gen] += sample(
                        new_generators, -len(self.generators[gen]) + self.nb_of_gen[gen]
                    )


def varpart_to_las(
    varpart: VariationalParitioning,
    outpath: Path,
) -> None:
    lasheader = laspy.LasHeader(version="1.4", point_format=3)
    lasheader.add_extra_dim(
        laspy.ExtraBytesParams(
            name="cluster_class_0",
            type=np.int64,  # type: ignore
        )
    )
    lasheader.add_extra_dim(
        laspy.ExtraBytesParams(
            name="cluster_class_1",
            type=np.int64,  # type: ignore
        )
    )
    lasheader.add_extra_dim(
        laspy.ExtraBytesParams(
            name="cluster_class_2",
            type=np.int64,  # type: ignore
        )
    )
    lasdata = laspy.LasData(lasheader)
    lasdata.x = varpart.data.pos[:, 0].cpu().numpy()
    lasdata.y = varpart.data.pos[:, 1].cpu().numpy()
    lasdata.z = varpart.data.pos[:, 2].cpu().numpy()
    lasdata.cluster_class_0 = varpart.cluster_class[:, 0].cpu().numpy()
    lasdata.cluster_class_1 = varpart.cluster_class[:, 1].cpu().numpy()
    lasdata.cluster_class_2 = varpart.cluster_class[:, 2].cpu().numpy()
    lasdata.classification = varpart.data.classification
    lasdata.write(str(outpath))


@click.command()
@click.option(
    "--in",
    "input_dir",
    help="Input directory containing las files.",
    required=True,
    type=click.Path(path_type=Path, exists=True),
)
@click.option(
    "--out",
    "output_dir",
    help="Output directory that will contain normals.",
    required=True,
    type=click.Path(path_type=Path, exists=True),
)
def main(
    input_dir: Path,
    output_dir: Path,
) -> None:
    for input_path in tqdm(list(input_dir.iterdir())):
        if input_path.is_file():
            lasfile = laspy.read(str(input_path))
        else:
            raise ValueError(f"{str(input_path)} doesn't exist.")
        las_data = DataHolder(lasfile)
        las_data.estimate_plane_quadrics()
        varpart = VariationalParitioning(las_data)
        varpart.make_clusters()
        output_path = output_dir / input_path.name
        varpart_to_las(varpart=varpart, outpath=output_path)
        break


if __name__ == "__main__":
    main()
