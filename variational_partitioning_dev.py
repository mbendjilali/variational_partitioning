"""
# Re-implementation of Variational Shape Reconstruction via
# Quadric Error Metrics https://doi.org/10.1145/3588432.3591529.
"""

import torch
from torch.nn.functional import normalize
from data import Data
from data_analytics import ANALYTICS_KEYS
from quadric import DiffusedQuadrics
from scipy.spatial import Delaunay as _delaunay
from random import sample
from pathlib import Path
from tqdm import tqdm
import numpy as np
import laspy


empty_flag = -torch.inf


class VariationalPartitioning:
    _IN_TYPE = Data

    def _process(
        self,
        data: Data,
    ):
        self.shape = data.pos.shape[0]  # type: ignore
        self.classification = torch.tensor(data.classification).to("cuda:1")
        self.qem_classification = torch.zeros(self.shape, device="cuda:1")
        self.pos = data.pos
        self.nb_of_nodes = [max(self.shape // 3000, 4)]
        self.nb_of_nodes.append(max(self.shape // 300, self.nb_of_nodes[-1]))
        self.nb_of_nodes.append(max(self.shape // 30, self.nb_of_nodes[-1]))
        print(self.nb_of_nodes)
        self.cluster_class = empty_flag * torch.ones(
            (self.shape, 3), dtype=torch.int64, device="cuda:1"
        )
        self.nodes = {
            0: empty_flag * torch.ones((self.nb_of_nodes[0], 6), device="cuda:1"),
            1: empty_flag * torch.ones((self.nb_of_nodes[1], 6), device="cuda:1"),
            2: empty_flag * torch.ones((self.nb_of_nodes[2], 6), device="cuda:1"),
        }
        # Initialize first level
        idx = sample(range(self.shape), 4)
        infos = torch.tensor([[id, c, c] for c, id in enumerate(idx)], device="cuda:1")
        if data.pos.device != "cuda:1":
            data.pos = data.pos.to("cuda:1")
            data.normal = data.normal.to("cuda:1")
        pos = data.pos[idx]
        self.nodes[0][:4, :] = torch.cat([infos, pos], dim=1)
        return self.make_clusters(data)

    def compute_cost(  # In Development
        self,
        data: Data,
        nodes: torch.Tensor,
    ):
        """
        Computes the QEM and total cost of the addition of one point
        to any node within a particular leveleration.
        """
        # L2 cost for Voronoï-like regularity.
        l2_cost = torch.sum((data.pos.unsqueeze(1) - nodes) ** 2, dim=2)
        # Homogeneous coordinates of the node.
        ones_column = torch.ones((nodes.size(0), 1), device="cuda:1")
        h_coord = torch.cat((nodes, ones_column), dim=1)
        # Computation of the QEM.
        # Done in two stages to relieve the RAM.
        qems_1 = torch.sum(
            h_coord * (data.diffused_q[self.shape // 2:, :, :] @ h_coord.t()).transpose(2, 1), dim=2
        )
        qems_2 = torch.sum(
            h_coord * (data.diffused_q[:self.shape // 2, :, :] @ h_coord.t()).transpose(2, 1), dim=2
        )
        qems_2 = torch.cat((qems_1, qems_2), dim=0)
        del qems_1
        return l2_cost, normalize(qems_2, dim=1)

    def make_clusters(
        self,
        data: Data,
    ):
        # Three levels in the graph ( + raw PCL)
        for level in range(3):
            last_iteration = False
            exit_criterion = False
            # Fills up the level with nodes
            if level > 0:
                # Initialize level with level - 1
                self.nodes[level][: self.nb_of_nodes[level - 1], :] = self.nodes[
                    level - 1
                ]
                # level parents are level - 1 cluster indices
                self.nodes[level][: self.nb_of_nodes[level - 1], 2] = self.nodes[
                    level - 1
                ][:, 1]
            while True:
                if not last_iteration:
                    # One final run after max nb of nodes is reached
                    last_iteration = exit_criterion
                else:
                    break
                # Computes total costs and QEMs, assigns cluster to every point
                valid_nodes = self.valid_nodes(level=level)
                valid_nodes_pos = valid_nodes[:, 3:]
                total_costs, qems = self.compute_cost(
                    data=data, nodes=valid_nodes_pos,
                )
                # K-means
                argmin_costs = torch.argmin(total_costs, dim=1)
                cc = valid_nodes[:, 1]
                self.cluster_class[:, level] = cc[argmin_costs]
                idx = valid_nodes[:, 0]
                # Looks for max QEM points within each cluster
                exit_criterion = self.update_and_add_nodes(
                    data=data,
                    level=level,
                    qems=qems,
                    idx=idx,
                    cc=cc,
                )
            if level == 0:
                # 0th level nodes are their own parents.
                self.nodes[level][:, 2] = self.nodes[level][:, 1]
            idx = self.nodes[level][:, 0].to(torch.int64)
            self.cluster_class[idx, level] = self.nodes[level][:, 1]
        self.reclassify_using_qem(data=data, level=2)
        return None

    def update_and_add_nodes(
        self,
        data: torch.Tensor,
        qems: torch.Tensor,
        level: int,
        idx: torch.Tensor,
        cc: torch.Tensor,
    ) -> bool:
        nb_of_nodes = idx.size(0)
        count = 0
        # Create a mask indicating where each class value occurs
        mask = torch.zeros((self.shape, cc.size(0)), dtype=torch.bool, device="cuda:1")
        class_indices = (
            self.cluster_class[:, level] - 1
        ).long()  # Convert to integer indices
        indices = torch.arange(self.shape, device="cuda:1")
        mask[indices, class_indices] = True
        for i in range(len(cc)):
            if not torch.any(mask[:, i]):
                continue
            cluster_qems = qems[mask[:, i]][:, i]
            cluster_indices = indices[mask[:, i]]
            highest_qem_id: int = cluster_indices[cluster_qems.argmax()]
            lowest_qem_id: int = cluster_indices[cluster_qems.argmin()]
            # Update
            if torch.any(lowest_qem_id == idx):
                continue
            elif torch.any(highest_qem_id == idx):
                continue
            elif highest_qem_id == lowest_qem_id:
                continue
            self.nodes[level][i, 0] = lowest_qem_id
            self.nodes[level][i, 3:] = data.pos[lowest_qem_id]
            if nb_of_nodes + count >= self.nb_of_nodes[level]:
                return True
            # Add
            self.nodes[level][nb_of_nodes + count, 0] = highest_qem_id
            self.nodes[level][nb_of_nodes + count, 1] = nb_of_nodes + count
            self.nodes[level][nb_of_nodes + count, 2] = self.nodes[level][i, 2]
            self.nodes[level][nb_of_nodes + count, 3:] = data.pos[highest_qem_id]
            count += 1
        return False

    def get_mean_tensor(self,
                        tensor: torch.Tensor,
                        level: int,
                        ) -> torch.Tensor:
        labels = (
            self.cluster_class[:, level]
            .view(self.cluster_class[:, level].size(0), 1)
            .expand(-1, tensor.size(1))
        ).to(torch.int64)
        unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
        # Check for out-of-bounds indices after unique operation
        max_i = unique_labels.size(0) - 1
        invalid_indices_mask = (unique_labels < 0) | (unique_labels > max_i)
        if invalid_indices_mask.any():
            raise ValueError("Invalid indices detected after unique operation")

        # Use scatter_add_ with proper indices
        res = torch.zeros_like(unique_labels, dtype=torch.float)
        res.scatter_add_(0, labels, tensor)

        res = res / labels_count.float().unsqueeze(1)
        return res

    def get_pointers(self, level: int):
        pointer = torch.tensor(
            [
                (self.cluster_class[:, level] == cluster_id)
                .nonzero(as_tuple=True)[0]
                .shape[0]
                for cluster_id in range(-1, self.nb_of_nodes[level])
            ]
        ).cumsum(dim=0)
        return pointer

    def get_values(self, level: int):
        idx = [
            (self.cluster_class[:, level] == id).nonzero(as_tuple=True)[0]
            for id in range(self.nb_of_nodes[level])
        ]
        value = torch.cat(idx)
        return value

    def get_edge_i(self, pos: torch.Tensor) -> torch.Tensor:
        # Perform Delaunay triangulation
        triangulation = _delaunay(pos.cpu())
        # Access the vertex_neighbor_vertices attribute
        indptr, indices = triangulation.vertex_neighbor_vertices
        s = torch.arange(indptr.shape[0] - 1).repeat_interleave(
            torch.from_numpy((indptr[1:] - indptr[:-1]).astype("int64"))
        )
        t = torch.from_numpy(indices.astype("int64"))
        return torch.vstack((s, t))

    def valid_nodes(self, level: int) -> torch.Tensor:
        # Reshape the tensor and filter out non-empty flags simultaneously
        valid_nodes = self.nodes[level].reshape(-1, 6)
        valid_nodes = valid_nodes[valid_nodes[:, 0] != empty_flag]
        return valid_nodes

    def reclassify_using_qem(self, data: Data, level: int):
        classif = torch.tensor(data.classification, dtype=torch.float32, device="cuda:1")
        for c in self.cluster_class[:, level].unique():
            mask = self.cluster_class[:, level] == c
            unique_labels, counts = classif[mask].unique(return_counts=True)
            if counts.max() / counts.sum() >= 0.65:
                self.qem_classification[mask] += unique_labels[counts.argmax()]
            else:
                self.qem_classification[mask] = classif[mask]


def varpart_to_las(
    varpart: VariationalPartitioning,
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
    lasheader.add_extra_dim(
        laspy.ExtraBytesParams(
            name="qem_classification",
            type=np.int64,  # type: ignore
        )
    )
    lasheader.add_extra_dim(
        laspy.ExtraBytesParams(
            name="impurity",
            type=np.int64,
        )
    )
    impurity = calculate_point_impurity(clusters=varpart.cluster_class[:, 2],
                                        classes=varpart.classification,
                                        )
    lasdata = laspy.LasData(lasheader)
    lasdata.x = varpart.pos[:, 0].cpu().numpy()
    lasdata.y = varpart.pos[:, 1].cpu().numpy()
    lasdata.z = varpart.pos[:, 2].cpu().numpy()
    lasdata.cluster_class_0 = varpart.cluster_class[:, 0].cpu().numpy()
    lasdata.cluster_class_1 = varpart.cluster_class[:, 1].cpu().numpy()
    lasdata.cluster_class_2 = varpart.cluster_class[:, 2].cpu().numpy()
    lasdata.impurity = impurity.cpu().numpy()
    lasdata.qem_classification = varpart.qem_classification.cpu().numpy()
    lasdata.write(str(outpath))


def calculate_point_impurity(clusters, classes):
    impurity = torch.zeros_like(clusters, dtype=torch.int64, device="cuda:1")
    for c in clusters.unique():
        mask = clusters == c
        unique_labels, counts = classes[mask].unique(return_counts=True)
        cc = torch.full_like(impurity[mask], unique_labels[counts.argmax()])
        impurity[mask] = (classes[mask] != cc).to(torch.int64)
    return impurity

def main() -> None:
    input_dir = Path("/data/Moussa/input_las")
    for input_path in tqdm(list(input_dir.iterdir())):
        outpath = Path("/data/Moussa/qem_classification_las") / input_path.name
        if input_path.is_file():
            lasfile = laspy.read(str(input_path))
        else:
            raise ValueError(f"{str(input_path)} doesn't exist.")
        las_data = Data(lasfile, keys=ANALYTICS_KEYS)
        quadrics = DiffusedQuadrics()
        las_data = quadrics._process(las_data)
        varpart = VariationalPartitioning()
        varpart._process(las_data)
        varpart_to_las(varpart=varpart, outpath=outpath)


if __name__ == "__main__":
    main()
