import torch
from pathlib import Path
import laspy
from tqdm import tqdm
import numpy as np


def normalize_positions(pos: torch.Tensor) -> torch.Tensor:
    pos[:, 0] = (pos[:, 0] - pos[:, 0].min()) / (pos[:, 0].max() - pos[:, 0].min())
    pos[:, 1] = (pos[:, 1] - pos[:, 1].min()) / (pos[:, 1].max() - pos[:, 1].min())
    pos[:, 2] = (pos[:, 2] - pos[:, 2].min()) / (pos[:, 2].max() - pos[:, 2].min())
    return pos


def calculate_point_impurity(clusters, classes):
    impurity = torch.zeros_like(clusters, dtype=torch.int64, device="cuda:1")
    for c in clusters.unique():
        mask = clusters == c
        unique_labels, counts = classes[mask].unique(return_counts=True)
        cc = torch.full_like(impurity[mask], unique_labels[counts.argmax()])
        impurity[mask] = (classes[mask] != cc).to(torch.int64)
    return impurity


if __name__ == "__main__":
    input_dir = "/data/Moussa/spt_clustering_las"
    for inpath in tqdm(Path(input_dir).iterdir()):
        lasdata = laspy.read(inpath)
        clusters = torch.tensor(
            lasdata.cluster_class_2.copy(),
            dtype=torch.int64,
            device="cuda:1",
        )
        classes = torch.tensor(
            lasdata.classification.copy(),
            dtype=torch.int64,
            device="cuda:1",
        )
        impurity = calculate_point_impurity(clusters=clusters, classes=classes)
        lasheader = lasdata.header
        if "impurity" not in list(lasheader.point_format.dimension_names):
            lasheader.add_extra_dim(
                laspy.ExtraBytesParams(
                    name="impurity",
                    type=np.int64,
                )
            )
        pos = torch.vstack([torch.tensor(lasdata.x.copy()),
                            torch.tensor(lasdata.y.copy()),
                            torch.tensor(lasdata.z.copy()),
                            ],).t()
        pos = normalize_positions(pos)
        new_lasdata = laspy.LasData(lasheader)
        new_lasdata.x = pos[:, 0].cpu().numpy()
        new_lasdata.y = pos[:, 1].cpu().numpy()
        new_lasdata.z = pos[:, 2].cpu().numpy()
        new_lasdata.classification = lasdata.classification.copy()
        new_lasdata["cluster_class_0"] = lasdata.cluster_class_0.copy()
        new_lasdata["cluster_class_1"] = lasdata.cluster_class_1.copy()
        new_lasdata["cluster_class_2"] = lasdata.cluster_class_2.copy()
        new_lasdata["impurity"] = impurity.cpu().numpy()
        new_lasdata.write(inpath)
