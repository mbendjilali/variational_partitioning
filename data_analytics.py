from data import Data
from typing import List
from pathlib import Path
from tqdm import tqdm
import laspy

ANALYTICS_KEYS = [
    "linearity",
    "planarity",
    "scattering",
    "verticality",
    "curvature",
    "length",
    "surface",
    "volume",
    "normal",
]


def data_to_las(data: Data,
                keys: List[str],
                outpath: Path) -> None:
    lasheader = laspy.LasHeader(version="1.4", point_format=3)
    for key in keys:
        if key != "normal":
            lasheader.add_extra_dim(
                laspy.ExtraBytesParams(
                    name=key,
                    type="float32",  # type: ignore
                )
            )
            continue
        lasheader.add_extra_dim(
            laspy.ExtraBytesParams(
                name="normal_x",
                type="float32",
            )
        )
        lasheader.add_extra_dim(
            laspy.ExtraBytesParams(
                name="normal_y",
                type="float32",
            )
        )
        lasheader.add_extra_dim(
            laspy.ExtraBytesParams(
                name="normal_z",
                type="float32",
            )
        )

    lasdata = laspy.LasData(lasheader)
    lasdata.x = data.pos[:, 0].cpu().numpy()
    lasdata.y = data.pos[:, 1].cpu().numpy()
    lasdata.z = data.pos[:, 2].cpu().numpy()
    lasdata.classification = data.classification
    for key in keys:
        if key != "normal":
            lasdata[key] = data.__getattribute__(key).flatten().cpu().numpy()
        else:
            lasdata.normal_x = data.normal[:, 0].cpu().numpy()
            lasdata.normal_y = data.normal[:, 1].cpu().numpy()
            lasdata.normal_z = data.normal[:, 2].cpu().numpy()
    lasdata.write(str(outpath))


if __name__ == "__main__":
    input_dir = Path("/data/Moussa/input_las")
    for input_path in tqdm(list(input_dir.iterdir())):
        outpath = Path("/data/Moussa/data_analytics_las") / input_path.name
        if input_path.is_file():
            lasfile = laspy.read(str(input_path))
        else:
            raise ValueError(f"{str(input_path)} doesn't exist.")
        data = Data(lasfile, keys=ANALYTICS_KEYS)
        data_to_las(
            data=data,
            keys=ANALYTICS_KEYS,
            outpath=outpath,
        )
