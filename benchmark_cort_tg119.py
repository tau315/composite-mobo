"""High-dimensional CORT TG119 radiotherapy benchmark.

The benchmark controls 418 beamlet intensities and uses the public CORT
dose-influence matrices. The dataset is downloaded from GigaDB on first use,
unless ``CORT_TG119_DIR`` points to an already extracted copy.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import urllib.request
import zipfile

import numpy as np
from scipy import sparse
from scipy.io import loadmat
import torch

from benchmark_common import BenchmarkProblem, run_benchmark


DIM = 418
MAX_BEAMLET_INTENSITY = 70.0
DATASET_URL = (
    "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/"
    "pub/10.5524/100001_101000/100110/TG119.zip"
)
DEFAULT_CACHE_ROOT = Path(
    os.environ.get("LOCALAPPDATA", Path.home() / ".cache")
)
DEFAULT_DATA_DIR = DEFAULT_CACHE_ROOT / "composite_mobo" / "cort_tg119"
DATA_DIR = Path(os.environ.get("CORT_TG119_DIR", DEFAULT_DATA_DIR))
BEAM_ORDER = ("Gantry0", "Gantry72", "Gantry144", "Gantry216", "Gantry288")


def _required_files(data_dir: Path) -> list[Path]:
    files = [
        data_dir / "BODY_VOILIST.mat",
        data_dir / "Core_VOILIST.mat",
        data_dir / "OuterTarget_VOILIST.mat",
    ]
    files.extend(data_dir / f"{beam}_Couch0_D.mat" for beam in BEAM_ORDER)
    return files


def _download_dataset(data_dir: Path) -> None:
    if all(path.exists() for path in _required_files(data_dir)):
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "TG119.zip"
    partial = data_dir / "TG119.zip.part"
    if not archive.exists():
        print(f"Downloading public CORT TG119 data to {archive.resolve()} ...")
        request = urllib.request.Request(
            DATASET_URL, headers={"User-Agent": "composite-mobo-benchmark/1.0"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            with partial.open("wb") as destination:
                shutil.copyfileobj(response, destination)
        partial.replace(archive)

    print(f"Extracting {archive.resolve()} ...")
    with zipfile.ZipFile(archive) as bundle:
        root = data_dir.resolve()
        for member in bundle.infolist():
            target = (data_dir / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError("unsafe path in the CORT archive")
        bundle.extractall(data_dir)

    missing = [path.name for path in _required_files(data_dir) if not path.exists()]
    if missing:
        raise RuntimeError(f"CORT archive is missing required files: {missing}")


class CORTTG119Oracle:
    """Lazy dose oracle and objective-component evaluator."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self._target: sparse.csr_matrix | None = None
        self._core: sparse.csr_matrix | None = None
        self._normal: sparse.csr_matrix | None = None
        self._normalizers: tuple[float, float, float] | None = None
        self._cache: dict[bytes, np.ndarray] = {}

    def _load(self) -> None:
        if self._target is not None:
            return

        _download_dataset(self.data_dir)
        target_indices = (
            loadmat(self.data_dir / "OuterTarget_VOILIST.mat")["v"].ravel() - 1
        ).astype(np.int64)
        core_indices = (
            loadmat(self.data_dir / "Core_VOILIST.mat")["v"].ravel() - 1
        ).astype(np.int64)
        body_indices = (
            loadmat(self.data_dir / "BODY_VOILIST.mat")["v"].ravel() - 1
        ).astype(np.int64)
        excluded = np.union1d(target_indices, core_indices)
        normal_indices = np.setdiff1d(body_indices, excluded)

        target_parts = []
        core_parts = []
        normal_parts = []
        for beam in BEAM_ORDER:
            matrix = loadmat(
                self.data_dir / f"{beam}_Couch0_D.mat", variable_names=("D",)
            )["D"].tocsr()
            target_parts.append(matrix[target_indices])
            core_parts.append(matrix[core_indices])
            normal_parts.append(matrix[normal_indices])

        self._target = sparse.hstack(target_parts, format="csr")
        self._core = sparse.hstack(core_parts, format="csr")
        self._normal = sparse.hstack(normal_parts, format="csr")

        upper = self._dose_components(np.ones((1, DIM), dtype=np.float64))[0]
        target_scale = 1.0 + 0.25 * max(upper[1] - 1.05, 0.0) ** 2
        core_scale = max(0.5 * (upper[2] + upper[3]), 1.0e-12)
        normal_scale = max(0.5 * (upper[4] + upper[5]), 1.0e-12)
        self._normalizers = (target_scale, core_scale, normal_scale)
        print(
            "Loaded CORT TG119: "
            f"{self._target.shape[1]} beamlets, "
            f"{self._target.shape[0]} target voxels, "
            f"{self._core.shape[0]} core voxels, "
            f"{self._normal.shape[0]} normal-tissue voxels.",
            flush=True,
        )

    def _dose_components(self, normalized_X: np.ndarray) -> np.ndarray:
        if self._target is None or self._core is None or self._normal is None:
            raise RuntimeError("CORT matrices have not been loaded")
        fluence = MAX_BEAMLET_INTENSITY * normalized_X.T
        target_dose = np.asarray(self._target @ fluence)
        core_dose = np.asarray(self._core @ fluence)
        normal_dose = np.asarray(self._normal @ fluence)
        return np.stack(
            (
                np.quantile(target_dose, 0.05, axis=0),
                np.quantile(target_dose, 0.98, axis=0),
                core_dose.mean(axis=0),
                np.quantile(core_dose, 0.98, axis=0),
                normal_dose.mean(axis=0),
                np.quantile(normal_dose, 0.98, axis=0),
            ),
            axis=-1,
        )

    def evaluate_components(self, X: torch.Tensor) -> torch.Tensor:
        self._load()
        points = np.asarray(X.detach().double().cpu(), dtype=np.float64)
        values = np.empty((len(points), 6), dtype=np.float64)
        missing_indices: list[int] = []
        missing_points: list[np.ndarray] = []
        keys: list[bytes] = []

        for index, point in enumerate(points):
            key = point.tobytes()
            keys.append(key)
            cached = self._cache.get(key)
            if cached is None:
                missing_indices.append(index)
                missing_points.append(point)
            else:
                values[index] = cached

        if missing_points:
            calculated = self._dose_components(np.stack(missing_points))
            for index, value in zip(missing_indices, calculated):
                values[index] = value
                self._cache[keys[index]] = value.copy()
        return torch.from_numpy(values).to(dtype=torch.double, device=X.device)

    def compose(self, H: torch.Tensor) -> torch.Tensor:
        self._load()
        if self._normalizers is None:
            raise RuntimeError("CORT objective normalizers are unavailable")
        target_scale, core_scale, normal_scale = self._normalizers

        target_d95 = H[..., 0].clamp_min(0.0)
        target_d2 = H[..., 1].clamp_min(0.0)
        target_penalty = (
            (1.0 - target_d95).clamp_min(0.0).square()
            + 0.25 * (target_d2 - 1.05).clamp_min(0.0).square()
        ) / target_scale
        core_exposure = (
            0.5 * H[..., 2].clamp_min(0.0)
            + 0.5 * H[..., 3].clamp_min(0.0)
        ) / core_scale
        normal_exposure = (
            0.5 * H[..., 4].clamp_min(0.0)
            + 0.5 * H[..., 5].clamp_min(0.0)
        ) / normal_scale
        return torch.stack(
            (target_penalty, core_exposure, normal_exposure), dim=-1
        )


ORACLE = CORTTG119Oracle(DATA_DIR)

PROBLEM = BenchmarkProblem(
    name="CORT TG119 radiotherapy (3 objectives, 418 dimensions)",
    slug="cort_tg119_3obj_418d",
    dim=DIM,
    num_objectives=3,
    suite="high",
    evaluate_components=ORACLE.evaluate_components,
    compose=ORACLE.compose,
    ideal=torch.zeros(3, dtype=torch.double),
    ref_point=torch.full((3,), 2.5, dtype=torch.double),
)


if __name__ == "__main__":
    run_benchmark(PROBLEM)
