"""Convert the pinned PHM 2025 engine snapshots into causal TimeF windows."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import ZipFile

import pyarrow as pa

from timenet.connectors import BaseConnector
from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import OrdinalAxis
from timenet.errors import TimeNetConnectorError
from timenet.json import JsonMapping
from timenet.types import ScalarPredictionTask, TimeSeriesSpec
from timenet.types.specs import MissingReason, MissingValues
from timenet.types.splits import Split
from timenet_connectors.download import ensure_archive


if TYPE_CHECKING:
    import pandas as pd


SENSORS = (
    "Sensed_Altitude",
    "Sensed_Mach",
    "Sensed_Pamb",
    "Sensed_Pt2",
    "Sensed_TAT",
    "Sensed_WFuel",
    "Sensed_VAFN",
    "Sensed_VBV",
    "Sensed_Fan_Speed",
    "Sensed_Core_Speed",
    "Sensed_T25",
    "Sensed_T3",
    "Sensed_Ps3",
    "Sensed_T45",
)
OPTIONAL_SENSORS = ("Sensed_P25", "Sensed_T5")
TARGETS = ("Cycles_to_WW", "Cycles_to_HPC_SV", "Cycles_to_HPT_SV")
WINDOW_POSITIONS = 151
WINDOW_STRIDE = 10
CYCLE_STEP = 10
ARCHIVES = {
    "train": (
        "https://phm-datasets.s3.us-east-1.amazonaws.com/Data_Challenge_PHM2025_training_data.zip",
        "6bcac5f9eff0a53da31da825e5df6da1501113c0046d4e45b1226b41f3750e2b",
    ),
    "test": (
        "https://phm-datasets.s3.us-east-1.amazonaws.com/Data_Challenge_PHM2025_test_data.zip",
        "df66be7b4577f329a61f4fe8fa6b4ec714ba908f8d00d6b89415dea25da9e812",
    ),
    "validation": (
        "https://phm-datasets.s3.us-east-1.amazonaws.com/Data_Challenge_PHM2025_validation_data_gdcmktwb68.zip",
        "7f2a2c324af39b5e86563179eac1d0c968107aebbbbcbe09589073ebedc11e2d",
    ),
}


@dataclass(frozen=True)
class PHMSource:
    """One extracted CSV and its official partition."""

    partition: str
    path: Path


@dataclass(frozen=True)
class _Window:
    frame: pd.DataFrame
    start: int
    stop: int
    cycle_col: str
    cycle_start: int
    metadata: JsonMapping


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signal_spec(column: str) -> TimeSeriesSpec:
    sensor = column.startswith("Sensed_")
    return TimeSeriesSpec(
        spec_type=f"phm-2025-{column.lower().replace('_', '-')}",
        name=column,
        unit_value=None,  # The source does not specify physical units.
        dtype="float64" if sensor else "int32",
        missing=MissingValues(reason=MissingReason.SENSOR_GAP) if sensor else None,
    )


def _values(frame: pd.DataFrame, column: str, start: int, stop: int, offset: int) -> pa.Array:
    values = frame[column].iloc[start:stop]
    if offset:
        values = values.sub(offset)
    return pa.array(values, type=pa.float64() if column.startswith("Sensed_") else pa.int32(), from_pandas=True)


def _record(window: _Window) -> Record:
    columns = (
        window.cycle_col,
        "Snapshot",
        *SENSORS,
        *(s for s in OPTIONAL_SENSORS if s in window.frame.columns),
    )
    axis = OrdinalAxis()
    signals = tuple(
        Signal.from_loader(
            spec=_signal_spec(column),
            name="Cycle" if column == window.cycle_col else column,
            time_axis=axis,
            loader=lambda col=column: _values(
                window.frame,
                col,
                window.start,
                window.stop,
                window.cycle_start if col == window.cycle_col else 0,
            ),
            n_values=window.stop - window.start,
        )
        for column in columns
    )
    return Record(sources=(Source(name="engine snapshots", signals=signals),), metadata=window.metadata)


class PHM2025EHMConnector(BaseConnector[PHMSource]):
    """Build PHM engine-maintenance prediction windows from official ZIP archives."""

    def download(self, cache_dir: Path) -> list[PHMSource]:  # ruff: ignore[no-self-use] - BaseConnector hook
        """Fetch pinned archives or extract locally supplied copies without parsing CSVs."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_dir = os.environ.get("PHM2025_ARCHIVE_DIR")
        for partition, (url, expected_hash) in ARCHIVES.items():
            target = cache_dir / partition
            if local_dir:
                archive_path = Path(local_dir) / f"{partition if partition != 'train' else 'training'}.zip"
                if not archive_path.is_file() or _hash(archive_path) != expected_hash:
                    raise TimeNetConnectorError(f"Missing or checksum-mismatched PHM archive: {archive_path}")
                if not target.exists():
                    target.mkdir()
                    with ZipFile(archive_path) as archive:
                        archive.extractall(target)
            else:
                asyncio.run(ensure_archive(url, target, sha256=expected_hash))

        training = cache_dir / "train" / "training_data.csv"
        test = sorted((cache_dir / "test" / "test").glob("test_*.csv"))
        validation = sorted((cache_dir / "validation" / "val").glob("val_*.csv"))
        if not training.is_file() or not test or not validation:
            raise TimeNetConnectorError("PHM archives lack the expected CSV files")
        return [
            PHMSource("train", training),
            *(PHMSource("test", path) for path in test),
            *(PHMSource("validation", path) for path in validation),
        ]

    def convert(self, raw_refs: list[PHMSource]) -> TimeFDataset:  # ruff: ignore[too-many-branches] - source validation
        """Create labeled training windows and unlabeled official inference records."""
        import pandas as pd  # ruff: ignore[import-outside-top-level] - isolated connector dependency

        dataset = TimeFDataset(metadata=self.metadata())
        for ref in raw_refs:
            frame = pd.read_csv(ref.path)
            cycle_col = "Cycles_Since_New" if ref.partition == "train" else "Cycles"
            required = {"ESN", cycle_col, "Snapshot", *SENSORS}
            if ref.partition == "train":
                required.update(TARGETS)
            missing = required.difference(frame.columns)
            if missing:
                raise TimeNetConnectorError(f"{ref.path}: missing {sorted(missing)}")
            frame = frame.sort_values(["ESN", cycle_col, "Snapshot"]).reset_index(drop=True)
            if frame.duplicated(["ESN", cycle_col, "Snapshot"]).any():
                raise TimeNetConnectorError(f"{ref.path}: duplicate snapshot key")
            if ref.partition == "train":
                for esn, engine_group in frame.groupby("ESN", sort=True):
                    engine = engine_group.reset_index(drop=True)
                    cycles = engine[cycle_col].drop_duplicates().to_numpy()
                    if len(cycles) < WINDOW_POSITIONS or any(cycles[1:] - cycles[:-1] != CYCLE_STEP):
                        raise TimeNetConnectorError(f"{ref.path}: engine {esn} has an unexpected cycle grid")
                    if engine[list(TARGETS)].isna().any().any():
                        raise TimeNetConnectorError(f"{ref.path}: engine {esn} has missing labels")
                    if (engine.groupby(cycle_col)[list(TARGETS)].nunique() != 1).any().any():
                        raise TimeNetConnectorError(f"{ref.path}: engine {esn} has conflicting labels")
                    for position in range(0, len(cycles) - WINDOW_POSITIONS + 1, WINDOW_STRIDE):
                        start_cycle = int(cycles[position])
                        end_cycle = int(cycles[position + WINDOW_POSITIONS - 1])
                        start = int(engine[cycle_col].searchsorted(start_cycle, side="left"))
                        stop = int(engine[cycle_col].searchsorted(end_cycle, side="right"))
                        record = dataset.add_record(
                            record=_record(
                                _Window(
                                    frame=engine,
                                    start=start,
                                    stop=stop,
                                    cycle_col=cycle_col,
                                    cycle_start=start_cycle,
                                    metadata={
                                        "source_partition": "train",
                                        "esn": int(str(esn)),
                                        "start_cycle_since_new": start_cycle,
                                        "end_cycle_since_new": end_cycle,
                                    },
                                )
                            )
                        )
                        labels = engine.iloc[stop - 1]
                        for target in TARGETS:
                            dataset.add_task(
                                task=ScalarPredictionTask(
                                    inputs=(record,),
                                    targets=(float(labels[target]),),
                                    target_name=target,
                                    unit="cycle",
                                    split=Split.TRAIN,
                                )
                            )
            else:
                if frame.ESN.nunique() != 1:
                    raise TimeNetConnectorError(f"{ref.path}: expected one engine per inference file")
                cycles = frame[cycle_col].drop_duplicates().to_numpy()
                if len(cycles) != WINDOW_POSITIONS or cycles.tolist() != list(
                    range(0, WINDOW_POSITIONS * CYCLE_STEP, CYCLE_STEP)
                ):
                    raise TimeNetConnectorError(f"{ref.path}: unexpected inference cycle grid")
                dataset.add_record(
                    record=_record(
                        _Window(
                            frame=frame,
                            start=0,
                            stop=len(frame),
                            cycle_col=cycle_col,
                            cycle_start=0,
                            metadata={
                                "source_partition": ref.partition,
                                "esn": int(frame.ESN.iloc[0]),
                                "source_file": ref.path.stem,
                            },
                        )
                    )
                )
        return dataset


CONNECTOR = PHM2025EHMConnector
