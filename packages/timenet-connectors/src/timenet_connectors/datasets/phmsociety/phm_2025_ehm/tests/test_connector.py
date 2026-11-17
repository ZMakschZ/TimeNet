"""Exercise PHM conversion with synthetic source-shaped snapshots."""

from pathlib import Path

import numpy as np
import pytest

from timenet.types import ScalarPredictionTask
from timenet_connectors.datasets.phmsociety.phm_2025_ehm import connector


def test_training_window_is_causal_and_preserves_missing_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(connector, "WINDOW_POSITIONS", 3)
    source = Path(__file__).parent / "fixtures" / "training_data.csv"
    dataset = connector.PHM2025EHMConnector().convert([connector.PHMSource("train", source)])

    assert len(dataset.records) == 1
    assert len(dataset.tasks) == 3
    record = dataset.records[0]
    assert record.metadata["end_cycle_since_new"] == 20
    signals = {signal.name: signal for signal in record.signals}
    np.testing.assert_array_equal(signals["Cycle"].to_numpy(), [0, 0, 10, 10, 20, 20])
    assert signals["Sensed_T45"].to_arrow().null_count == 1
    assert [task.target_name for task in dataset.tasks if isinstance(task, ScalarPredictionTask)] == list(
        connector.TARGETS
    )
    assert [task.targets for task in dataset.tasks] == [(10.0,), (280.0,), (480.0,)]
