"""Offset-aware timing model utilities.

The trained neural network predicts the energy-dependent time walk. The wrapper
in this module estimates run-specific detector offsets from the residual timing
after that energy correction.
"""

from __future__ import annotations

from typing import Any

import numpy as np


FEATURE_COLUMNS = ["Ei", "index_i", "Ej", "index_j"]


class OffsetAwareTimingModel:
    def __init__(
        self,
        time_walk_model: Any,
        detector_count: int = 18,
        feature_columns: list[str] | None = None,
        default_offset_range: tuple[float, float] = (-20.0, 20.0),
        default_min_pair_count: int = 100,
    ) -> None:
        self.time_walk_model = time_walk_model
        self.detector_count = detector_count
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.default_offset_range = default_offset_range
        self.default_min_pair_count = default_min_pair_count

    def predict(self, features: Any, **kwargs: Any) -> np.ndarray:
        return self.predict_time_walk(features, **kwargs)

    def predict_time_walk(self, features: Any, **kwargs: Any) -> np.ndarray:
        features = self._select_features(features)
        prediction_attempts = (
            {"regressor__verbose": 0},
            {"verbose": 0},
            kwargs,
            {},
        )
        last_error = None
        for predict_kwargs in prediction_attempts:
            try:
                predictions = self.time_walk_model.predict(features, **predict_kwargs)
                return np.asarray(predictions, dtype=float).reshape(-1)
            except TypeError as exc:
                last_error = exc
        raise last_error

    def model_corrected_tdiff(self, pair_data: Any, target_column: str = "T_Diff") -> np.ndarray:
        observed = np.asarray(pair_data[target_column], dtype=float)
        return observed - self.predict_time_walk(pair_data)

    def pair_offset_correction(self, index_i: np.ndarray, index_j: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        return offsets[index_i] - offsets[index_j]

    def final_corrected_tdiff(
        self,
        pair_data: Any,
        offsets: np.ndarray,
        target_column: str = "T_Diff",
    ) -> np.ndarray:
        index_i = np.asarray(pair_data["index_i"], dtype=np.intp)
        index_j = np.asarray(pair_data["index_j"], dtype=np.intp)
        return self.model_corrected_tdiff(pair_data, target_column) - self.pair_offset_correction(
            index_i,
            index_j,
            offsets,
        )

    def empty_offset_statistics(self) -> tuple[np.ndarray, np.ndarray]:
        shape = (self.detector_count, self.detector_count)
        return np.zeros(shape, dtype=float), np.zeros(shape, dtype=float)

    def accumulate_offset_statistics(
        self,
        pair_data: Any,
        pair_sum: np.ndarray,
        pair_counts: np.ndarray,
        target_column: str = "T_Diff",
        offset_range: tuple[float, float] | None = None,
    ) -> None:
        offset_range = offset_range or self.default_offset_range
        residual = self.model_corrected_tdiff(pair_data, target_column)
        index_i = np.asarray(pair_data["index_i"], dtype=np.intp)
        index_j = np.asarray(pair_data["index_j"], dtype=np.intp)
        mask = (
            (index_i != index_j)
            & np.isfinite(residual)
            & (offset_range[0] <= residual)
            & (residual <= offset_range[1])
        )
        if not np.any(mask):
            return
        np.add.at(pair_sum, (index_i[mask], index_j[mask]), residual[mask])
        np.add.at(pair_counts, (index_i[mask], index_j[mask]), 1.0)

    def solve_detector_offsets(
        self,
        pair_sum: np.ndarray,
        pair_counts: np.ndarray,
        min_pair_count: int | None = None,
    ) -> np.ndarray:
        min_pair_count = min_pair_count or self.default_min_pair_count
        rows: list[np.ndarray] = []
        targets: list[float] = []
        weights: list[float] = []
        for index_i in range(self.detector_count):
            for index_j in range(self.detector_count):
                count = pair_counts[index_i, index_j]
                if index_i == index_j or count < min_pair_count:
                    continue
                row = np.zeros(self.detector_count, dtype=float)
                row[index_i] = 1.0
                row[index_j] = -1.0
                rows.append(row)
                targets.append(pair_sum[index_i, index_j] / count)
                weights.append(np.sqrt(count))

        if not rows:
            return np.zeros(self.detector_count, dtype=float)

        matrix = np.vstack(rows)
        target = np.asarray(targets, dtype=float)
        weight = np.asarray(weights, dtype=float)
        offsets, *_ = np.linalg.lstsq(matrix * weight[:, np.newaxis], target * weight, rcond=None)
        active = np.flatnonzero(
            np.sum(pair_counts >= min_pair_count, axis=0)
            + np.sum(pair_counts >= min_pair_count, axis=1)
        )
        if active.size:
            offsets[active] -= float(np.mean(offsets[active]))
        return offsets

    def estimate_detector_offsets(
        self,
        pair_data: Any,
        target_column: str = "T_Diff",
        offset_range: tuple[float, float] | None = None,
        min_pair_count: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pair_sum, pair_counts = self.empty_offset_statistics()
        self.accumulate_offset_statistics(
            pair_data,
            pair_sum,
            pair_counts,
            target_column=target_column,
            offset_range=offset_range,
        )
        offsets = self.solve_detector_offsets(pair_sum, pair_counts, min_pair_count)
        pair_means = np.divide(
            pair_sum,
            pair_counts,
            out=np.zeros_like(pair_sum),
            where=pair_counts > 0,
        )
        return offsets, pair_counts, pair_means

    def _select_features(self, features: Any) -> Any:
        if hasattr(features, "columns"):
            return features[self.feature_columns]
        return features
