from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train import (  # noqa: E402
    TemporalSplitConfig,
    get_holdout_split,
    iter_tscv_splits,
    load_temporal_frame,
)


class TestTSCVTemporalOrder(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = TemporalSplitConfig()
        cls.frame = load_temporal_frame(cls.config)

    def assert_temporal_integrity(
        self,
        split_name: str,
        train_indices,
        evaluation_indices,
    ) -> None:
        self.assertGreater(len(train_indices), 0, f"{split_name}: treino vazio")
        self.assertGreater(len(evaluation_indices), 0, f"{split_name}: validacao vazia")

        train_frame = self.frame.loc[train_indices]
        evaluation_frame = self.frame.loc[evaluation_indices]

        train_min_date = pd.Timestamp(train_frame["CreateDate"].min())
        train_max_date = pd.Timestamp(train_frame["CreateDate"].max())
        evaluation_min_date = pd.Timestamp(evaluation_frame["CreateDate"].min())
        evaluation_max_date = pd.Timestamp(evaluation_frame["CreateDate"].max())

        self.assertLess(
            train_max_date,
            evaluation_min_date,
            (
                f"{split_name}: ordem temporal invalida "
                f"(train_max={train_max_date.date()} >= eval_min={evaluation_min_date.date()})"
            ),
        )
        self.assertLessEqual(
            train_min_date,
            train_max_date,
            f"{split_name}: intervalo de treino invalido",
        )
        self.assertLessEqual(
            evaluation_min_date,
            evaluation_max_date,
            f"{split_name}: intervalo de validacao invalido",
        )

        index_overlap = pd.Index(train_indices).intersection(pd.Index(evaluation_indices))
        self.assertEqual(
            len(index_overlap),
            0,
            f"{split_name}: existem indices compartilhados entre treino e validacao",
        )

        train_rides = pd.Index(train_frame[self.config.ride_column].astype("int64"))
        evaluation_rides = pd.Index(
            evaluation_frame[self.config.ride_column].astype("int64")
        )
        ride_overlap = train_rides.intersection(evaluation_rides)
        self.assertEqual(
            len(ride_overlap),
            0,
            f"{split_name}: existem RideIDs compartilhados entre treino e validacao",
        )

    def test_cv_folds_temporal_order_overall(self) -> None:
        for split_window, train_indices, evaluation_indices in iter_tscv_splits(
            self.frame,
            self.config,
        ):
            if split_window.gap_start is not None and split_window.gap_end is not None:
                self.assertGreater(
                    split_window.gap_start,
                    split_window.train_end,
                    f"{split_window.name}: gap_start deve ser maior que train_end",
                )
                self.assertGreater(
                    split_window.evaluation_start,
                    split_window.gap_end,
                    f"{split_window.name}: evaluation_start deve ser maior que gap_end",
                )
            self.assert_temporal_integrity(
                split_window.name,
                train_indices,
                evaluation_indices,
            )

    def test_cv_folds_temporal_order_target_categories(self) -> None:
        for category_id in self.config.target_categories:
            for split_window, train_indices, evaluation_indices in iter_tscv_splits(
                self.frame,
                self.config,
                category_id=category_id,
            ):
                split_name = f"{split_window.name}_category_{category_id}"
                self.assert_temporal_integrity(
                    split_name,
                    train_indices,
                    evaluation_indices,
                )

    def test_holdout_temporal_order_overall(self) -> None:
        split_window, train_indices, evaluation_indices = get_holdout_split(
            self.frame,
            self.config,
        )
        if split_window.gap_start is not None and split_window.gap_end is not None:
            self.assertGreater(
                split_window.gap_start,
                split_window.train_end,
                "holdout: gap_start deve ser maior que train_end",
            )
            self.assertGreater(
                split_window.evaluation_start,
                split_window.gap_end,
                "holdout: evaluation_start deve ser maior que gap_end",
            )
        self.assert_temporal_integrity(
            split_window.name,
            train_indices,
            evaluation_indices,
        )


if __name__ == "__main__":
    unittest.main()
