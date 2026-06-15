from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class FinalLightGBMModel:
    category_id: int
    category_name: str
    model_name: str
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    preprocessor: Any
    booster: Any
    metadata: dict[str, Any]

    def predict(self, frame: pd.DataFrame) -> list[float]:
        missing_columns = sorted(set(self.feature_columns).difference(frame.columns))
        if missing_columns:
            raise ValueError(f"Colunas ausentes para predicao: {missing_columns}")

        prepared_frame = frame[self.feature_columns].copy()
        transformed_frame = self.preprocessor.transform(prepared_frame)
        predictions = self.booster.predict(transformed_frame)
        return [float(value) for value in predictions]
