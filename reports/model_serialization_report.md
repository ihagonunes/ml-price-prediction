# Model Serialization Report

- Generated: `2026-05-20T17:11:55.787711`
- Framework: `joblib` with compression level 3
- Model type: LightGBM (tuned via Optuna)

## Serialized Models

|   CategoryID | CategoryName   | ModelFile                                                                                                      | ModelType        |   NumBoostRound |   TrainingSamples |   FeatureCount |   TrainingMAE |   TrainingRMSE |   TrainingMAPE |   TrainingR2 |   ValidationSampleMAE | SerializationOK   |
|-------------:|:---------------|:---------------------------------------------------------------------------------------------------------------|:-----------------|----------------:|------------------:|---------------:|--------------:|---------------:|---------------:|-------------:|----------------------:|:------------------|
|            2 | UberX          | C:\Users\ihan.nunes\OneDrive - ESPM\Documentos\GitMartech\ml-price-prediction\models\model_uberx.joblib        | LightGBM (tuned) |             350 |             94749 |             35 |       4.41448 |        8.52399 |       14.8867  |     0.901352 |               9.11829 | True              |
|            4 | Uber Black     | C:\Users\ihan.nunes\OneDrive - ESPM\Documentos\GitMartech\ml-price-prediction\models\model_uber_black.joblib   | LightGBM (tuned) |             350 |             35235 |             30 |       4.52756 |        7.88057 |        8.88846 |     0.958724 |              18.5368  | True              |
|            9 | Uber Comfort   | C:\Users\ihan.nunes\OneDrive - ESPM\Documentos\GitMartech\ml-price-prediction\models\model_uber_comfort.joblib | LightGBM (tuned) |             300 |             64586 |             31 |       2.61961 |        4.80012 |        5.80724 |     0.982938 |               4.90045 | True              |

## How to Load and Use

```python
import joblib
import pandas as pd

bundle = joblib.load('models/model_uberx.joblib')
preprocessor = bundle['preprocessor']
model = bundle['model']

x_input = preprocessor.transform(input_dataframe)
predictions = model.predict(x_input)
```

## Input Requirements

The input DataFrame must contain the same columns used during training.
See the feature lists in `src/train.py` (NUMERIC_FEATURE_COLUMNS,
BOOLEAN_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS) and the
auxiliary cross-price columns defined in CATEGORY_DATASETS.

## Validation

Each model was loaded after serialization and tested with a sample
of the training data to verify prediction capability.