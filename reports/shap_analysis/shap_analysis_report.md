# SHAP Feature Importance Analysis

- Generated: `2026-05-20T17:24:08.727903`
- Method: `TreeExplainer` (LightGBM native)
- Models: tuned LightGBM for UberX, Uber Comfort, Uber Black

## UberX

Top 15 features by mean |SHAP value|:

|   CategoryID | CategoryName   | Feature                             |   MeanAbsSHAP |   StdAbsSHAP |   Rank |
|-------------:|:---------------|:------------------------------------|--------------:|-------------:|-------:|
|            2 | UberX          | numeric__Price_Comfort              |     11.2192   |    11.0295   |      1 |
|            2 | UberX          | numeric__Price_Black                |      3.87639  |     4.40704  |      2 |
|            2 | UberX          | numeric__UserPriorCategoryPriceMean |      1.61851  |     1.70729  |      3 |
|            2 | UberX          | numeric__ProductProviderID          |      1.31185  |     1.35333  |      4 |
|            2 | UberX          | numeric__DestinationLng             |      1.15817  |     1.22607  |      5 |
|            2 | UberX          | numeric__OriginLat                  |      0.881615 |     1.00377  |      6 |
|            2 | UberX          | numeric__OriginLng                  |      0.833939 |     0.874391 |      7 |
|            2 | UberX          | numeric__DestinationLat             |      0.779322 |     1.18846  |      8 |
|            2 | UberX          | numeric__ScheduleHour               |      0.762162 |     0.55415  |      9 |
|            2 | UberX          | numeric__FareIDWasImputed           |      0.599055 |     0.289113 |     10 |
|            2 | UberX          | numeric__WaitingTime                |      0.355218 |     0.53994  |     11 |
|            2 | UberX          | numeric__CompanyID                  |      0.348024 |     0.348985 |     12 |
|            2 | UberX          | numeric__ScheduleMonth              |      0.319058 |     0.378313 |     13 |
|            2 | UberX          | numeric__UserPriorPaidPriceMean     |      0.29967  |     0.48372  |     14 |
|            2 | UberX          | numeric__CreateHour                 |      0.284211 |     0.246701 |     15 |

## Uber Black

Top 15 features by mean |SHAP value|:

|   CategoryID | CategoryName   | Feature                             |   MeanAbsSHAP |   StdAbsSHAP |   Rank |
|-------------:|:---------------|:------------------------------------|--------------:|-------------:|-------:|
|            4 | Uber Black     | numeric__Price_Comfort              |     15.2952   |    10.355    |      1 |
|            4 | Uber Black     | numeric__Price_UberX                |     13.6887   |    10.658    |      2 |
|            4 | Uber Black     | numeric__DestinationLng             |      2.87702  |     1.77986  |      3 |
|            4 | Uber Black     | numeric__DestinationLat             |      2.20269  |     1.18877  |      4 |
|            4 | Uber Black     | numeric__OriginLat                  |      1.9519   |     1.45128  |      5 |
|            4 | Uber Black     | numeric__OriginLng                  |      1.93001  |     1.77553  |      6 |
|            4 | Uber Black     | numeric__UserPriorCategoryPriceMean |      1.46345  |     1.74061  |      7 |
|            4 | Uber Black     | numeric__ScheduleHour               |      1.0041   |     0.718427 |      8 |
|            4 | Uber Black     | numeric__CreateHour                 |      0.641368 |     0.469767 |      9 |
|            4 | Uber Black     | numeric__ScheduleMonth              |      0.582981 |     0.804206 |     10 |
|            4 | Uber Black     | numeric__ProductProviderID          |      0.332704 |     1.06745  |     11 |
|            4 | Uber Black     | numeric__UserPriorCategoryRideCount |      0.315497 |     0.570146 |     12 |
|            4 | Uber Black     | numeric__CompanyID                  |      0.230308 |     0.363968 |     13 |
|            4 | Uber Black     | numeric__UserPriorPaidPriceMean     |      0.178901 |     0.231961 |     14 |
|            4 | Uber Black     | numeric__ScheduleDayOfWeek          |      0.154254 |     0.238785 |     15 |

## Uber Comfort

Top 15 features by mean |SHAP value|:

|   CategoryID | CategoryName   | Feature                             |   MeanAbsSHAP |   StdAbsSHAP |   Rank |
|-------------:|:---------------|:------------------------------------|--------------:|-------------:|-------:|
|            9 | Uber Comfort   | numeric__Price_UberX                |     22.9079   |    21.438    |      1 |
|            9 | Uber Comfort   | numeric__Price_Black                |      2.46802  |     2.44837  |      2 |
|            9 | Uber Comfort   | numeric__UserPriorCategoryPriceMean |      1.34535  |     1.08826  |      3 |
|            9 | Uber Comfort   | numeric__OriginLat                  |      1.03726  |     1.12264  |      4 |
|            9 | Uber Comfort   | numeric__OriginLng                  |      0.865458 |     0.674216 |      5 |
|            9 | Uber Comfort   | numeric__DestinationLat             |      0.701544 |     0.706596 |      6 |
|            9 | Uber Comfort   | numeric__ScheduleMonth              |      0.563053 |     0.547158 |      7 |
|            9 | Uber Comfort   | numeric__UserPriorPaidPriceMean     |      0.500541 |     0.441731 |      8 |
|            9 | Uber Comfort   | categorical__ProductID_Bag          |      0.439601 |     0.426929 |      9 |
|            9 | Uber Comfort   | categorical__ProductID_Comfort      |      0.42562  |     0.675856 |     10 |
|            9 | Uber Comfort   | numeric__DestinationLng             |      0.387546 |     0.434002 |     11 |
|            9 | Uber Comfort   | numeric__ProductProviderID          |      0.339915 |     0.775564 |     12 |
|            9 | Uber Comfort   | numeric__WaitingTime                |      0.32965  |     0.30327  |     13 |
|            9 | Uber Comfort   | numeric__CompanyID                  |      0.286716 |     0.293974 |     14 |
|            9 | Uber Comfort   | numeric__ScheduleHour               |      0.21253  |     0.297804 |     15 |

## Interpretation

- SHAP values measure each feature's contribution to the prediction relative to the base value.
- Positive SHAP = pushes price up; Negative SHAP = pushes price down.
- Mean |SHAP| ranks features by overall impact magnitude.
- Cross-category price features (e.g., Price_Comfort in UberX model) capture relative pricing signals.
- User history features (UserPrior*) capture customer behavior patterns.
- Temporal features (Create*) capture demand seasonality and surge pricing windows.