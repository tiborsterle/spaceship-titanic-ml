# Spaceship Titanic – Passenger Transport Prediction

Binary classification on the [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) Kaggle competition: predict whether a passenger was transported to an alternate dimension. We compare three models on a shared preprocessing pipeline: **Random Forest**, **XGBoost** and a **PyTorch MLP**.

Final project for the course *Introduction to Artificial Intelligence* at the Warsaw University of Technology (Erasmus+, 2026).

## Results

| Model | Val accuracy | F1 | ROC-AUC | Kaggle test accuracy |
|---|---|---|---|---|
| Random Forest | 0.807 | 0.805 | 0.900 | 0.793 |
| XGBoost | 0.808 | 0.807 | **0.910** | 0.796 |
| **MLP (PyTorch)** | **0.812** | **0.813** | 0.909 | **0.806** |

The MLP scored best on both validation and the official Kaggle test set. All three models are within about one percentage point of each other, and the drop from validation to test accuracy was only 1–1.5 points. For comparison, top public submissions reach about 0.82–0.83.

> Note: Random Forest and XGBoost are evaluated on a 20% stratified validation split. The MLP uses a 10% split, which also serves as the early stopping signal.

## Approach

### Preprocessing (shared by all models)
- **Feature engineering:** `Cabin` (format `deck/number/side`) is split into `Deck`, `CabinNum` and `Side`, and the five amenity columns are summed into `TotalSpending`. `PassengerId` and `Name` are dropped.
- **Missing values:** median for numerical columns and mode for categorical columns, computed on the training set only to avoid data leakage.
- **Encoding:** categorical and boolean columns are label-encoded.
- **Scaling:** `StandardScaler` for the MLP only.

### Models
- **Random Forest:** 500 trees, `max_depth=20`, `min_samples_leaf=2`, chosen with a grid search over `n_estimators` ∈ {200, 300, 500} and `max_depth` ∈ {15, 20, None}.
- **XGBoost:** 500 trees, `max_depth=8`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`.
- **MLP:** `BatchNorm1d` → 3 hidden layers (256, 128, 64) with ReLU and Dropout 0.3 → sigmoid output. Trained with Adam and binary cross-entropy, `ReduceLROnPlateau` and early stopping (patience 15).

### Key findings
- **CryoSleep** is the strongest predictor: about 83% of passengers in cryo-sleep were transported.
- **Zero spending** almost always means transported, because passengers in cryo-sleep don't spend anything on board.
- **HomePlanet** matters too: Europa ~67% transported vs. Earth ~43%.

## Project structure

```
├── src/
│   ├── RandomForest.py
│   ├── XGBoost.py
│   └── MLP.py
├── dataset/          # train.csv and test.csv from Kaggle (not included)
├── results/          # generated: metrics, figures, Kaggle submissions
├── docs/
│   └── Spaceship_Titanic_Final_Documentation.pdf
└── requirements.txt
```

## How to run

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download `train.csv` and `test.csv` from the [competition page](https://www.kaggle.com/competitions/spaceship-titanic/data) and put them in `dataset/`.
3. Run any of the models:
   ```bash
   python src/RandomForest.py
   python src/XGBoost.py
   python src/MLP.py
   ```

Each script writes its metrics, plots (confusion matrix, ROC curve and feature importance or loss curve) and a Kaggle submission file to `results/<model>/`.

## Possible improvements
- Model stacking
- A broader hyperparameter search for the MLP
- Group-level features from the passenger group ID inside `PassengerId`

## Documentation
The full project report is in [`docs/Spaceship_Titanic_Final_Documentation.pdf`](docs/Spaceship_Titanic_Final_Documentation.pdf).

## Authors
- Tibor Sterle
- Francesco Gaibazzi
