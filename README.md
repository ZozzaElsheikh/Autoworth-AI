# AutoWorth AI 🚗

A machine learning system that predicts the fair market price of a used car and rates whether a seller's asking price is a good deal — trained on the 100,000 UK Used Car Dataset.

**[Live demo screenshot / GIF here once you have one]**

---

## Problem

Given a car's make, model, year, mileage, transmission, fuel type, and engine size, AutoWorth AI predicts what it should reasonably sell for — then compares that prediction to a seller's actual asking price and rates the deal on a 5-tier scale, from 🟢 Great Deal to 🔴 Overpriced.

## Results

| Model | Val R² | Val MAE |
|---|---|---|
| Linear Regression | 0.863 | £2,232 |
| Decision Tree | 0.934 (overfit — see notebook) | £1,466 |
| Gradient Boosting (tuned) | 0.945 | £1,501 |
| Neural Network (bonus) | 0.948 | £1,338 |
| **Random Forest (final)** | **0.957** | **£1,169** |

**Final test set performance (Random Forest):** R² = 0.9572, MAE = £1,162, RMSE = £2,075

## Pipeline

1. **Data loading & combining** — 9 manufacturer datasets merged, tagged by brand
2. **Cleaning** — investigated and resolved impossible values (bad years, zero engine sizes, corrupted mpg readings, a hidden whitespace bug, mislabeled Mercedes trims) rather than blindly dropping them
3. **EDA** — 6 visualizations uncovering non-linear price/mileage relationships, brand-tier pricing, and feature correlations
4. **Feature engineering** — `car_age`, `mileage_per_year`, `is_premium_make`
5. **Preprocessing** — one-hot encoding + scaling, fit only on training data to prevent leakage
6. **Modeling** — 4 required regression models + a bonus neural network, compared on R²/MAE/RMSE
7. **Tuning** — diagnosed Gradient Boosting as underfitting, tuned it toward more capacity (Val R² 0.905 → 0.945)
8. **Model selection** — Random Forest chosen based on metrics, not narrative, even though its tuning story was less dramatic
9. **Model understanding** — feature importance, Actual vs Predicted, learning curve, and an investigation into the model's worst predictions (it struggles most with rare/high-end vehicles — a documented, honest limitation)
10. **Smart Deal Advisor** — turns the price prediction into an actionable buy/pass signal
11. **Streamlit app** — a working demo UI for the whole pipeline

## Key Findings

- Engine size was the strongest linear predictor of price, but transmission type (manual vs. automatic) turned out to be the single most important feature for the final model — capturing a non-linear price split that correlation analysis alone missed.
- Year and mileage were highly correlated (-0.74), consistent with tree-based models outperforming Linear Regression.
- The model's largest errors cluster around rare, high-performance vehicles (supercars, low-volume trims) — a known limitation of tree ensembles extrapolating beyond dense regions of their training data.

## Tech Stack

Python, pandas, scikit-learn, TensorFlow/Keras, Streamlit, matplotlib/seaborn

## Project Structure

```
autoworth-ai/
├── notebooks/
│   └── autoworth_ai.ipynb      # full pipeline: cleaning → EDA → modeling → evaluation
├── app/
│   ├── app.py                  # Streamlit application
│   ├── model.pkl / scaler.pkl / encoder.pkl
│   └── app_metadata.json
├── requirements.txt
└── README.md
```

## Running Locally

```bash
git clone https://github.com/<your-username>/autoworth-ai.git
cd autoworth-ai
pip install -r requirements.txt

# To retrain / explore the pipeline:
jupyter notebook notebooks/autoworth_ai.ipynb

# To run the app:
cd app
streamlit run app.py
```

## Dataset

[100,000 UK Used Car Dataset](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes) (Kaggle) — Audi, BMW, Ford, Hyundai, Mercedes, Skoda, Toyota, Vauxhall, Volkswagen listings.

## Limitations & Future Work

- No trim-level feature in the source data — the model can't distinguish, e.g., a standard Ford Focus from a Focus RS, which shows up in its worst predictions.
- Learning curve suggests the model hasn't fully saturated — more training data would likely improve it further.
- Random Forest wasn't hyperparameter-tuned (Gradient Boosting was, as a deliberate diagnostic exercise) — tuning RF too would allow a fairer head-to-head comparison.
