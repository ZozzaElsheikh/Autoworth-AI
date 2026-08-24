# Deal Advisor — Streamlit App

UK used car price predictor + deal rating, built on the Random Forest model
from `ML-Final-Project.ipynb`.

## Setup

1. Put these 4 files in the **same folder** as `app.py`:
   - `model.pkl`   — `joblib.dump(rf, 'model.pkl')` (already saved by your notebook, cell 52)
   - `scaler.pkl`  — same folder, from cell 52
   - `encoder.pkl` — same folder, from cell 52
   - `app_metadata.json` — from cell 53 (run it if you haven't yet)

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run it:
   ```
   streamlit run app.py
   ```

## How it works

- Sidebar form collects the same raw fields your model was trained on:
  make, model, transmission, fuel type, year, mileage, tax, mpg, engine size.
- `car_age`, `mileage_per_year`, and `is_premium_make` are engineered
  identically to the notebook (reference year 2020, premium = Audi/BMW/Mercedes).
- Numerical features go through the saved `StandardScaler`, categoricals
  through the saved `OneHotEncoder` (`handle_unknown='ignore'`), then
  `np.hstack([X_num, X_cat, is_premium])` — the exact order used in training
  (verified: 7 numerical + 208 one-hot columns + 1 flag = 216 features).
- The Random Forest predicts a market price; `deal_advisor()` compares it to
  the seller's asking price using your original 5-tier logic (Great / Good /
  Fair / Slightly Overpriced / Overpriced).
- A semicircular gauge (SVG, no external chart lib) sweeps a needle across
  the 5 color-coded zones based on % difference, clamped to ±15% for display.

## Notes

- If `model.pkl` isn't next to `app.py`, the app shows a clear error listing
  which file(s) are missing rather than crashing.
- The Model dropdown is filtered to only the models that actually belong to
  the selected Make, using `models_by_make` from `app_metadata.json`.
