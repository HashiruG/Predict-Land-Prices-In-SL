# Predicting Land Prices in Sri Lanka with Ensemble Learning

Machine Learning and Pattern Recognition - hands-on project.

I collected 5,001 "Land for Sale" ads from ikman.lk (Colombo, Gampaha and Kalutara), cleaned them to 2,719 unique plots,
built location, size and keyword features, and compared **bagging, boosting and stacking** models for predicting the
**price per perch**. I also compared a tuned Random Forest with XGBoost and tried three Random Forest + neural network hybrids.
The results are shown in a Streamlit dashboard.

## How to run (Windows)

Double-click `run_all.bat`, or run these commands in this folder:

```
python -m pip install -r requirements.txt
python -m pytest -q
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebook\Notebook.ipynb
python -m streamlit run app\app.py
```

The notebook must be run once before the dashboard, because it creates the cleaned data, the results and the trained
models (about 10 minutes). After that, `run_dashboard.bat` opens the dashboard directly.
I used Python 3.14; all random seeds are 42, so the results are the same on every run.

## Folders

- `notebook/Notebook.ipynb` - the full analysis with outputs and explanations
- `app/app.py` - Streamlit dashboard (7 tabs)
- `src/features.py` - cleaning and feature engineering (used by both the notebook and the dashboard)
- `scraper/scrape_ikman.py` - collects the ads from ikman.lk (1.5 s between requests, public pages only)
- `tests/test_features.py` - unit tests for the cleaning and feature code
- `data/raw_listings.csv` - the ads I collected on 23 Nov 2026
- `data/area_coordinates.csv` - approximate centre of each area (made by hand)

The raw data is included because ikman ads change every day, so scraping again would give different results.

## Main results

| Model | CV R² | Test R² | Test MAPE |
|---|---|---|---|
| Stacking (RF + ET + XGBoost + CatBoost -> Ridge) (best) | 0.847 | 0.844 | 44.4% |
| Gradient Boosting | 0.846 | 0.844 | 44.4% |
| Random Forest | 0.839 | 0.832 | 45.9% |
| Ridge regression (baseline) | 0.835 | 0.823 | 48.0% |
| Single decision tree | 0.746 | 0.713 | 64.6% |
| Random Forest (tuned) vs XGBoost (tuned) | 0.849 vs 0.849 | 0.838 vs 0.843 | - |
| ANN alone / best RF + ANN hybrid | - | 0.773 / 0.834 | - |

Models are ranked by 5-fold cross-validation R² on the training set. The full table and explanations are in the notebook.

## Limitations

The prices are asking prices, not sale prices; I only know the town of each plot, not its exact location; and the data is
one snapshot in time.
