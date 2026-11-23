"""
Ensemble Learning Dashboard - Sri Lankan land prices (ikman.lk)

Run from the project folder:
    streamlit run app/app.py        (or double-click run_dashboard.bat)

Needs the notebook to have been run once (it creates data/clean_land.csv,
results/*.csv and app/models/*.joblib).

The tabs follow the order of the project (and of the video):
1 Data -> 2 Features -> 3 Ensemble lab -> 4 Model comparison -> 5 RF vs XGBoost -> 6 RF + ANN -> 7 Price estimator
"""
import json
import sys
import time
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import (CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET,  # noqa: E402
                      build_input_row)

st.set_page_config(page_title="Land Price Prediction - Ensemble Learning", layout="wide")

SEED = 42
MAIN = "#1F4E5F"      # one colour for everything ...
HIGHLIGHT = "#D1495B"  # ... and one to point at the thing I talk about
LIGHT = "#B8C4C9"

FEATURE_NAMES = {"dist_fort_km": "Distance to Colombo Fort", "dist_expressway_km": "Distance to expressway",
                 "lat": "Latitude", "lon": "Longitude", "is_colombo_city": "Colombo 1-15", "log_perches": "Land size (log)",
                 "is_large_land": "Large land (40+ P)", "price_was_total": "Price given as total",
                 "district": "District", "area": "Area", "is_member": "ikman member", "is_featured": "Featured ad",
                 "is_urgent_ad": "Urgent ad"}


def nice(f):
    return FEATURE_NAMES.get(f, f.replace("kw_", "Keyword: ").replace("_", " "))


def why(*points):
    """A small box with the reasons behind the results on this tab."""
    with st.container(border=True):
        st.markdown("**Why?**\n" + "\n".join(f"- {p}" for p in points))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(ROOT / "data" / "clean_land.csv")
    coords = pd.read_csv(ROOT / "data" / "area_coordinates.csv")
    res = {}
    for name in ["leaderboard", "rf_vs_xgb", "hybrid_rf_ann", "permutation_importance", "cleaning_log",
                 "stacking_weights", "keyword_effects", "ann_architecture"]:
        p = ROOT / "results" / f"{name}.csv"
        res[name] = pd.read_csv(p) if p.exists() else None
    if res["leaderboard"] is not None:   # ranked by cross-validation, never by the test set
        res["leaderboard"] = res["leaderboard"].sort_values("CV_R2_mean", ascending=False).reset_index(drop=True)
    sp = ROOT / "results" / "summary.json"
    summary = json.load(open(sp)) if sp.exists() else {}
    dp = ROOT / "results" / "duplicate_effect.json"
    summary["dup"] = json.load(open(dp)) if dp.exists() else None
    return df, coords, res, summary


@st.cache_resource
def load_models():
    p = ROOT / "app" / "models" / "final_models.joblib"
    return joblib.load(p) if p.exists() else {}


@st.cache_data
def split(df):
    from sklearn.model_selection import train_test_split
    X, y = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=SEED)


if not (ROOT / "data" / "clean_land.csv").exists():
    st.error("Run notebook/ensemble_land_prices.ipynb first - it creates the data, results and models this dashboard uses.")
    st.stop()

df, coords, res, summary = load_data()
models = load_models()
X_train, X_test, y_train, y_test = split(df)
lb = res["leaderboard"]


def rs(x):
    """Formats rupees: Rs 1.25M / Rs 850k"""
    return f"Rs {x/1e6:,.2f}M" if x >= 1e6 else f"Rs {x/1e3:,.0f}k"


def metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    t, p = np.exp(y_true), np.exp(y_pred)
    return {"R²": r2_score(y_true, y_pred), "RMSE (log)": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE (log)": mean_absolute_error(y_true, y_pred), "MAPE %": 100 * np.mean(np.abs(t - p) / t)}


LOG_AXIS = dict(scale=alt.Scale(type="log"), axis=alt.Axis(values=[1e4, 1e5, 1e6, 1e7, 1e8], format="~s"))

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Predicting Land Prices in Sri Lanka with Ensemble Learning")

tabs = st.tabs(["1. Data", "2. Features", "3. Ensemble lab", "4. Model comparison",
                "5. RF vs XGBoost", "6. RF + ANN", "7. Price estimator"])

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Ads collected", f"{summary.get('rows_scraped', 0):,}")
    c[1].metric("Unique plots after cleaning", f"{len(df):,}")
    c[2].metric("Areas", df["area"].nunique())
    c[3].metric("Median price per perch", rs(df["price_per_perch"].median()))

    area = (df.groupby(["area", "district"])
              .agg(lat=("lat", "first"), lon=("lon", "first"), ads=("price_per_perch", "size"),
                   median=("price_per_perch", "median")).reset_index())
    left, right = st.columns(2)
    with left:
        st.subheader("Where the ads are")
        area["size"] = 60 + area["ads"] * 7
        st.map(area, latitude="lat", longitude="lon", size="size", color="#1F4E5FAA", zoom=9, height=430)
    with right:
        st.subheader("Most and least expensive areas")
        enough = area[area["ads"] >= 10].sort_values("median", ascending=False)
        ends = pd.concat([enough.head(8).assign(group="Most expensive"), enough.tail(8).assign(group="Cheapest")])
        st.altair_chart(alt.Chart(ends).mark_bar().encode(
            x=alt.X("median:Q", title="Median Rs per perch (log scale)",
                    scale=alt.Scale(type="log", domain=[5e4, 5e7]), axis=alt.Axis(values=[1e5, 1e6, 1e7], format="~s")),
            x2=alt.datum(5e4),   # log scale has no zero, so the bars start at Rs 50k
            y=alt.Y("area:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320, labelOverlap=False)),
            color=alt.Color("group:N", scale=alt.Scale(domain=["Most expensive", "Cheapest"], range=[MAIN, LIGHT]),
                            legend=alt.Legend(title=None, orient="bottom")),
            tooltip=["area", "district", "ads", alt.Tooltip("median:Q", format=",.0f")]).properties(height=430),
            use_container_width=True)

    st.subheader("Cleaning")
    a, b = st.columns([1, 1])
    if res["cleaning_log"] is not None:
        cl = res["cleaning_log"].rename(columns={"step": "Step", "rows": "Rows left"})
        a.dataframe(cl, hide_index=True, use_container_width=True)
    with b:
        d = summary.get("dup")
        points = [f"Some agents posted land from Kurunegala and Ratnapura under Colombo pages, so I removed ads from "
                  f"outside the three districts, and took the area from the title when the page area was wrong "
                  f"({summary.get('area_corrected_from_title', '?')} ads)."]
        if d:
            points.append(f"The same plot was often posted many times. With the copies, Random Forest scored R² "
                          f"**{d['rf_r2_with_duplicates']:.3f}**; without them **{d['rf_r2_without']:.3f}**. The higher score "
                          "was **leakage** (the same plot in both train and test), so I removed duplicate plots.")
        why(*points)

# ---------------------------------------------------------------------------
# 2. Features
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Features I built from each ad")
    st.markdown("""
| Group | Features | Why |
|---|---|---|
| Target | log(price per perch) | Prices are very skewed; the log makes errors relative (10% is 10% for any plot) |
| Location | distance to Colombo Fort, distance to nearest expressway, latitude, longitude, Colombo 1-15 flag, area, district | Location is the main driver of land price |
| Size | log(perches), large land (40+ perches) | Big plots are cheaper per perch |
| Price text | price was given as a total | Total prices were converted to per perch |
| Title keywords (English + Sinhala) | house, commercial, residential, agricultural, bare land, main road, prime, plots, water view, urgent | The ad page has no land-type field, so I read it from the title |
| ikman flags | member, featured, urgent ad | Shown on the listing page |
""")

    a, b = st.columns(2)
    with a:
        st.subheader("Price vs distance to Colombo Fort")
        pts = df.sample(min(2500, len(df)), random_state=1)
        trend = (df.assign(band=(df["dist_fort_km"] // 2) * 2 + 1).groupby("band")
                   .agg(price=("price_per_perch", "median"), n=("price_per_perch", "size")).reset_index())
        trend = trend[trend["n"] >= 10]
        dots = alt.Chart(pts).mark_circle(size=12, opacity=0.25, color=LIGHT).encode(
            x=alt.X("dist_fort_km:Q", title="Distance to Colombo Fort (km)"),
            y=alt.Y("price_per_perch:Q", title="Rs per perch (log scale)", **LOG_AXIS),
            tooltip=["title", "area", alt.Tooltip("price_per_perch:Q", format=",.0f")])
        line = alt.Chart(trend).mark_line(color=MAIN, strokeWidth=3).encode(x="band:Q", y="price:Q")
        st.altair_chart((dots + line).properties(height=330), use_container_width=True)
    with b:
        st.subheader("Effect of title keywords")
        kw = res["keyword_effects"]
        if kw is not None:
            kw = kw.rename(columns={kw.columns[0]: "keyword", kw.columns[1]: "effect"})
            kw["keyword"] = kw["keyword"].str.replace("kw_", "").str.replace("_", " ")
            kw["direction"] = np.where(kw["effect"] > 0, "Higher price", "Lower price")
            st.altair_chart(alt.Chart(kw).mark_bar().encode(
                x=alt.X("effect:Q", title="Difference in median log price (0.1 ~ 10%)", axis=alt.Axis(tickCount=5)),
                y=alt.Y("keyword:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
                color=alt.Color("direction:N", scale=alt.Scale(domain=["Higher price", "Lower price"], range=[MAIN, HIGHLIGHT]),
                                legend=alt.Legend(title=None, orient="bottom"))).properties(height=330),
                use_container_width=True)

    pi = res["permutation_importance"]
    top3 = ""
    if pi is not None:
        pi = pi.rename(columns={pi.columns[0]: "feature"})
        rf_col = [c for c in pi.columns if c.startswith("Random")][0]
        top3 = ", ".join(nice(f) for f in pi.sort_values(rf_col, ascending=False)["feature"].head(3))
    kwp = ""
    if res["keyword_effects"] is not None:
        k = res["keyword_effects"].set_axis(["kw", "eff"], axis=1).sort_values("eff")
        lbl = lambda x: "'" + x.replace("kw_", "").replace("_", " ") + "'"
        kwp = (f"In raw medians, {lbl(k.iloc[-1]['kw'])} and {lbl(k.iloc[-2]['kw'])} ads are pricier and "
               f"{lbl(k.iloc[0]['kw'])} and {lbl(k.iloc[1]['kw'])} ads cheaper - but mostly because of *where* those ads are "
               "(e.g. agricultural land is far from Colombo). Once location is in the model, keywords add only a little, "
               "and they are simple word matches.")
    near = df.loc[df["dist_fort_km"] < 5, "price_per_perch"].median()
    far = df.loc[df["dist_fort_km"] > 30, "price_per_perch"].median()
    why(f"Land within 5 km of the Fort has a median of {rs(near)} per perch; beyond 30 km it is {rs(far)}. "
        "So distance to the Fort is the single most useful number.",
        kwp,
        f"In the models, the most important features are: **{top3}** (see tab 5).")

# ---------------------------------------------------------------------------
# 3. Ensemble lab - train a model live
# ---------------------------------------------------------------------------
with tabs[2]:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor,
                                  GradientBoostingRegressor, RandomForestRegressor, StackingRegressor,
                                  VotingRegressor)
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.tree import DecisionTreeRegressor

    c1, c2, c3 = st.columns(3)
    c1.markdown("**Bagging** - many deep trees trained in parallel on random samples, then averaged. "
                "Reduces *variance*. e.g. Random Forest, Extra Trees.")
    c2.markdown("**Boosting** - small trees trained one after another, each fixing the previous errors. "
                "Reduces *bias*. e.g. XGBoost, LightGBM, CatBoost.")
    c3.markdown("**Stacking** - different models predict, and a meta-model (Ridge) learns how to combine them.")
    st.divider()

    def pre():
        return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5,
                                                        sparse_output=False), CATEGORICAL_FEATURES),
                                  ("num", "passthrough", NUMERIC_FEATURES)])

    ctrl, out = st.columns([1, 2], gap="large")
    with ctrl:
        fam = st.radio("Family", ["Bagging", "Boosting", "Stacking"], horizontal=True)
        available = {
            "Bagging": ["Random Forest", "Extra Trees", "Bagging (decision trees)"],
            "Boosting": ["Gradient Boosting (sklearn)", "XGBoost", "LightGBM", "CatBoost", "AdaBoost"],
            "Stacking": ["Stacking (RF + ET + boosting -> Ridge)", "Voting (average)"],
        }[fam]
        name = st.selectbox("Model", available)

        model = None
        if fam == "Bagging":
            n = st.slider("Number of trees", 50, 800, 300, 50)
            depth = st.select_slider("Max depth", [4, 6, 8, 12, 16, 20, None], value=None)
            mf = st.slider("Share of features per split", 0.1, 1.0, 0.5, 0.1,
                           help="Random Forest's key idea: each split only sees a random subset of features")
            if name == "Random Forest":
                model = RandomForestRegressor(n_estimators=n, max_depth=depth, max_features=mf, n_jobs=-1, random_state=SEED)
            elif name == "Extra Trees":
                model = ExtraTreesRegressor(n_estimators=n, max_depth=depth, max_features=mf, n_jobs=-1, random_state=SEED)
            else:
                model = BaggingRegressor(DecisionTreeRegressor(max_depth=depth), n_estimators=n, n_jobs=-1, random_state=SEED)
        elif fam == "Boosting":
            n = st.slider("Number of trees (rounds)", 50, 1500, 500, 50)
            lr = st.select_slider("Learning rate", [0.01, 0.02, 0.05, 0.1, 0.2, 0.3], value=0.05)
            depth = st.slider("Tree depth", 2, 10, 5)
            try:
                if name == "Gradient Boosting (sklearn)":
                    model = GradientBoostingRegressor(n_estimators=n, learning_rate=lr, max_depth=depth, subsample=0.8, random_state=SEED)
                elif name == "XGBoost":
                    import xgboost as xgb
                    model = xgb.XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=depth, subsample=0.8,
                                             colsample_bytree=0.8, n_jobs=-1, random_state=SEED)
                elif name == "LightGBM":
                    import lightgbm as lgb
                    model = lgb.LGBMRegressor(n_estimators=n, learning_rate=lr, max_depth=depth, num_leaves=2 ** depth - 1,
                                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
                elif name == "CatBoost":
                    import catboost as cb
                    model = cb.CatBoostRegressor(iterations=n, learning_rate=lr, depth=min(depth, 10), random_seed=SEED,
                                                 verbose=0, allow_writing_files=False)
                else:
                    model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=depth), n_estimators=n, learning_rate=lr, random_state=SEED)
            except ImportError:
                st.warning(f"{name} is not installed. Run: pip install {name.lower()}")
        else:
            n = st.slider("Trees per base model", 100, 600, 300, 100)
            try:
                import xgboost as xgb
                booster = xgb.XGBRegressor(n_estimators=n, learning_rate=0.05, max_depth=5, subsample=0.8,
                                           colsample_bytree=0.8, n_jobs=-1, random_state=SEED)
            except ImportError:
                booster = GradientBoostingRegressor(n_estimators=n, learning_rate=0.05, max_depth=4, random_state=SEED)
            base = [("rf", RandomForestRegressor(n_estimators=n, max_features=0.5, n_jobs=-1, random_state=SEED)),
                    ("et", ExtraTreesRegressor(n_estimators=n, max_features=0.5, n_jobs=-1, random_state=SEED)),
                    ("boost", booster)]
            st.caption("Base models: Random Forest, Extra Trees, XGBoost. "
                       + ("Meta-model: Ridge regression." if name.startswith("Stacking") else "Combined by a simple average."))
            model = (StackingRegressor(base, final_estimator=RidgeCV(), cv=5) if name.startswith("Stacking")
                     else VotingRegressor(base))
        train = model is not None and st.button("Train and test", type="primary", use_container_width=True)

    with out:
        if train:
            pipe = Pipeline([("pre", pre()), ("model", model)])
            with st.spinner("Training..."):
                t0 = time.perf_counter(); pipe.fit(X_train, y_train); fit_s = time.perf_counter() - t0
            pred = pipe.predict(X_test)
            m = metrics(y_test, pred)
            k = st.columns(4)
            k[0].metric("Test R²", f"{m['R²']:.3f}")
            k[1].metric("RMSE (log)", f"{m['RMSE (log)']:.3f}")
            k[2].metric("MAPE", f"{m['MAPE %']:.1f}%")
            k[3].metric("Time", f"{fit_s:.1f}s")
            plot = pd.DataFrame({"actual": np.exp(y_test.values), "predicted": np.exp(pred)})
            lim = [plot.min().min(), plot.max().max()]
            sc = alt.Chart(plot).mark_circle(size=16, opacity=0.45, color=MAIN).encode(
                x=alt.X("actual:Q", title="Actual Rs per perch", **LOG_AXIS),
                y=alt.Y("predicted:Q", title="Predicted Rs per perch", **LOG_AXIS))
            diag = alt.Chart(pd.DataFrame({"actual": lim, "predicted": lim})).mark_line(
                color=HIGHLIGHT, strokeDash=[4, 4]).encode(x="actual", y="predicted")
            st.altair_chart((sc + diag).properties(height=360, title=f"{name}: predicted vs actual (test set)"),
                            use_container_width=True)
        else:
            st.info("Pick a model and press **Train and test**.")

# ---------------------------------------------------------------------------
# 4. Model comparison
# ---------------------------------------------------------------------------
with tabs[3]:
    if lb is None:
        st.info("Run the notebook to create results/leaderboard.csv")
    else:
        st.subheader("All models - same split, same features")
        st.caption("Ranked by 5-fold cross-validation R².")
        best = lb.iloc[0]
        chart_df = lb.assign(best=lb["model"].eq(best["model"]))
        lo = max(0.0, np.floor((lb["CV_R2_mean"].min() - 0.03) * 20) / 20)
        st.altair_chart(alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("CV_R2_mean:Q", title="Cross-validation R²", scale=alt.Scale(domain=[lo, 0.9], clamp=True), axis=alt.Axis(tickCount=5)),
            y=alt.Y("model:N", sort=list(lb["model"]), title=None, axis=alt.Axis(labelLimit=320)),
            color=alt.condition(alt.datum.best, alt.value(HIGHLIGHT), alt.value(MAIN)),
            tooltip=["model", "family", alt.Tooltip("CV_R2_mean:Q", format=".3f"), alt.Tooltip("R2:Q", format=".3f", title="test R²")]
        ).properties(height=30 * len(lb)), use_container_width=True)

        table = pd.DataFrame({
            "Rank": range(1, len(lb) + 1), "Model": lb["model"], "Family": lb["family"],
            "CV R²": [f"{m:.3f} ± {s:.3f}" for m, s in zip(lb["CV_R2_mean"], lb["CV_R2_std"])],
            "Test R²": lb["R2"].map("{:.3f}".format), "Test RMSE (log)": lb["RMSE"].map("{:.3f}".format),
            "Test MAPE": lb["MAPE_%"].map("{:.1f}%".format), "Fit time": lb["fit_seconds"].map("{:.1f}s".format)})
        st.dataframe(table, hide_index=True, use_container_width=True, height=35 * (len(table) + 1) + 3)

        tree = lb[lb["model"].str.contains("decision tree", case=False) & lb["family"].eq("Baseline")]
        ridge = lb[lb["family"].eq("Baseline") & ~lb.index.isin(tree.index)].iloc[0]
        fam_best = lb.groupby("family").head(1).set_index("family")
        pts = []
        if len(tree):
            pts.append(f"**Ensembles beat a single tree** ({tree.iloc[0]['R2']:.2f} -> {best['R2']:.2f}): one deep tree "
                       "memorises the training data (high variance); averaging or boosting many trees fixes that.")
        pts.append(f"**Ensembles only just beat {ridge['model']}** ({ridge['R2']:.3f} vs {best['R2']:.3f} test R²): Ridge has one "
                   "column per area, so it already learns each area's price level. Ensembles add the interactions "
                   "(e.g. size matters differently in Colombo 7 than in Horana) - a real but small gain.")
        if "Bagging" in fam_best.index and "Boosting" in fam_best.index:
            pts.append(f"**Boosting edged out bagging** (CV {fam_best.loc['Boosting','CV_R2_mean']:.3f} vs "
                       f"{fam_best.loc['Bagging','CV_R2_mean']:.3f}): boosting reduces bias; after cleaning, variance "
                       "was no longer the main problem.")
        pts.append(f"**{best['model']} ranked first, but only slightly**: stacking helps most when the base models make "
                   "different mistakes; here they make similar ones.")
        ex = lb[(lb.index > 4) & (lb["R2"] > lb["R2"].median())]
        pts.append("**Why rank by cross-validation, not test score?** Picking the winner on the test set quietly overfits it."
                   + (f" e.g. {ex.iloc[0]['model']} looks good on test ({ex.iloc[0]['R2']:.3f}) but is weaker in CV "
                      f"({ex.iloc[0]['CV_R2_mean']:.3f})." if len(ex) else ""))
        why(*pts)

        a, b = st.columns(2)
        if res["stacking_weights"] is not None:
            with a.expander("Stacking: weight the meta-model gives each base model"):
                w = res["stacking_weights"].copy(); w.columns = ["Base model", "Ridge weight"]
                st.dataframe(w.round(3), hide_index=True, use_container_width=True)
        with b.expander("Tree ensembles on tabular vs image/text data"):
            st.markdown("- **Tabular data (this project):** tree ensembles are usually best - mixed feature types, skewed values, "
                        "a few thousand rows, little tuning.\n"
                        "- **Images / text:** the meaning is in neighbouring pixels or word order, so CNNs and transformers win. "
                        "Trees only work on features extracted from them (e.g. embeddings) or hand-made flags like my title keywords.")

# ---------------------------------------------------------------------------
# 5. RF vs XGBoost
# ---------------------------------------------------------------------------
with tabs[4]:
    d = res["rf_vs_xgb"]
    if d is None:
        st.info("Run the notebook first.")
    else:
        st.subheader("Random Forest vs XGBoost (both tuned)")
        cols = st.columns(len(d))
        for c, (_, r) in zip(cols, d.iterrows()):
            with c.container(border=True):
                st.markdown(f"#### {r['model']}")
                m1, m2 = st.columns(2)
                m1.metric("Test R²", f"{r['R2']:.3f}")
                m2.metric("CV R²", f"{r['CV_R2_mean']:.3f}", help=f"± {r['CV_R2_std']:.3f} across the 5 folds")
                m3, m4 = st.columns(2)
                m3.metric("MAPE", f"{r['MAPE_%']:.1f}%")
                m4.metric("Training time", f"{r['fit_seconds']:.1f}s")
                with st.expander("Best hyper-parameters"):
                    st.json(json.loads(r["best_params"]))

        r0, r1 = d.iloc[0], d.iloc[1]
        tie = abs(r0["R2"] - r1["R2"]) < max(r0["CV_R2_std"], r1["CV_R2_std"])
        fast = r0 if r0["fit_seconds"] < r1["fit_seconds"] else r1
        why(("**A statistical tie.** " if tie else "")
            + f"The test R² gap ({abs(round(r0['R2'], 3) - round(r1['R2'], 3)):.3f}) is smaller than the spread between cross-validation folds "
            f"(± {max(r0['CV_R2_std'], r1['CV_R2_std']):.3f}), so neither model is really better on this data.",
            f"So the choice depends on other things: Random Forest is easier to tune and harder to overfit; "
            f"{fast['model'].split(' (')[0]} trained faster ({fast['fit_seconds']:.1f}s).",
            "Both models rely on the same features (chart below): location first, then size.")

        pi = res["permutation_importance"]
        if pi is not None:
            pi = pi.rename(columns={pi.columns[0]: "feature"})
            pi = pi.assign(_m=pi.drop(columns="feature").mean(axis=1)).nlargest(10, "_m").drop(columns="_m")
            pi["feature"] = pi["feature"].map(nice)
            long = pi.melt("feature", var_name="model", value_name="importance")
            st.subheader("What each model relies on")
            st.altair_chart(alt.Chart(long).mark_bar().encode(
                y=alt.Y("feature:N", sort=list(pi["feature"]), title=None, axis=alt.Axis(labelLimit=320)),
                x=alt.X("importance:Q", title="Drop in test R² when the feature is shuffled", axis=alt.Axis(tickCount=5)),
                color=alt.Color("model:N", scale=alt.Scale(range=[MAIN, HIGHLIGHT]), legend=alt.Legend(title=None, orient="bottom")),
                yOffset="model:N").properties(height=380), use_container_width=True)

# ---------------------------------------------------------------------------
# 6. RF + ANN hybrid
# ---------------------------------------------------------------------------
with tabs[5]:
    h = res["hybrid_rf_ann"]
    st.subheader("Combining Random Forest with a neural network (ANN)")
    st.markdown("""
| | Method | How it works |
|---|---|---|
| H1 | RF feature selection -> ANN | RF ranks the inputs; the ANN only uses the top 10 |
| H2 | RF prediction as ANN input | RF's out-of-fold prediction is added as an extra input to the ANN |
| H3 | ANN corrects RF's errors | RF predicts first; a small ANN learns RF's remaining error; final = RF + ANN |
""")
    a, b = st.columns([1.2, 1])
    if h is not None:
        with a:
            hh = h.assign(kind=np.where(h["model"].str.startswith("H"), "Hybrid", "Single model"))
            lo = max(0.0, np.floor((hh["R2"].min() - 0.03) * 20) / 20)
            st.altair_chart(alt.Chart(hh).mark_bar().encode(
                x=alt.X("R2:Q", title="Test R²", scale=alt.Scale(domain=[lo, 0.9], clamp=True), axis=alt.Axis(tickCount=5)),
                y=alt.Y("model:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
                color=alt.Color("kind:N", scale=alt.Scale(domain=["Single model", "Hybrid"], range=[LIGHT, MAIN]),
                                legend=alt.Legend(title=None, orient="bottom")),
                tooltip=["model", alt.Tooltip("R2:Q", format=".3f"), alt.Tooltip("MAPE_%:Q", format=".1f")]
            ).properties(height=260), use_container_width=True)
        with b:
            hi = h.set_index("model")
            ann, rf = hi.loc["ANN alone", "R2"], hi.loc["RF alone", "R2"]
            besth = h[h["model"].str.startswith("H")].sort_values("R2", ascending=False).iloc[0]
            why(f"**The ANN alone is the weakest** ({ann:.3f} vs RF {rf:.3f}): {len(X_train):,} training rows is small for a "
                "neural network, and most inputs are 0/1 columns, which trees handle naturally.",
                f"**The best hybrid ({besth['model'].split(' ')[0]}, {besth['R2']:.3f}) does not beat RF alone**: RF had already "
                "captured the pattern, so there was little error left for the ANN to fix.")
    if res["ann_architecture"] is not None:
        st.markdown("**ANN architecture**")
        arch = res["ann_architecture"].copy()
        arch.columns = [c.replace("_", " ").capitalize() for c in arch.columns]
        st.dataframe(arch.astype(str), hide_index=True, use_container_width=True)
    st.caption("RF + ARIMA is not applied: it needs a time series, and this data is a single snapshot.")

# ---------------------------------------------------------------------------
# 7. Price estimator
# ---------------------------------------------------------------------------
with tabs[6]:
    if not models:
        st.info("Run the notebook to create app/models/final_models.joblib")
    else:
        labels = {"kw_house": "House on it", "kw_commercial": "Commercial",
                  "kw_residential": "Residential", "kw_agricultural": "Agricultural",
                  "kw_bare_land": "Bare land", "kw_main_road": "Main road",
                  "kw_prime": "Prime / luxury", "kw_plots": "Block-out plots",
                  "kw_water_view": "Water view", "kw_urgent": "Urgent sale"}
        inp, outp = st.columns([1, 1.3], gap="large")
        with inp:
            st.subheader("Your plot")
            areas = sorted(df["area"].unique())
            area_sel = st.selectbox("Area", areas, index=areas.index("Maharagama") if "Maharagama" in areas else 0)
            district = coords.set_index("area").loc[area_sel, "district"]
            perches = st.number_input("Land size (perches)", 1.0, 1600.0, 10.0, 0.5)
            st.markdown("What does the ad say?")
            k1, k2 = st.columns(2)
            flags = {k: (k1 if i % 2 == 0 else k2).checkbox(v, value=(k == "kw_residential"))
                     for i, (k, v) in enumerate(labels.items())}
        row = build_input_row(area_sel, district, perches, flags, coords, is_member=1)
        preds = {n: float(np.exp(m.predict(row[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[0])) for n, m in models.items()}
        same = df[df["area"] == area_sel]
        with outp:
            st.subheader("Estimated asking price per perch")
            cols = st.columns(len(preds))
            for c, (n, v) in zip(cols, preds.items()):
                c.metric(n, rs(v), help=f"Total for {perches:g} perches: Rs {v*perches:,.0f}")
            main = "Stacking" if "Stacking" in preds else next(iter(preds))
            st.markdown(f"Total for {perches:g} perches ({main}): **Rs {preds[main]*perches/1e6:,.2f} million**")
            if len(same):
                st.caption(f"Median of {len(same)} ads in {area_sel}: {rs(same['price_per_perch'].median())} per perch")
                near = same.iloc[(same["perches"] - perches).abs().argsort()[:5]]
                st.markdown("**Real ads in this area with a similar size**")
                st.dataframe(pd.DataFrame({"Title": near["title"], "Perches": near["perches"],
                                           "Rs per perch": near["price_per_perch"].map(lambda x: f"{x:,.0f}")}),
                             hide_index=True, use_container_width=True)
