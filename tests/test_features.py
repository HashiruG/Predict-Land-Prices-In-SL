"""Unit tests for src/features.py.   Run from the project folder:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import (CATEGORICAL_FEATURES, FORT, KEYWORDS, NUMERIC_FEATURES, _area_patterns,  # noqa: E402
                      area_from_title, build_input_row, clean_listings, fix_text, haversine_km,
                      is_out_of_region, keyword_flags, parse_price, parse_size_perches, to_price_per_perch)

COORDS = pd.read_csv(ROOT / "data" / "area_coordinates.csv")
PATTERNS = _area_patterns(COORDS["area"].tolist())


# --- parsing -----------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ("16.1 perches", 16.1), ("2.0 acres", 320.0), ("1,000.0 perches", 1000.0),
    ("Bedrooms: 3 | Bathrooms: 2", None), (None, None)])
def test_parse_size(text, expected):
    got = parse_size_perches(text)
    assert (np.isnan(got) if expected is None else got == pytest.approx(expected))


@pytest.mark.parametrize("text, amount, kind", [
    ("Rs 5,700,000 per perch", 5_700_000, "per_perch"),
    ("Rs 28,500,000 total price", 28_500_000, "total"),
    ("Rs 485,000 per acre", 485_000, "per_acre"),
    ("Rs 38,500,000", 38_500_000, "unknown")])
def test_parse_price(text, amount, kind):
    assert parse_price(text) == (amount, kind)


def test_price_per_perch_conversions():
    assert to_price_per_perch(5_700_000, "per_perch", 10) == 5_700_000
    assert to_price_per_perch(28_500_000, "total", 27) == pytest.approx(1_055_555.56, rel=1e-6)
    assert to_price_per_perch(4_800_000, "per_acre", 10) == 30_000
    assert np.isnan(to_price_per_perch(1_000_000, "unknown", 10))


# --- geography ---------------------------------------------------------------
def test_haversine_worked_example():
    # Maharagama -> Colombo Fort is about 13.3 km
    assert haversine_km(6.8480, 79.9265, *FORT) == pytest.approx(13.33, abs=0.05)
    assert haversine_km(*FORT, *FORT) == 0


def test_every_area_has_coordinates_in_western_province():
    assert COORDS["area"].is_unique
    assert COORDS["district"].isin(["Colombo", "Gampaha", "Kalutara"]).all()
    assert COORDS["lat"].between(6.3, 7.4).all() and COORDS["lon"].between(79.8, 80.3).all()


# --- text repair and location ------------------------------------------------
def test_fix_text_repairs_garbled_utf8():
    garbled = "Kotte – Road".encode("utf-8").decode("latin-1")
    assert fix_text(garbled) == "Kotte – Road"
    assert fix_text("Already fine") == "Already fine"
    assert fix_text(None) is None


@pytest.mark.parametrize("title, page, area, source", [
    ("Land For Sale In Wathupitiwala, Nittambuwa M04", "Colombo 3", "Nittambuwa", "title"),    # rule B
    ("Developed Land Plots For Sale In Yakkala E01 Colombo 3", "Colombo 3", "Kalagedihena", "title"),  # rule B
    ("Land for Sale in Colombo 07 - Barnes place", "Colombo 3", "Colombo 7", "title"),        # rule C
    ("Kottawa Malapalla Land For Sale", "Malabe", "Kottawa", "title"),                         # rule C
    ("Residential Land for Sale in Watareka Homagama", "Meegoda", "Meegoda", "page"),          # page named
    ("Prime Land for Sale in Colombo 03", "Colombo 3", "Colombo 3", "page"),
    ("Beautiful land for sale", "Horana", "Horana", "page"),                                    # rule A
])
def test_area_from_title(title, page, area, source):
    assert area_from_title(title, page, PATTERNS) == (area, source)


@pytest.mark.parametrize("title, out", [
    ("Land For Sale In Dambadeniya A01 Colombo 3", True),
    ("Land for Sale in Kurunegala Polgahawela Y 15", True),
    ("Super Residential Land Blocks in Eheliyagoda", True),
    ("Galle Road Facing Colombo 3 Land", False),
    ("45 Perch Land For Sale Facing Kandy Road", False),
    ("Prime Land for Sale in Nugegoda", False)])
def test_out_of_region(title, out):
    assert is_out_of_region(title) is out


# --- keywords ----------------------------------------------------------------
def test_keyword_flags_english_and_sinhala():
    f = keyword_flags("10P Bare Land with Old House Facing Main Road")
    assert f["kw_bare_land"] == f["kw_house"] == f["kw_main_road"] == 1 and f["kw_agricultural"] == 0
    assert keyword_flags("අගනා නේවාසික බිම් කොටස්")["kw_plots"] == 1
    assert keyword_flags("අගනා නේවාසික බිම් කොටස්")["kw_residential"] == 1
    assert keyword_flags("Land for sale in Homagama")["kw_house"] == 0      # 'homagama' must not match 'home'
    assert set(keyword_flags(None)) == set(KEYWORDS)


# --- full cleaning pipeline --------------------------------------------------
def _raw():
    rows = [
        # id, area, title, details, price
        ("a1", "Nugegoda", "Land for Sale in Nugegoda", "10.0 perches", "Rs 4,000,000 per perch"),
        ("a1", "Nugegoda", "Land for Sale in Nugegoda", "10.0 perches", "Rs 4,000,000 per perch"),   # same id
        ("a2", "Nugegoda", "Prime land Nugegoda A2", "10.0 perches", "Rs 4,000,000 per perch"),       # duplicate plot
        ("a3", "Colombo 3", "Land For Sale In Dambadeniya A01 Colombo 3", "12.5 perches", "Rs 155,000 per perch"),
        ("a4", "Colombo 3", "Land for Sale in Colombo 07", "20.0 perches", "Rs 25,000,000 per perch"),
        ("a5", "Horana", "House for sale", "Bedrooms: 3 | Bathrooms: 2", "Rs 38,500,000"),           # no size
        ("a6", "Horana", "Land in Horana", "20.0 perches", "Rs 10,000,000 total price"),
        ("a7", "Horana", "Land in Horana", "1.0 perches", "Rs 1 per perch"),                         # typo
    ]
    df = pd.DataFrame(rows, columns=["id", "area", "title", "details", "price"])
    df["district"] = df["area"].map(COORDS.set_index("area")["district"])
    df["is_member"] = df["is_featured"] = df["is_urgent_ad"] = 0
    return df


def test_clean_listings_rules():
    df, log = clean_listings(_raw(), COORDS, verbose=False)
    assert set(df["id"]) == {"a1", "a4", "a6"}
    assert df.set_index("id").loc["a4", "area"] == "Colombo 7"
    assert df.set_index("id").loc["a6", "price_per_perch"] == 500_000
    assert df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].notna().all().all()
    assert np.allclose(df["log_price_per_perch"], np.log(df["price_per_perch"]))
    assert [n for _, n in log] == sorted([n for _, n in log], reverse=True)   # rows never increase


def test_duplicate_plots_can_be_kept_for_the_leakage_check():
    kept, _ = clean_listings(_raw(), COORDS, verbose=False, drop_duplicate_plots=False)
    assert {"a1", "a2"} <= set(kept["id"])


def test_build_input_row_matches_training_columns():
    row = build_input_row("Maharagama", "Colombo", 10, {"kw_residential": 1}, COORDS, is_member=1)
    assert list(row[NUMERIC_FEATURES + CATEGORICAL_FEATURES].columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert row.loc[0, "dist_fort_km"] == pytest.approx(13.33, abs=0.05)
    assert row.loc[0, "log_perches"] == pytest.approx(np.log(10))
