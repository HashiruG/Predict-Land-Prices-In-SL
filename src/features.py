"""
features.py - cleaning and feature engineering for the ikman.lk land listings.

The notebook and the Streamlit app both import this file, so the same rules are
used for training and for live predictions.
"""
import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Reference points
# ---------------------------------------------------------------------------
# Colombo Fort is the usual "zero point" for distances in Sri Lanka.
# Distance to the Fort is used as the main location feature.
FORT = (6.9344, 79.8428)

# Approximate coordinates of expressway interchanges in the Western Province
# (E01 Southern, E02 Outer Circular, E03 Katunayake, E04 Central - first section).
# Approximate to within about 1 km - good enough for a distance feature.
EXPRESSWAY_ENTRANCES = {
    "Kottawa": (6.8410, 79.9640),
    "Kahathuduwa": (6.7790, 79.9890),
    "Gelanigama": (6.7290, 80.0110),
    "Dodangoda": (6.5480, 80.0480),
    "Welipenna": (6.4550, 80.0990),
    "Athurugiriya": (6.8770, 79.9840),
    "Kaduwela": (6.9230, 79.9880),
    "Kadawatha": (7.0020, 79.9600),
    "Kerawalapitiya": (6.9890, 79.8870),
    "Peliyagoda": (6.9640, 79.8810),
    "Ja-Ela": (7.0700, 79.8960),
    "Katunayake": (7.1640, 79.8780),
    "Mirigama": (7.2400, 80.1000),
}

PERCHES_PER_ACRE = 160.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points on Earth, in kilometres."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# 2. Parsing the raw text fields
# ---------------------------------------------------------------------------
def fix_text(x):
    """Repairs text that was decoded with the wrong character set.
    Python's requests library guesses ISO-8859-1 when a server does not name its charset,
    so 'Kotte – Road' arrived as 'Kotte â\x80\x93 Road' and Sinhala became 'à¶...'.
    Re-encoding as Latin-1 and decoding as UTF-8 restores the original text."""
    if not isinstance(x, str):
        return x
    try:
        return x.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return x  # already correct


def parse_size_perches(details: str):
    """'16.1 perches' -> 16.1 ; '2.0 acres' -> 320.0 ; anything else -> NaN"""
    if not isinstance(details, str):
        return np.nan
    m = re.search(r"([\d,]+(?:\.\d+)?)\s*(perch|acre)", details.lower())
    if not m:
        return np.nan
    value = float(m.group(1).replace(",", ""))
    return value * PERCHES_PER_ACRE if m.group(2) == "acre" else value


def parse_price(price: str):
    """Returns (amount_in_rupees, price_type) where price_type is
    'per_perch', 'total', 'per_acre' or 'unknown'."""
    if not isinstance(price, str):
        return np.nan, "unknown"
    m = re.search(r"([\d,]+(?:\.\d+)?)", price)
    if not m:
        return np.nan, "unknown"
    amount = float(m.group(1).replace(",", ""))
    low = price.lower()
    if "per perch" in low:
        kind = "per_perch"
    elif "per acre" in low:
        kind = "per_acre"
    elif "total" in low:
        kind = "total"
    else:
        kind = "unknown"
    return amount, kind


def to_price_per_perch(amount, kind, perches):
    """Puts every listing on the same scale: rupees per perch."""
    if kind == "per_perch":
        return amount
    if kind == "per_acre":
        return amount / PERCHES_PER_ACRE
    if kind == "total" and perches and perches > 0:
        return amount / perches
    return np.nan


# ---------------------------------------------------------------------------
# 2b. Correcting the location using the title
# ---------------------------------------------------------------------------
# ikman's area pages sometimes show ads from other places (for example the
# "Colombo 3" page listed plots in Nittambuwa). Sellers almost always write
# the real place in the title, so if the title clearly names a different
# known area, the title is trusted. Neighbourhood names are mapped to the ikman
# area they belong to.
ALIASES = {
    # spelling variants
    "thalawathugoda": "Talawatugoda", "thalawatugoda": "Talawatugoda",
    "rathmalana": "Ratmalana", "mt. lavinia": "Mount Lavinia", "mt lavinia": "Mount Lavinia",
    "kaluthara": "Kalutara City", "kalutara": "Kalutara City", "negambo": "Negombo",
    "jaela": "Ja-Ela", "ja ela": "Ja-Ela", "aturugiriya": "Athurugiriya",
    "dehiwela": "Dehiwala", "kalubovila": "Kalubowila", "borelesgamuwa": "Boralesgamuwa",
    "aluthgama": "Alutgama", "kiribathgoa": "Kiribathgoda", "attanagalla": "Nittambuwa",
    # neighbourhoods -> ikman area
    "wellawatta": "Colombo 6", "wellawatte": "Colombo 6", "pamankada": "Colombo 6",
    "bambalapitiya": "Colombo 4", "bambalapity": "Colombo 4", "kollupitiya": "Colombo 3",
    "borella": "Colombo 8", "kotahena": "Colombo 13", "dematagoda": "Colombo 9",
    "havelock town": "Colombo 5", "kirulapana": "Colombo 5", "kirulapone": "Colombo 5",
    "narahenpita": "Colombo 5", "thimbirigasyaya": "Colombo 5", "cinnamon gardens": "Colombo 7",
    "modara": "Colombo 15", "mattakkuliya": "Colombo 15", "grandpass": "Colombo 14",
    "maradana": "Colombo 10", "slave island": "Colombo 2",
    "kahathuduwa": "Polgasowita", "hokandara": "Talawatugoda", "pelawatta": "Battaramulla",
    "pelawatte": "Battaramulla", "madapatha": "Piliyandala", "yakkala": "Kalagedihena",
    "weliweriya": "Delgoda", "udugampola": "Minuwangoda", "wathupitiwala": "Nittambuwa",
    "handapangoda": "Padukka", "panagoda": "Godagama", "uyanwatta": "Bandaragama",
    "katubedda": "Moratuwa", "bellanwila": "Boralesgamuwa", "pitakotte": "Kotte",
    "ethul kotte": "Kotte", "ethulkotte": "Kotte", "madiwela": "Kotte", "delkanda": "Nugegoda",
    "arawwala": "Pannipitiya", "mahara": "Kadawatha", "makola": "Kiribathgoda",
    "kadirana": "Negombo", "katana": "Negombo", "malapalla": "Kottawa",
    "mulleriyawa": "Angoda", "attidiya": "Dehiwala", "watareka": "Meegoda",
    # more spelling variants found in the titles
    "awissawella": "Avissawella", "udugampala": "Minuwangoda", "mathugama": "Matugama",
    "katunayaka": "Katunayake", "mountlavinia": "Mount Lavinia", "hanwalla": "Hanwella",
    "katubadda": "Moratuwa", "baththaramulla": "Battaramulla", "talawathugoda": "Talawatugoda",
    "hadapangoda": "Padukka", "kiridiwela": "Kirindiwela", "diuwlapitiya": "Divulapitiya",
    "thalagala": "Talagala", "attideya": "Dehiwala",
    # more neighbourhoods -> ikman area
    "kaluaggala": "Avissawella", "pasyala": "Nittambuwa", "akuregoda": "Battaramulla",
    "gothatuwa": "Angoda", "himbutana": "Angoda", "ambathale": "Angoda", "bokundara": "Piliyandala",
    "gonawala": "Kelaniya", "diddeniya": "Hanwella", "pamunugama": "Ja-Ela", "ekala": "Ja-Ela",
    "uswetakeiyawa": "Wattala", "hekitta": "Wattala", "kerawalapitiya": "Wattala",
    "batagama": "Kandana", "andiambalama": "Katunayake", "kurana": "Negombo",
    "mirihana": "Nugegoda", "embuldeniya": "Nugegoda", "udahamulla": "Nugegoda",
    "thalahena": "Malabe", "korathota": "Kaduwela", "nawinna": "Maharagama",
    "kalapaluwawa": "Rajagiriya", "buthgamuwa": "Rajagiriya", "polhengoda": "Colombo 5",
    "vauxhall": "Colombo 2", "malwana": "Biyagama", "koralaima": "Horana",
    "radawana": "Kirindiwela", "pingamuwa": "Kirindiwela", "payagala": "Kalutara City",
    "welipenna": "Matugama", "meepe": "Padukka", "bollatha": "Ganemulla",
    "dunagaha": "Divulapitiya", "dewalapola": "Minuwangoda", "kiriwaththuduwa": "Homagama",
    "palanwatta": "Pannipitiya", "pelanwatta": "Pannipitiya",
}

# Places OUTSIDE the Western Province that agents attach to Colombo/Gampaha area pages
# (e.g. "Land For Sale In Dambadeniya A01 Colombo 3" is really ~100 km away in Kurunegala
# district). Such ads cannot be located with our coordinates, so they are removed.
OUT_OF_REGION = [
    "kurunegala", "polgahawela", "pannala", "mallawagedara", "mellawagedara", "melsiripura",
    "melasiripura", "dambadeniya", "padeniya", "wariyapola", "kuliyapitiya", "narammala",
    "giriulla", "alawwa", "bingiriya", "ibbagamuwa", "rambukkana", "kegalle", "warakapola",
    "chilaw", "puttalam", "wennappuwa", "marawila", "anuradhapura", "ratnapura", "nuwara eliya",
    "dambulla", "hambantota", "matara", "ambalangoda", "hikkaduwa", "trincomalee", "batticaloa",
    "jaffna", "badulla", "eheliyagoda", "kuruwita", "kiriella", "ruwanwella", "kitulgala", "deraniyagala", "kandy(?! ?(rd|road))", "galle(?! ?(rd|road|face))",
]
OUT_OF_REGION_RE = re.compile(r"\b(" + "|".join(OUT_OF_REGION) + r")\b")


def is_out_of_region(title):
    return isinstance(title, str) and bool(OUT_OF_REGION_RE.search(title.lower()))


def _area_patterns(known_areas):
    pats = []
    for a in known_areas:
        m = re.match(r"^Colombo (\d+)$", a)
        if m:  # 'Colombo 7', 'Colombo 07', 'Colombo.07', 'Col 7', 'Col 07'
            n = int(m.group(1))
            pats.append((rf"\bcol(?:ombo)?[\s.\-]*0?{n}\b", a))
        else:
            base = a.replace(" City", "").lower()
            pats.append((rf"\b{re.escape(base)}\b", a))
    for alias, a in ALIASES.items():
        pats.append((rf"\b{re.escape(alias)}\b", a))
    return pats


def _is_city(area):
    return bool(re.match(r"^Colombo \d+$", str(area)))


def area_from_title(title, page_area, patterns):
    """Returns (area, source). source is 'page' or 'title'.

    1. Find every known area / alias named in the title, ordered by position in the title.
    2. Nothing found -> keep the page's area.
    3. Page area is one of Colombo 1-15 and the title also names a place outside the
       city -> use that place (spam pattern: "Land In Yakkala E01 Colombo 3").
    4. Otherwise, if the page's area is named, keep it; if not, use the first place named.
    """
    if not isinstance(title, str):
        return page_area, "page"
    t = title.lower()
    hits = []
    for p, a in patterns:
        m = re.search(p, t)
        if m:
            hits.append((m.start(), a))
    found = []
    for _, a in sorted(hits):
        if a not in found:
            found.append(a)
    if not found:
        return page_area, "page"
    if _is_city(page_area):
        outside = [a for a in found if not _is_city(a)]
        if outside:
            return outside[0], "title"
    if page_area in found:
        return page_area, "page"
    return found[0], "title"


# ---------------------------------------------------------------------------
# 3. Keyword features from the ad title
# ---------------------------------------------------------------------------
# Each entry: feature name -> regular expression searched in the lower-case title.
KEYWORDS = {
    "kw_house": r"\bhouse\b|\bhome\b|with (a |an )?(old )?(house|building)|bungalow|annex|නිවස|නිවාස|ගෙය",
    "kw_commercial": r"commercial|business|shop|warehouse|factory|industrial|showroom|වාණිජ|ව්‍යාපාරික",
    "kw_residential": r"residential|housing|resident|නේවාසික|පදිංචියට",
    "kw_agricultural": r"paddy|coconut|cinnamon|rubber|tea|estate|cultivat|agri|farm|කුඹුර|පොල් ඉඩම|වගා",
    "kw_bare_land": r"bare|empty|vacant|cleared|හිස් ඉඩම",
    "kw_main_road": r"main road|facing|frontage|galle road|high level|low level|kandy road|negombo road|highway|junction|town|පාරට මුහුණලා|බස් පාරට|ප්‍රධාන පාර|නගරය",
    "kw_prime": r"prime|luxury|exclusive|valuable|rare|super|premium|අගනා|වටිනා",
    "kw_plots": r"plots?\b|blocks?\b|subdivid|land blocks|blocks? of|බිම් කොටස",
    "kw_water_view": r"lake|river|beach|sea|canal|water|view|වෙල් යාය|ඔයට|ගඟට",
    "kw_urgent": r"urgent|quick sale|negotiable|below market|reduced|හදිසි",
}


def keyword_flags(title: str) -> dict:
    t = title.lower() if isinstance(title, str) else ""
    return {k: int(bool(re.search(p, t))) for k, p in KEYWORDS.items()}


# ---------------------------------------------------------------------------
# 4. Full cleaning pipeline
# ---------------------------------------------------------------------------
def clean_listings(raw: pd.DataFrame, coords: pd.DataFrame, verbose=True, drop_duplicate_plots=True):
    """raw: one row per scraped ad. coords: area -> lat/lon.
    drop_duplicate_plots=False keeps repeated plots (only used to show their effect).
    Returns (clean_df, log) where log records how many rows each rule removed."""
    log = []
    df = raw.copy()
    for col in ("title", "details", "price"):
        df[col] = df[col].apply(fix_text)
    log.append(("Scraped rows", len(df)))

    df = df.drop_duplicates(subset="id")
    log.append(("After removing duplicate ads (same ad id)", len(df)))

    # Same title + same size + same price posted twice = re-post of the same plot
    df = df.drop_duplicates(subset=["title", "details", "price"])
    log.append(("After removing re-posts (same title, size and price)", len(df)))

    df["perches"] = df["details"].apply(parse_size_perches)
    parsed = df["price"].apply(parse_price)
    df["price_amount"] = [p[0] for p in parsed]
    df["price_type"] = [p[1] for p in parsed]
    df["price_per_perch"] = [
        to_price_per_perch(a, k, s)
        for a, k, s in zip(df["price_amount"], df["price_type"], df["perches"])
    ]
    df = df.dropna(subset=["perches", "price_per_perch"])
    log.append(("After dropping ads with no readable size or price", len(df)))

    df = df[~df["title"].apply(is_out_of_region)]
    log.append(("After removing ads that name a place outside the Western Province", len(df)))

    # Fix the area using the title, then attach the correct district
    patterns = _area_patterns(coords["area"].tolist())
    fixed = [area_from_title(t, a, patterns) for t, a in zip(df["title"], df["area"])]
    df["page_area"] = df["area"]
    df["area"] = [f[0] for f in fixed]
    df["area_source"] = [f[1] for f in fixed]
    df["district"] = df["area"].map(coords.set_index("area")["district"]).fillna(df["district"])
    log.append((f"Areas corrected from the title: {(df['area_source'] == 'title').sum()} rows", len(df)))

    # Sanity limits (typos such as 'Rs 1' or '100000 perches')
    df = df[(df["perches"] >= 1) & (df["perches"] <= 1600)]          # 1 perch to 10 acres
    df = df[(df["price_per_perch"] >= 20_000) & (df["price_per_perch"] <= 60_000_000)]
    log.append(("After sanity limits (1-1600 perches, Rs 20k-60M per perch)", len(df)))

    # Duplicate plots: agents post the same block-out project many times with slightly
    # different titles. Same area + same size + same price per perch = one plot. Keeping the
    # copies would put the same plot in both the training and the test set (leakage).
    if drop_duplicate_plots:
        df = df.drop_duplicates(subset=["area", "perches", "price_per_perch"])
        log.append(("After removing duplicate plots (same area, size and price per perch)", len(df)))

    # Location features
    df = df.merge(coords[["area", "lat", "lon"]], on="area", how="left")
    df = df.dropna(subset=["lat", "lon"])
    df["dist_fort_km"] = haversine_km(df["lat"], df["lon"], *FORT)
    exp = np.array(list(EXPRESSWAY_ENTRANCES.values()))
    d = np.stack([haversine_km(df["lat"].values, df["lon"].values, la, lo) for la, lo in exp], axis=1)
    df["dist_expressway_km"] = d.min(axis=1)
    df["is_colombo_city"] = df["area"].str.match(r"^Colombo \d+$").astype(int)

    # Size features
    df["log_perches"] = np.log(df["perches"])
    df["is_large_land"] = (df["perches"] >= 40).astype(int)  # 40 perches = quarter acre

    # Price type as a feature (total-price ads are often land WITH a house)
    df["price_was_total"] = (df["price_type"] == "total").astype(int)

    # Title keywords
    kw = pd.DataFrame([keyword_flags(t) for t in df["title"]], index=df.index)
    df = pd.concat([df, kw], axis=1)

    # Target
    df["log_price_per_perch"] = np.log(df["price_per_perch"])

    # Outliers: +-2 standard deviations inside each area (price levels are very different between areas).
    # Areas with fewer than 5 ads are left alone (a standard deviation of 3 numbers means little).
    def _within_2sd(col):
        g = df.groupby("area")[col]
        m, s, n = g.transform("mean"), g.transform("std"), g.transform("count")
        return (n < 5) | ((df[col] - m).abs() <= 2 * s)

    df = df[_within_2sd("log_price_per_perch")]
    log.append(("After removing price outliers (+-2 SD within each area)", len(df)))
    df = df[_within_2sd("log_perches")]
    log.append(("After removing land-size outliers (+-2 SD within each area)", len(df)))

    df = df.reset_index(drop=True)
    if verbose:
        for step, n in log:
            print(f"{step:<60} {n:>6}")
    return df, log


# Columns used as model inputs
NUMERIC_FEATURES = [
    "log_perches", "dist_fort_km", "dist_expressway_km", "lat", "lon",
    "is_colombo_city", "is_large_land", "price_was_total",
    "is_member", "is_featured", "is_urgent_ad",
] + list(KEYWORDS.keys())
CATEGORICAL_FEATURES = ["district", "area"]
TARGET = "log_price_per_perch"


def build_input_row(area, district, perches, title_flags: dict, coords: pd.DataFrame,
                    price_was_total=0, is_member=0, is_featured=0, is_urgent_ad=0):
    """Builds one feature row for the dashboard's price estimator."""
    c = coords.set_index("area").loc[area]
    lat, lon = float(c["lat"]), float(c["lon"])
    exp = np.array(list(EXPRESSWAY_ENTRANCES.values()))
    row = {
        "district": district, "area": area,
        "log_perches": np.log(perches), "lat": lat, "lon": lon,
        "dist_fort_km": float(haversine_km(lat, lon, *FORT)),
        "dist_expressway_km": float(min(haversine_km(lat, lon, la, lo) for la, lo in exp)),
        "is_colombo_city": int(bool(re.match(r"^Colombo \d+$", area))),
        "is_large_land": int(perches >= 40),
        "price_was_total": price_was_total, "is_member": is_member,
        "is_featured": is_featured, "is_urgent_ad": is_urgent_ad,
    }
    for k in KEYWORDS:
        row[k] = int(title_flags.get(k, 0))
    return pd.DataFrame([row])
