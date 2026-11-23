"""
scrape_ikman.py - collects "Land for Sale" listings from ikman.lk for the
Western Province (Colombo, Gampaha, Kalutara districts).

How it works:
  1. Every ikman search page embeds its data as JSON in a <script> tag that
     starts with  window.initialData = {...}
  2. Download the page, cut out that JSON, and read the list of ads.
  3. One request every ~1.5 seconds, only public listing pages, only a few
     pages per area - polite, small-scale collection for academic use.
     ikman's robots.txt (checked 23 Nov 2026) does not disallow these pages.

Run:  python scraper/scrape_ikman.py
Output: data/raw_listings.csv
"""
import csv
import json
import time
from pathlib import Path

import requests

BASE = "https://ikman.lk"
DISTRICTS = ["Colombo", "Gampaha", "Kalutara"]
HEADERS = {"User-Agent": "Mozilla/5.0 (student research project; land price study)"}
DELAY_SECONDS = 1.5
OUT = Path(__file__).resolve().parents[1] / "data" / "raw_listings.csv"

FIELDS = ["id", "district", "area", "title", "details", "price", "is_featured",
          "is_member", "membership_level", "is_top_ad", "is_urgent_ad",
          "is_verified", "time_stamp", "last_bump", "slug", "page"]


def get_initial_data(url: str) -> dict:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.encoding = "utf-8"          # ikman pages are UTF-8 (needed for Sinhala titles and dashes)
    html = r.text
    start = html.index("window.initialData = ") + len("window.initialData = ")
    end = html.index("</script>", start)
    blob = html[start:end].strip().rstrip(";")
    return json.loads(blob)


def list_areas() -> list:
    """Reads ikman's location tree and returns the areas of our three districts."""
    data = get_initial_data(f"{BASE}/en/ads/colombo/land-for-sale")
    locs = data["serp"]["locations"]
    areas = []
    for loc in locs.values():
        if loc["name"] in DISTRICTS and loc["parentId"] == -1:
            for child_id in loc["children"]:
                child = locs[str(child_id)]
                n = int(str(child.get("count", 0)).replace(",", ""))
                if n >= 8:  # skip areas with almost no listings
                    areas.append({"district": loc["name"], "area": child["name"],
                                  "slug": child["slug"], "n": n})
    return areas


def main():
    areas = list_areas()
    jobs = []
    for a in areas:
        pages = 3 if a["n"] >= 60 else (2 if a["n"] >= 25 else 1)
        jobs += [(a, p) for p in range(1, pages + 1)]
    print(f"{len(areas)} areas, {len(jobs)} pages to read")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, (a, p) in enumerate(jobs, 1):
            try:
                d = get_initial_data(f"{BASE}/en/ads/{a['slug']}/land-for-sale?page={p}")
                ads = d["serp"]["ads"]["data"].get("ads", [])
            except Exception as e:  # keep going if one page fails
                print("  failed", a["slug"], p, e)
                ads = []
            for ad in ads:
                if ad["id"] in seen:
                    continue
                seen.add(ad["id"])
                w.writerow({
                    "id": ad["id"], "district": a["district"], "area": a["area"],
                    "title": ad.get("title"), "details": ad.get("details"),
                    "price": ad.get("price"),
                    "is_featured": int(bool(ad.get("isFeaturedAd"))),
                    "is_member": int(bool(ad.get("isMember"))),
                    "membership_level": ad.get("membershipLevel"),
                    "is_top_ad": int(bool(ad.get("isTopAd"))),
                    "is_urgent_ad": int(bool(ad.get("isUrgentAd"))),
                    "is_verified": int(bool(ad.get("isVerified"))),
                    "time_stamp": ad.get("timeStamp"), "last_bump": ad.get("lastBumpUpDate"),
                    "slug": ad.get("slug"), "page": p,
                })
            print(f"[{i}/{len(jobs)}] {a['area']} page {p}: {len(ads)} ads, {len(seen)} unique so far")
            time.sleep(DELAY_SECONDS)
    print("Saved", OUT)


if __name__ == "__main__":
    main()
