"""Download HadUK-Grid daily 5 km rainfall from CEDA and keep only a Cumbria box, winters only.

Run in Google Colab (recommended) or on your own computer:

    pip install xarray netCDF4 requests
    export CEDA_TOKEN="paste-your-token-here"      # Colab: os.environ["CEDA_TOKEN"] = "..."
    python download_haduk_cumbria.py --test        # 1 file only: checks token, names, box
    python download_haduk_cumbria.py               # full download (Oct 1961 - Mar 2025)

Each monthly UK file is downloaded, cut to the Cumbria box straight away, and then
deleted, so disk use stays small. Finished months are cached in `subsets/`, so if the
download is interrupted, running the script again carries on where it stopped.

Output: cumbria_winter_rainfall_5km.nc (a few MB; about 1.2 GB passes through, one 3 MB file at a time): daily rainfall (mm) on a 16 x 16
grid of 5 km cells, October-March only.

Data: Met Office; Hollis, D.; McCarthy, M.; Kendon, M.; Legg, T.; Simpson, I. (2018):
HadUK-Grid gridded and regional average climate observations for the UK. Centre for
Environmental Data Analysis. Contains public sector information licensed under the
Open Government Licence v3.0.
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests
import xarray as xr

DAP = "https://dap.ceda.ac.uk"
TOKEN_CHECK = DAP + "/badc/ARCHIVE_INFO/ACCESS_TEST/RESTRICTED/TOKEN_CHECK"
DAY_FOLDER = "/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/{version}/5km/rainfall/day/"
# Inside day/ the files sit in a dated release folder, e.g. day/v20260512/ for v1.3.2.ceda.

# Cumbria box in British National Grid metres (cell centres must fall inside [lo, hi)).
# 80 km x 80 km = 16 x 16 cells of 5 km. Covers the Lake District, including Honister
# and Thirlmere, where Storm Desmond (5-6 Dec 2015) brought record rainfall.
EASTING = (320_000, 400_000)
NORTHING = (480_000, 560_000)
WINTER_MONTHS = {10, 11, 12, 1, 2, 3}


def session(token):
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    return s


def check_token(s):
    r = s.get(TOKEN_CHECK, timeout=60)
    if r.status_code != 200 or "<html" in r.text.lower()[:500]:
        sys.exit("Token check FAILED. Make a new token at https://services.ceda.ac.uk/account/token/ "
                 "(tokens expire after 3 days) and check you are logged in to CEDA.")
    print("Token check OK:", r.text.strip()[:120])


def find_release_folder(s, version, release):
    """Return the URL of the folder holding the .nc files: --release if given, else
    the newest dated folder (vYYYYMMDD) inside day/, else day/latest/."""
    day_url = DAP + DAY_FOLDER.format(version=version)
    if release:
        return day_url + release.strip("/") + "/"
    r = s.get(day_url, timeout=120)
    if r.status_code != 200:
        sys.exit(f"Could not list {day_url} (HTTP {r.status_code}). Open it in a browser while logged "
                 "in to check the version folder name, then pass --version.")
    dated = sorted(set(re.findall(r'\b(v\d{8})\b', r.text)))
    if dated:
        print(f"Release folders found in day/: {dated}; using the newest, {dated[-1]}")
        return day_url + dated[-1] + "/"
    return day_url + "latest/"


def list_files(s, url):
    r = s.get(url, timeout=120)
    if r.status_code != 200:
        sys.exit(f"Could not list {url} (HTTP {r.status_code}). Open it in a browser while logged in "
                 "and pass the release folder name with --release (e.g. --release v20260512).")
    names = sorted(set(re.findall(r'([\w.-]+_day_(\d{8})-(\d{8})\.nc)', r.text)))
    files = [(n, start) for n, start, _ in names]
    if not files:
        sys.exit(f"No daily .nc files found in {url}. Open it in a browser and check the file names.")
    print(f"Found {len(files)} daily files in {url}")
    print("  e.g.", files[0][0], "...", files[-1][0])
    return url, files


def wanted(start, first_winter, last_winter):
    """True if the month belongs to a winter (Oct-Mar) starting in first_winter..last_winter."""
    year, month = int(start[:4]), int(start[4:6])
    if month not in WINTER_MONTHS:
        return False
    winter = year if month >= 10 else year - 1      # Dec 2015 and Jan 2016 -> winter 2015
    return first_winter <= winter <= last_winter


def find_names(ds):
    var = "rainfall" if "rainfall" in ds.data_vars else next(
        v for v in ds.data_vars if ds[v].ndim == 3)
    x = next(c for c in ds[var].dims if "x" in c.lower())
    y = next(c for c in ds[var].dims if "y" in c.lower())
    return var, x, y


def subset(path):
    with xr.open_dataset(path) as ds:
        var, x, y = find_names(ds)
        xv, yv = ds[x].values, ds[y].values
        box = ds[var].isel(
            {x: np.flatnonzero((xv >= EASTING[0]) & (xv < EASTING[1])),
             y: np.flatnonzero((yv >= NORTHING[0]) & (yv < NORTHING[1]))}).load()
    box = box.rename({x: "x", y: "y"}).sortby("y").sortby("x")
    if box.sizes["x"] != 16 or box.sizes["y"] != 16:
        raise RuntimeError(f"Expected a 16 x 16 box, got {dict(box.sizes)}: check the grid coordinates.")
    return box.astype("float32").rename("rainfall")


def download(s, url, dest, tries=4):
    for k in range(tries):
        try:
            with s.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            return
        except Exception as e:                          # network hiccup: wait and retry
            print(f"    retry {k + 1}/{tries} after error: {e}")
            time.sleep(10 * (k + 1))
    raise RuntimeError(f"Failed to download {url}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v1.3.2.ceda")
    p.add_argument("--release", default="", help="dated folder inside day/, e.g. v20260512 (auto if empty)")
    p.add_argument("--first-winter", type=int, default=1961)
    p.add_argument("--last-winter", type=int, default=2024)   # winter 2024 = Oct 2024 - Mar 2025
    p.add_argument("--out", default="cumbria_winter_rainfall_5km.nc")
    p.add_argument("--test", action="store_true", help="download one month only, as a check")
    a = p.parse_args()

    token = os.environ.get("CEDA_TOKEN", "").strip()
    if not token:
        sys.exit("Set the CEDA_TOKEN environment variable first (see the top of this file).")
    s = session(token)
    check_token(s)
    folder_url, files = list_files(s, find_release_folder(s, a.version, a.release))

    todo = [(n, st) for n, st in files if wanted(st, a.first_winter, a.last_winter)]
    if a.test:
        todo = [t for t in todo if t[1].startswith("201512")] or todo[:1]
    print(f"{len(todo)} monthly files to process")

    cache = Path("subsets"); cache.mkdir(exist_ok=True)
    tmp = Path("tmp_download.nc")
    for i, (name, start) in enumerate(todo, 1):
        out = cache / name
        if out.exists():
            continue
        print(f"[{i}/{len(todo)}] {name}")
        download(s, folder_url + name, tmp)
        subset(tmp).to_netcdf(out)
        tmp.unlink()

    parts = [xr.open_dataarray(cache / n) for n, _ in todo]
    da = xr.concat(parts, dim="time").sortby("time")
    da.attrs.update(
        units="mm", source=f"HadUK-Grid {a.version}, 5 km daily rainfall, CEDA",
        licence="Open Government Licence v3.0",
        citation=("Met Office; Hollis, D.; McCarthy, M.; Kendon, M.; Legg, T.; Simpson, I. (2018): "
                  "HadUK-Grid gridded and regional average climate observations for the UK. CEDA."),
        region=f"Easting {EASTING}, Northing {NORTHING} (British National Grid), Oct-Mar only")
    da.to_netcdf(a.out)

    # Quick checks you should look at
    n_missing_cells = int(da.isnull().all("time").sum())
    print(f"\nSaved {a.out}: {da.sizes['time']} days, {da.sizes['y']} x {da.sizes['x']} cells")
    print(f"Period: {str(da.time.values[0])[:10]} to {str(da.time.values[-1])[:10]}")
    print(f"Cells that are always missing (sea): {n_missing_cells} of 256")
    try:
        d = da.sel(time="2015-12-05")
        print(f"Storm Desmond check, 5 Dec 2015: wettest cell {float(d.max()):.0f} mm, "
              f"box mean {float(d.mean()):.0f} mm (expect a very large value for a 5 km cell)")
    except KeyError:
        print("(5 Dec 2015 not in this download)")


if __name__ == "__main__":
    main()
