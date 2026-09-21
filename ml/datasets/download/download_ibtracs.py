"""
Automated downloader for NOAA NCEI IBTrACS v04r01 best-track dataset.
Official Endpoint: https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/
"""
import argparse
import sys
import urllib.request
from pathlib import Path

IBTRACS_BASE_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv"

VALID_BASINS = {
    "NI": "ibtracs.NI.list.v04r01.csv",      # North Indian Ocean (Bay of Bengal & Arabian Sea)
    "ALL": "ibtracs.ALL.list.v04r01.csv",    # Global archive
    "NA": "ibtracs.NA.list.v04r01.csv",      # North Atlantic
    "EP": "ibtracs.EP.list.v04r01.csv",      # Eastern Pacific
    "WP": "ibtracs.WP.list.v04r01.csv",      # Western Pacific
    "SI": "ibtracs.SI.list.v04r01.csv",      # South Indian Ocean
}

def download_ibtracs(basin: str = "NI", output_dir: Path = None) -> Path:
    basin_upper = basin.upper()
    if basin_upper not in VALID_BASINS:
        raise ValueError(f"Unknown basin '{basin}'. Valid basins: {list(VALID_BASINS.keys())}")

    filename = VALID_BASINS[basin_upper]
    url = f"{IBTRACS_BASE_URL}/{filename}"

    if output_dir is None:
        # Default destination: CycloneAI/data/external/
        project_root = Path(__file__).resolve().parents[3]
        output_dir = project_root / "data" / "external"

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / filename

    print(f"[INFO] Source URL: {url}")
    print(f"[INFO] Destination: {destination}")

    # Set up request with standard User-Agent header
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CycloneAI-Research/1.0 (Smart India Hackathon 2026; Academic Prototype)"}
    )

    print("[INFO] Starting download from NOAA NCEI servers...")
    try:
        with urllib.request.urlopen(req) as response, open(destination, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 64  # 64 KB chunks

            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    mb_down = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    sys.stdout.write(f"\r[INFO] Progress: {percent:5.1f}% ({mb_down:.2f} MB / {mb_total:.2f} MB)")
                    sys.stdout.flush()
                else:
                    sys.stdout.write(f"\r[INFO] Downloaded: {downloaded / (1024 * 1024):.2f} MB")
                    sys.stdout.flush()

        print(f"\n[SUCCESS] Successfully downloaded: {destination.name} ({destination.stat().st_size / (1024 * 1024):.2f} MB)")
        return destination

    except urllib.error.HTTPError as e:
        print(f"\n[ERROR] HTTP Error {e.code}: {e.reason}")
        raise
    except Exception as e:
        print(f"\n[ERROR] Download failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download official NOAA NCEI IBTrACS best-track data.")
    parser.add_argument("--basin", type=str, default="NI", help="Basin code (default: NI for North Indian Ocean, or ALL, NA, WP, etc.)")
    args = parser.parse_args()
    download_ibtracs(basin=args.basin)
