import os
import time as time_lib
import pandas as pd
from datetime import datetime
from database import create_schema, export_to_csv, get_collection_stats
from collectors.youtube_collector import collect_youtube
from collectors.bluesky_collector import collect_bluesky
from collectors.hackernews_collector import collect_hackernews
from config import CSV_OUTPUT_DIR


def print_banner():
    print("\n" + "=" * 60)
    print("  DATA COLLECTION PIPELINE")
    print("  Research: Predictive Insights Through Big Data")
    print("  Analytics: Enhancing Customer Behavior Analysis")
    print("  in Digital Markets (2027-2030)")
    print("=" * 60)
    print(f"  Started : "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("\n  Platform Priority:")
    print("  [1] YouTube     → PRIMARY SOURCE")
    print("  [2] Bluesky     → SECONDARY SOURCE")
    print("  [3] HackerNews  → TERTIARY SOURCE")
    print("=" * 60)


def run_collector(name: str,
                  collector_fn) -> dict:
    """
    Runs a single collector safely and returns
    a result dict with records, duration, status.
    Always returns float duration — never a string.
    """
    print(f"\n[→] Running {name} collector...")
    start = time_lib.time()

    try:
        df       = collector_fn()
        duration = float(time_lib.time() - start)
        records  = len(df) if df is not None else 0

        return {
            "records" : records,
            "duration": duration,   # always float
            "status"  : "success"
        }

    except Exception as e:
        duration = float(time_lib.time() - start)
        print(f"  [✗] {name} fatal error: {e}")
        return {
            "records" : 0,
            "duration": duration,   # always float
            "status"  : f"failed"
        }


def main():
    print_banner()
    pipeline_start = time_lib.time()

    # database init
    print("\n[SETUP] Initializing PostgreSQL schema...")
    try:
        create_schema()
        print("  ✓ Schema ready")
    except Exception as e:
        print(f"  ✗ Schema error: {e}")
        return

    # run collectors
    results = {}

    results["youtube"] = run_collector(
        "YouTube (Primary)",
        collect_youtube
    )

    results["bluesky"] = run_collector(
        "Bluesky (Secondary)",
        collect_bluesky
    )

    results["hackernews"] = run_collector(
        "HackerNews (Tertiary)",
        collect_hackernews
    )

    # export csv
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

    print("\n[export] Generating CSV exports...")

    exports = [
        ("raw_youtube_data", f"youtube_{timestamp}.csv"),
        ("raw_bluesky_data", f"bluesky_{timestamp}.csv"),
        ("raw_hackernews_data", f"hackernews_{timestamp}.csv"),
        ("raw_combined_dataset", f"combined_raw_{timestamp}.csv"),
        ("collection_log", f"collection_log_{timestamp}.csv")
    ]

    for table, filename in exports:
        try:
            export_to_csv(
                table,
                f"{CSV_OUTPUT_DIR}/{filename}"
            )
        except Exception as e:
            print(f"  ✗ Export failed [{table}]: {e}")

    # summary
    pipeline_duration = float(time_lib.time() - pipeline_start)
    total_records     = sum(
        v.get("records", 0) for v in results.values()
    )

    weight_map = {
        "youtube"   : "(primary)",
        "bluesky"   : "(secondary)",
        "hackernews": "(tertiary)"
    }

    print("\n" + "=" * 60)
    print(" COLLECTION PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  {'Platform':<28} "
          f"{'Records':>8} "
          f"{'Duration':>10} "
          f"{'Status'}")
    print(f"  {'-'*28} "
          f"{'-'*8} "
          f"{'-'*10} "
          f"{'-'*12}")

    for platform, data in results.items():
        label   = f"{platform} {weight_map.get(platform, '')}"
        records = int(data.get("records", 0) or 0)
        status  = str(data.get("status",  "unknown"))

        try:
            duration = float(data.get("duration", 0) or 0)
            dur_str  = f"{duration:.1f}s"
        except (TypeError, ValueError):
            dur_str  = "N/A"

        print(f"  {label:<28} "
              f"{records:>8,} "
              f"{dur_str:>10} "
              f"{status}")

    print(f"  {'─'*60}")
    print(f"  {'TOTAL':<28} {total_records:>8,} "
          f"{pipeline_duration:>9.1f}s")
    print("=" * 60)

    # database stats
    print("\n[database] Collection summary:")
    try:
        stats = get_collection_stats()
        if stats is not None and not stats.empty:
            print(stats.to_string(index=False))
        else:
            print("  No data in database yet.")
    except Exception as e:
        print(f"  Could not fetch DB stats: {e}")

    print("\n" + "=" * 60)
    print(f"  CSV output : {CSV_OUTPUT_DIR}/")
    print(f"  Timestamp  : {timestamp}")
    print(f"  Total time : {pipeline_duration:.1f}s")
    print("=" * 60)
    print("\n[done] Raw dataset collection complete.")
    print("  Next step → Run: python prisma_filter.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()