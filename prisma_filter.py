# PRISMA 2020 Filtering Pipeline
#
# Reads   : raw_combined_dataset (PostgreSQL)
# Updates : filtering columns in place
# Exports : included/excluded CSV + flow log
#
# Inclusion Criteria (IC):
#   IC-1: Relates to digital market consumer behavior
#   IC-2: Published between 2021 and 2026
#   IC-3: Text is in English
#   IC-4: Contains at least 1 target keyword
#   IC-5: From reputable platform (YouTube/Reddit/etc.)
#
# Exclusion Criteria (EC):
#   EC-1: Duplicate text entry
#   EC-2: Bot-generated or spam content
#   EC-3: Not in English
#   EC-4: Text too short (< 30 characters)
#   EC-5: No target keywords found
#   EC-6: Outside 2021-2026 date range
#   EC-7: Contains explicit or spam keywords

import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from langdetect import detect, LangDetectException
from database import get_engine, get_connection, export_to_csv
from prisma_keywords import INCLUSION_KEYWORDS, EXCLUSION_KEYWORDS
from config import CSV_OUTPUT_DIR

# settings
MIN_TEXT_LENGTH = 30      # EC-4 threshold
MIN_YEAR = 2021           # IC-2 start year
MAX_YEAR = 2026           # IC-2 end year
MIN_KEYWORD_COUNT = 1     # IC-4 minimum keywords
BATCH_SIZE = 500          # records per DB update batch
PRISMA_OUTPUT_DIR = f"{CSV_OUTPUT_DIR}/prisma"

# criteria labels for logging
IC_LABELS = {
    "IC1": "Relates to digital market consumer behavior",
    "IC2": f"Published between {MIN_YEAR} and {MAX_YEAR}",
    "IC3": "Text is in English",
    "IC4": f"Contains >= {MIN_KEYWORD_COUNT} target keyword",
    "IC5": "From reputable platform"
}

EC_LABELS = {
    "EC1": "Duplicate text entry",
    "EC2": "Bot-generated or spam content",
    "EC3": "Not in English",
    "EC4": f"Text too short (< {MIN_TEXT_LENGTH} chars)",
    "EC5": "No target keywords found",
    "EC6": f"Outside {MIN_YEAR}-{MAX_YEAR} date range",
    "EC7": "Contains explicit or spam keywords"
}

def load_raw_records() -> pd.DataFrame:
    """Load all raw records from PostgreSQL."""
    engine = get_engine()
    df = pd.read_sql("""
        SELECT
            id,
            original_id,
            source,
            platform_weight,
            text,
            score,
            engagement,
            created_date,
            query_used,
            category,
            text_length,
            word_count,
            collected_at
        FROM raw_combined_dataset
        ORDER BY id ASC;
    """, engine)
    return df


def save_filter_results(df: pd.DataFrame) -> int:
    """
    Write all PRISMA filter decisions back to PostgreSQL
    in batches for efficiency.
    """
    cols_to_update = [
        "id",
        "is_duplicate",
        "is_bot",
        "is_english",
        "keyword_match",
        "keyword_count",
        "prisma_included",
        "prisma_stage",
        "exclusion_reason"
    ]

    # only use columns that exist
    available = [c for c in cols_to_update
                 if c in df.columns]
    records   = df[available].copy()

    # convert NaN to None for PostgreSQL compatibility
    records = records.where(pd.notnull(records), None)

    # ensure correct types
    bool_cols = ["is_duplicate", "is_bot",
                 "is_english", "keyword_match",
                 "prisma_included"]
    for col in bool_cols:
        if col in records.columns:
            records[col] = records[col].apply(
                lambda x: bool(x)
                if x is not None else None
            )

    if "keyword_count" in records.columns:
        records["keyword_count"] = records[
            "keyword_count"
        ].apply(
            lambda x: int(x) if x is not None else 0
        )

    records_list = records.to_dict("records")
    total        = len(records_list)
    updated      = 0

    print(f"\n  Saving {total:,} records to PostgreSQL...")

    conn = get_connection()
    cur  = conn.cursor()

    for i in tqdm(range(0, total, BATCH_SIZE),
                  desc="  DB Update"):
        batch = records_list[i:i + BATCH_SIZE]

        for rec in batch:
            try:
                rec_id = rec.pop("id")

                set_clause = ", ".join([
                    f"{k} = %({k})s"
                    for k in rec.keys()
                ])

                rec["id"] = rec_id

                cur.execute(f"""
                    UPDATE raw_combined_dataset
                    SET {set_clause}
                    WHERE id = %(id)s;
                """, rec)

                updated += cur.rowcount

            except Exception as e:
                conn.rollback()
                continue

        conn.commit()

    cur.close()
    conn.close()

    print(f"  ✓ Updated {updated:,} records")
    return updated

# filter functions
def safe_detect_language(text: str) -> str:
    """Detect language safely, returns 'unknown' on failure."""
    try:
        if not isinstance(text, str):
            return "unknown"
        if len(text.strip()) < 20:
            return "unknown"
        return detect(text)
    except LangDetectException:
        return "unknown"
    except Exception:
        return "unknown"


def count_inclusion_keywords(text: str) -> int:
    """Count how many inclusion keywords appear in text."""
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    return sum(
        1 for kw in INCLUSION_KEYWORDS
        if kw in text_lower
    )


def has_exclusion_keywords(text: str) -> bool:
    """Check if text contains any exclusion keywords."""
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(
        kw in text_lower
        for kw in EXCLUSION_KEYWORDS
    )


def is_bot_or_spam(text: str) -> bool:
    """Detect bot-generated or spam content."""
    if not isinstance(text, str):
        return True

    # check exclusion keywords first
    if has_exclusion_keywords(text):
        return True

    # bot pattern detection
    patterns = [
        r"(.)\1{9,}",               # 10+ repeated characters
        r"(https?://\S+\s*){3,}",   # 3+ URLs in one post
        r"^\s*$",                   # empty or whitespace only
        r"^[\W\d\s]{0,15}$"         # only symbols/numbers
    ]

    for pattern in patterns:
        try:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        except Exception:
            continue

    return False


def check_date_range(created_date) -> bool:
    """Check if record falls within MIN_YEAR to MAX_YEAR."""
    try:
        if pd.isna(created_date):
            return True  # keep if no date available
        year = pd.to_datetime(created_date).year
        return MIN_YEAR <= year <= MAX_YEAR
    except Exception:
        return True  # keep if date is unparseable


def assign_exclusion_reason(row: pd.Series) -> str:
    """Assign the primary exclusion reason for a record."""
    if row.get("prisma_included", False):
        return None

    if row.get("is_duplicate", False):
        return EC_LABELS["EC1"]

    if row.get("is_bot", False):
        return EC_LABELS["EC2"]

    text = str(row.get("text", ""))
    if len(text) < MIN_TEXT_LENGTH:
        return EC_LABELS["EC4"]

    if not row.get("is_english", True):
        return EC_LABELS["EC3"]

    if not row.get("in_date_range", True):
        return EC_LABELS["EC6"]

    if not row.get("keyword_match", False):
        return EC_LABELS["EC5"]

    return "Other"


def assign_prisma_stage(row: pd.Series) -> str:
    """Assign which PRISMA stage removed the record."""
    if row.get("prisma_included", False):
        return "included"

    if row.get("is_duplicate", False) or \
       row.get("is_bot", False):
        return "screening"

    if not row.get("is_english", True) or \
       not row.get("in_date_range", True) or \
       not row.get("keyword_match", False):
        return "eligibility"

    return "excluded_other"

# stage 1 — identification
def stage_1_identification(df: pd.DataFrame):
    print("\n" + "=" * 55)
    print("  Stage 1 — Identification")
    print("=" * 55)

    total      = len(df)
    per_source = df["source"].value_counts()

    print(f"\n  Total records identified : {total:,}")
    print(f"\n  Breakdown by platform:")
    print(f"  {'Platform':<20} {'Records':>10} "
          f"{'Weight':<15}")
    print(f"  {'-'*20} {'-'*10} {'-'*15}")

    for src in per_source.index:
        count  = per_source[src]
        weight = df[df["source"] == src][
            "platform_weight"
        ].iloc[0] if len(
            df[df["source"] == src]
        ) > 0 else "unknown"

        print(f"  {src:<20} {count:>10,} {weight:<15}")

    print(f"\n  n_identified = {total:,}")

    return total

# stage 2 — screening
# remove duplicates and bot/spam content
def stage_2_screening(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("  Stage 2 — Screening")
    print("=" * 55)

    n_before = len(df)

    # 2.1 duplicate detection
    print("\n  [2.1] Detecting duplicates (EC-1)...")

    df["text_clean"] = (
        df["text"]
        .fillna("")
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    df["is_duplicate"] = df.duplicated(
        subset = ["text_clean"],
        keep   = "first"
    )

    n_dupl = int(df["is_duplicate"].sum())
    print(f"    Duplicates found     : {n_dupl:,}")
    print(f"    Unique records       : "
          f"{n_before - n_dupl:,}")

    # 2.2 bot and spam detection
    print("\n  [2.2] Detecting bot/spam content (EC-2, EC-7)...")

    tqdm.pandas(desc="    Scanning")
    df["is_bot"] = df["text"].progress_apply(
        is_bot_or_spam
    )

    n_bot = int(df["is_bot"].sum())
    print(f"    Bot/spam detected    : {n_bot:,}")

    # 2.3 short text removal
    print(f"\n  [2.3] Removing short texts "
          f"(< {MIN_TEXT_LENGTH} chars) (EC-4)...")

    df["text_len_check"] = (
        df["text"].fillna("").str.len() >= MIN_TEXT_LENGTH
    )

    n_short = int((~df["text_len_check"]).sum())
    print(f"    Too short            : {n_short:,}")

    # screening pass flag
    df["screen_pass"] = (
        ~df["is_duplicate"] &
        ~df["is_bot"]       &
        df["text_len_check"]
    )

    n_pass    = int(df["screen_pass"].sum())
    n_removed = n_before - n_pass

    print(f"\n  {'─'*45}")
    print(f"  Records before screening  : {n_before:,}")
    print(f"  Removed (duplicates)      : {n_dupl:,}")
    print(f"  Removed (bot/spam)        : {n_bot:,}")
    print(f"  Removed (too short)       : {n_short:,}")
    print(f"  Records after screening   : {n_pass:,}")
    print(f"  Total removed             : {n_removed:,}")

    # clean up helper column
    df.drop(columns=["text_clean",
                     "text_len_check"],
            inplace=True,
            errors="ignore")

    return df

# stage 3 — eligibility
# language detection, date range, keyword matching
def stage_3_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("  Stage 3 — Eligibility")
    print("=" * 55)

    # only process records that passed screening
    mask         = df["screen_pass"].fillna(False)
    n_entering   = int(mask.sum())
    print(f"\n  Records entering eligibility : {n_entering:,}")

    # 3.1 language detection
    print("\n  [3.1] Language detection (IC-3, EC-3)...")

    tqdm.pandas(desc="    Detecting language")
    df.loc[mask, "detected_lang"] = (
        df.loc[mask, "text"]
        .progress_apply(safe_detect_language)
    )

    df["is_english"] = (
        df["detected_lang"].fillna("unknown") == "en"
    )

    n_english     = int((mask & df["is_english"]).sum())
    n_non_english = int((mask & ~df["is_english"]).sum())
    print(f"    English confirmed    : {n_english:,}")
    print(f"    Non-English removed  : {n_non_english:,}")

    # 3.2 Date Range Check
    print(f"\n  [3.2] Date range check "
          f"({MIN_YEAR}-{MAX_YEAR}) (IC-2, EC-6)...")

    lang_pass = mask & df["is_english"]

    df.loc[lang_pass, "in_date_range"] = (
        df.loc[lang_pass, "created_date"]
        .apply(check_date_range)
    )

    # fill True for records not checked
    df["in_date_range"] = (
        df["in_date_range"].fillna(True)
    )

    n_out_range = int(
        (lang_pass & ~df["in_date_range"]).sum()
    )
    print(f"    Out of range removed : {n_out_range:,}")
    print(f"    In date range        : "
          f"{int((lang_pass & df['in_date_range']).sum()):,}")

    # 3.3 keyword matching
    print(f"\n  [3.3] Keyword matching (IC-4, EC-5)...")
    print(f"    Checking {len(INCLUSION_KEYWORDS)} keywords...")

    date_pass = lang_pass & df["in_date_range"]

    tqdm.pandas(desc="    Matching keywords")
    df.loc[date_pass, "keyword_count"] = (
        df.loc[date_pass, "text"]
        .progress_apply(count_inclusion_keywords)
        .astype(int)
    )

    df["keyword_count"] = (
        df["keyword_count"].fillna(0).astype(int)
    )
    df["keyword_match"] = (
        df["keyword_count"] >= MIN_KEYWORD_COUNT
    )

    n_kw_match  = int((date_pass & df["keyword_match"]).sum())
    n_kw_fail   = int((date_pass & ~df["keyword_match"]).sum())
    print(f"    Keyword match        : {n_kw_match:,}")
    print(f"    No keyword match     : {n_kw_fail:,}")

    # eligibility pass flag
    df["eligibility_pass"] = (
        mask               &
        df["is_english"]   &
        df["in_date_range"]&
        df["keyword_match"]
    )

    n_elig_pass = int(df["eligibility_pass"].sum())
    n_elig_fail = n_entering - n_elig_pass

    print(f"\n  {'─'*45}")
    print(f"  Records entering eligibility : {n_entering:,}")
    print(f"  Removed (non-English)        : {n_non_english:,}")
    print(f"  Removed (out of date range)  : {n_out_range:,}")
    print(f"  Removed (no keyword match)   : {n_kw_fail:,}")
    print(f"  Records passing eligibility  : {n_elig_pass:,}")
    print(f"  Total removed                : {n_elig_fail:,}")

    return df

# stage 4 — inclusion
# final decision + assign labels
def stage_4_inclusion(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("  Stage 4 — Inclusion")
    print("=" * 55)

    # final inclusion = passed all previous stages
    df["prisma_included"] = (
        df.get("eligibility_pass",
               pd.Series(False, index=df.index))
        .fillna(False)
    )

    # assign exclusion reasons
    print("\n  Assigning exclusion reasons...")
    tqdm.pandas(desc="    Labeling")
    df["exclusion_reason"] = df.progress_apply(
        assign_exclusion_reason, axis=1
    )

    # assign PRISMA stage labels
    df["prisma_stage"] = df.apply(
        assign_prisma_stage, axis=1
    )

    n_included = int(df["prisma_included"].sum())
    n_excluded = len(df) - n_included

    print(f"\n  Final included records : {n_included:,}")
    print(f"  Final excluded records : {n_excluded:,}")

    # exclusion breakdown
    excl = (
        df[~df["prisma_included"]]["exclusion_reason"]
        .value_counts()
    )

    print(f"\n  Exclusion breakdown:")
    print(f"  {'Reason':<50} {'Count':>8}")
    print(f"  {'-'*50} {'-'*8}")

    for reason, count in excl.items():
        if reason:
            print(f"  {str(reason)[:50]:<50} {count:>8,}")

    # per source breakdown
    print(f"\n  Inclusion by platform:")
    print(f"  {'Platform':<20} {'Total':>8} "
          f"{'Included':>10} {'Excluded':>10} "
          f"{'Rate':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")

    for src in df["source"].unique():
        src_df   = df[df["source"] == src]
        total    = len(src_df)
        included = int(src_df["prisma_included"].sum())
        excluded = total - included
        rate     = (included / total * 100
                    if total > 0 else 0)
        print(f"  {src:<20} {total:>8,} "
              f"{included:>10,} {excluded:>10,} "
              f"{rate:>7.1f}%")

    return df

# export results
def export_results(df: pd.DataFrame,
                   timestamp: str):
    print("\n" + "=" * 55)
    print("  EXPORTING RESULTS")
    print("=" * 55)

    os.makedirs(PRISMA_OUTPUT_DIR, exist_ok=True)

    # columns to export for NLP stage
    nlp_cols = [
        "id", "original_id", "source",
        "platform_weight", "text",
        "score", "engagement", "created_date",
        "query_used", "category",
        "text_length", "word_count",
        "keyword_count", "keyword_match",
        "is_english", "prisma_stage"
    ]

    available_nlp = [
        c for c in nlp_cols
        if c in df.columns
    ]

    # file 1: included (for NLP pipeline)
    included_df  = df[
        df["prisma_included"] == True
    ][available_nlp].copy()

    path_included = (
        f"{PRISMA_OUTPUT_DIR}/"
        f"prisma_included_{timestamp}.csv"
    )
    included_df.to_csv(path_included, index=False)
    print(f"\n  ✓ Included dataset")
    print(f"    Records : {len(included_df):,}")
    print(f"    File    : {path_included}")

    # file 2: excluded (with reasons)
    excl_cols = [
        "id", "original_id", "source", "text",
        "is_duplicate", "is_bot", "is_english",
        "keyword_match", "keyword_count",
        "prisma_stage", "exclusion_reason",
        "created_date"
    ]
    available_excl = [
        c for c in excl_cols
        if c in df.columns
    ]

    excluded_df  = df[
        df["prisma_included"] == False
    ][available_excl].copy()

    path_excluded = (
        f"{PRISMA_OUTPUT_DIR}/"
        f"prisma_excluded_{timestamp}.csv"
    )
    excluded_df.to_csv(path_excluded, index=False)
    print(f"\n  ✓ Excluded dataset")
    print(f"    Records : {len(excluded_df):,}")
    print(f"    File    : {path_excluded}")

    # file 3: PRISMA flow log (for paper)
    flow_log = build_flow_log(df)
    path_log = (
        f"{PRISMA_OUTPUT_DIR}/"
        f"prisma_flow_log_{timestamp}.csv"
    )
    flow_log.to_csv(path_log, index=False)
    print(f"\n  ✓ PRISMA flow log")
    print(f"    File    : {path_log}")

    # file 4: full dataset with all filter columns
    path_full = (
        f"{PRISMA_OUTPUT_DIR}/"
        f"prisma_full_{timestamp}.csv"
    )
    df.to_csv(path_full, index=False)
    print(f"\n  ✓ Full dataset")
    print(f"    Records : {len(df):,}")
    print(f"    File    : {path_full}")

    return path_included

# build PRISMA flow log
# for paper documentation and PRISMA diagram
def build_flow_log(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    n_total = len(df)

    # stage 1
    rows.append({
        "prisma_stage"  : "1 - Identification",
        "description"   : "Total records from all platforms",
        "count"         : n_total,
        "criteria"      : "N/A"
    })

    for src in df["source"].unique():
        rows.append({
            "prisma_stage": "1 - Identification",
            "description" : f"Records from {src}",
            "count"       : len(df[df["source"] == src]),
            "criteria"    : "N/A"
        })

    # stage 2
    n_dupl = int(df.get("is_duplicate",
                        pd.Series(False,
                        index=df.index)).sum())
    n_bot  = int(df.get("is_bot",
                        pd.Series(False,
                        index=df.index)).sum())
    n_after_screen = n_total - n_dupl - n_bot

    rows.append({
        "prisma_stage": "2 - Screening",
        "description" : "Records after duplicate removal",
        "count"       : n_total - n_dupl,
        "criteria"    : "EC-1"
    })
    rows.append({
        "prisma_stage": "2 - Screening",
        "description" : "Removed — duplicates",
        "count"       : n_dupl,
        "criteria"    : "EC-1"
    })
    rows.append({
        "prisma_stage": "2 - Screening",
        "description" : "Removed — bot/spam",
        "count"       : n_bot,
        "criteria"    : "EC-2"
    })
    rows.append({
        "prisma_stage": "2 - Screening",
        "description" : "Records after screening",
        "count"       : n_after_screen,
        "criteria"    : "EC-1, EC-2"
    })

    # stage 3
    n_non_eng = int((~df.get("is_english",
                    pd.Series(True,
                    index=df.index))).sum())
    n_no_date = int((~df.get("in_date_range",
                    pd.Series(True,
                    index=df.index))).sum())
    n_no_kw   = int((~df.get("keyword_match",
                    pd.Series(True,
                    index=df.index))).sum())
    n_elig    = int(df.get("eligibility_pass",
                    pd.Series(False,
                    index=df.index)).sum())

    rows.append({
        "prisma_stage": "3 - Eligibility",
        "description" : "Removed — non-English",
        "count"       : n_non_eng,
        "criteria"    : "EC-3, IC-3"
    })
    rows.append({
        "prisma_stage": "3 - Eligibility",
        "description" : "Removed — out of date range",
        "count"       : n_no_date,
        "criteria"    : "EC-6, IC-2"
    })
    rows.append({
        "prisma_stage": "3 - Eligibility",
        "description" : "Removed — no keyword match",
        "count"       : n_no_kw,
        "criteria"    : "EC-5, IC-4"
    })
    rows.append({
        "prisma_stage": "3 - Eligibility",
        "description" : "Records passing eligibility",
        "count"       : n_elig,
        "criteria"    : "IC-1 to IC-5"
    })

    # stage 4
    n_included = int(df.get("prisma_included",
                    pd.Series(False,
                    index=df.index)).sum())
    n_excluded = n_total - n_included

    rows.append({
        "prisma_stage": "4 - Inclusion",
        "description" : "Final included records",
        "count"       : n_included,
        "criteria"    : "All IC met"
    })
    rows.append({
        "prisma_stage": "4 - Inclusion",
        "description" : "Total excluded records",
        "count"       : n_excluded,
        "criteria"    : "One or more EC"
    })

    return pd.DataFrame(rows)

# print PRISMA flow diagram
# copy values directly into your paper
def print_flow_diagram(df: pd.DataFrame):
    n_total   = len(df)
    n_dupl    = int(df.get("is_duplicate",
                   pd.Series(False,
                   index=df.index)).sum())
    n_bot     = int(df.get("is_bot",
                   pd.Series(False,
                   index=df.index)).sum())
    n_screen  = n_total - n_dupl - n_bot
    n_non_eng = int((~df.get("is_english",
                   pd.Series(True,
                   index=df.index))).sum())
    n_no_kw   = int((~df.get("keyword_match",
                   pd.Series(True,
                   index=df.index))).sum())
    n_elig    = int(df.get("eligibility_pass",
                   pd.Series(False,
                   index=df.index)).sum())
    n_incl    = int(df.get("prisma_included",
                   pd.Series(False,
                   index=df.index)).sum())
    n_excl    = n_total - n_incl

    # per source counts
    src_lines = ""
    for src in df["source"].unique():
        count     = len(df[df["source"] == src])
        src_lines += f"\n  │  {src:<22}: {count:>6,}         │"

    print("\n" + "=" * 55)
    print("  PRISMA 2020 FLOW DIAGRAM")
    print("  Use these numbers in your research paper")
    print("=" * 55)

    print(f"""
  ┌─────────────────────────────────────────────────┐
  │  Stage 1 — Identification                       │{src_lines}
  │                                                 │
  │  TOTAL IDENTIFIED           : {n_total:>8,}     │
  └────────────────────────┬────────────────────────┘
                           ↓
  ┌─────────────────────────────────────────────────┐
  │  Stage 2 — Screening                            │
  │  Records after screening    : {n_screen:>8,}    │
  │                                                 │
  │  Removed (EC-1 duplicates)  : {n_dupl:>8,}      │
  │  Removed (EC-2 bot/spam)    : {n_bot:>8,}       │
  └────────────────────────┬────────────────────────┘
                           ↓
  ┌─────────────────────────────────────────────────┐
  │  Stage 3 — Eligibility                          │
  │  Records after eligibility  : {n_elig:>8,}      │
  │                                                 │
  │  Removed (EC-3 non-English) : {n_non_eng:>8,}   │
  │  Removed (EC-5 no keywords) : {n_no_kw:>8,}     │
  └────────────────────────┬────────────────────────┘
                           ↓
  ┌─────────────────────────────────────────────────┐
  │  Stage 4 — Inclusion                            │
  │                                                 │
  │  ✓ FINAL INCLUDED           : {n_incl:>8,}      │
  │  ✗ TOTAL EXCLUDED           : {n_excl:>8,}      │
  └─────────────────────────────────────────────────┘
    """)

# main
def main():
    print("\n" + "=" * 55)
    print("  PRISMA 2020 FILTERING PIPELINE")
    print("  Customer Behavior Analysis in Digital Markets")
    print("=" * 55)
    print(f"  Started  : "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\n  Settings:")
    print(f"  Min text length  : {MIN_TEXT_LENGTH} chars")
    print(f"  Date range       : {MIN_YEAR} – {MAX_YEAR}")
    print(f"  Min keywords     : {MIN_KEYWORD_COUNT}")
    print(f"  Keywords loaded  : {len(INCLUSION_KEYWORDS)}")
    print(f"  Batch size       : {BATCH_SIZE}")

    print(f"\n  Inclusion Criteria:")
    for k, v in IC_LABELS.items():
        print(f"    {k}: {v}")

    print(f"\n  Exclusion Criteria:")
    for k, v in EC_LABELS.items():
        print(f"    {k}: {v}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # load raw data
    print("\n[load] Fetching records from PostgreSQL...")
    df = load_raw_records()

    if df is None or df.empty:
        print("\n  ✗ No records found.")
        print("  Run python main.py first.")
        return

    print(f"  ✓ Loaded {len(df):,} records")

    # initialize all filter columns
    df["is_duplicate"]    = False
    df["is_bot"]          = False
    df["is_english"]      = None
    df["detected_lang"]   = None
    df["in_date_range"]   = True
    df["keyword_count"]   = 0
    df["keyword_match"]   = False
    df["screen_pass"]     = False
    df["eligibility_pass"]= False
    df["prisma_included"] = False
    df["prisma_stage"]    = None
    df["exclusion_reason"]= None

    # run PRISMA stages
    stage_1_identification(df)
    df = stage_2_screening(df)
    df = stage_3_eligibility(df)
    df = stage_4_inclusion(df)

    # save to PostgreSQL
    print("\n[save] Writing filter decisions to PostgreSQL...")
    save_filter_results(df)

    # export esv
    print("\n[export] Exporting results to CSV...")
    export_results(df, timestamp)

    # print flow diagram
    print_flow_diagram(df)

    # final summary
    n_included = int(df["prisma_included"].sum())
    n_total    = len(df)
    rate       = n_included / n_total * 100 \
                 if n_total > 0 else 0

    print("=" * 55)
    print("  PRISMA FILTERING COMPLETE")
    print("=" * 55)
    print(f"  Total raw records  : {n_total:,}")
    print(f"  Included records   : {n_included:,}")
    print(f"  Excluded records   : {n_total - n_included:,}")
    print(f"  Inclusion rate     : {rate:.1f}%")
    print(f"  Output directory   : {PRISMA_OUTPUT_DIR}/")
    print(f"  Timestamp          : {timestamp}")
    print("=" * 55)
    print("\n  Next step → Run: python nlp_extraction.py")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()