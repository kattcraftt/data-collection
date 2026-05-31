import psycopg2
import psycopg2.extras
from sqlalchemy import create_engine
import pandas as pd
import os
from config import DB_CONFIG, DATABASE_URL


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def get_engine():
    return create_engine(DATABASE_URL)


def create_schema():
    conn = get_connection()
    cur = conn.cursor()

    # youtube entity
    cur.execute("""
        CREATE TABLE IF NOT EXISTS raw_youtube_data (
            id              SERIAL PRIMARY KEY,
            video_id        VARCHAR(50),
            comment_id      VARCHAR(100) UNIQUE NOT NULL,
            video_title     TEXT,
            channel_name    VARCHAR(200),
            comment_text    TEXT NOT NULL,
            like_count      INTEGER DEFAULT 0,
            reply_count     INTEGER DEFAULT 0,
            published_at    TIMESTAMP,
            query_used      VARCHAR(500),
            video_category  VARCHAR(100),
            view_count      BIGINT DEFAULT 0,
            video_like_count INTEGER DEFAULT 0,
            collected_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source          VARCHAR(20) DEFAULT 'youtube'
        );
    """)

    # bluesky entity
    cur.execute("""
        CREATE TABLE IF NOT EXISTS raw_bluesky_data (
            id              SERIAL PRIMARY KEY,
            post_cid        VARCHAR(200) UNIQUE NOT NULL,
            post_uri        VARCHAR(500),
            text            TEXT NOT NULL,
            author_handle   VARCHAR(200),
            like_count      INTEGER DEFAULT 0,
            repost_count    INTEGER DEFAULT 0,
            reply_count     INTEGER DEFAULT 0,
            created_at      TIMESTAMP,
            query_used      VARCHAR(500),
            collected_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source          VARCHAR(20) DEFAULT 'bluesky'
        );
    """)

    # hacker news entity
    cur.execute("""
        CREATE TABLE IF NOT EXISTS raw_hackernews_data (
            id              SERIAL PRIMARY KEY,
            hn_id           INTEGER UNIQUE NOT NULL,
            item_type       VARCHAR(20),
            title           TEXT,
            text            TEXT,
            full_text       TEXT,
            score           INTEGER DEFAULT 0,
            comment_count   INTEGER DEFAULT 0,
            author          VARCHAR(100),
            created_at      TIMESTAMP,
            url             TEXT,
            feed_source     VARCHAR(50),
            query_used      VARCHAR(500),
            collected_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source          VARCHAR(20) DEFAULT 'hackernews'
        );
    """)
    # combined dataset
    cur.execute("""
        CREATE TABLE IF NOT EXISTS raw_combined_dataset (
            id              SERIAL PRIMARY KEY,
            original_id     VARCHAR(200) UNIQUE NOT NULL,
            source          VARCHAR(20) NOT NULL,
            platform_weight VARCHAR(20),
            text            TEXT NOT NULL,
            score           INTEGER DEFAULT 0,
            engagement      INTEGER DEFAULT 0,
            created_date    TIMESTAMP,
            query_used      VARCHAR(500),
            category        VARCHAR(200),
            collected_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            -- Basic metrics
            text_length     INTEGER,
            word_count      INTEGER,

            -- PRISMA filtering columns (populated later)
            is_duplicate        BOOLEAN DEFAULT FALSE,
            is_english          BOOLEAN,
            keyword_match       BOOLEAN,
            keyword_count       INTEGER DEFAULT 0,
            is_bot              BOOLEAN DEFAULT FALSE,
            prisma_included     BOOLEAN,
            prisma_stage        VARCHAR(50),
            exclusion_reason    VARCHAR(200),

            -- NLP output columns (populated later)
            sentiment_negative  FLOAT,
            sentiment_neutral   FLOAT,
            sentiment_positive  FLOAT,
            sentiment_label     VARCHAR(20),
            topic_id            INTEGER,
            topic_probability   FLOAT,
            topic_label         VARCHAR(500),
            predicted_behavior  VARCHAR(50)
        );
    """)

    # logs
    cur.execute("""
        CREATE TABLE IF NOT EXISTS collection_log (
            id              SERIAL PRIMARY KEY,
            platform        VARCHAR(20),
            platform_weight VARCHAR(20),
            query_used      VARCHAR(500),
            records_fetched INTEGER DEFAULT 0,
            collected_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status          VARCHAR(50) DEFAULT 'success',
            error_message   TEXT,
            duration_seconds FLOAT
        );
    """)

    # collection summary view
    cur.execute("""
        CREATE OR REPLACE VIEW collection_summary AS
        SELECT
            source,
            platform_weight,
            COUNT(*) as total_records,
            AVG(text_length) as avg_text_length,
            AVG(score) as avg_score,
            MIN(created_date) as earliest_record,
            MAX(created_date) as latest_record,
            MAX(collected_at) as last_collected
        FROM raw_combined_dataset
        GROUP BY source, platform_weight
        ORDER BY total_records DESC;
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("[DB] Schema created successfully.")


def bulk_insert(table: str, records: list,
                conflict_col: str) -> int:
    if not records:
        return 0

    conn = get_connection()
    cur = conn.cursor()


    columns = records[0].keys()
    col_str = ", ".join(columns)
    placeholder = ", ".join([f"%({c})s" for c in columns])

    query = f"""
        INSERT INTO {table} ({col_str})
        VALUES ({placeholder})
        ON CONFLICT ({conflict_col}) DO NOTHING;     
    """

    inserted = 0
    for rec in records:
        try:
            cur.execute(query, rec)
            inserted += cur.rowcount
        except Exception as e:
            print(f"[DB] Insert error in {table}: {e}")
            conn.rollback()

    conn.commit()
    cur.close()
    conn.close()
    return inserted


def insert_combined_records(records: list) -> int:
    return bulk_insert(
        "raw_combined_dataset",
        records,
        "original_id"
    )


def log_collection(platform: str, weight: str,
                   query: str, count: int,
                   status: str = "success",
                   error: str = None,
                   duration: float = 0.0):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO collection_log (
            platform, platform_weight, query_used,
            records_fetched, status,
            error_message, duration_seconds
        ) VALUES (%s, %s, %s, %s, %s, %s, %s);
    """, (platform, weight, query,
          count, status, error, duration))
    conn.commit()
    cur.close()
    conn.close()


def export_to_csv(table: str, path: str) -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql(f"SELECT * FROM {table}", engine)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[CSV] {table}: {len(df)} records → {path}")
    return df


def get_collection_stats() -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql("SELECT * FROM collection_summary", engine)


# prisma filter screening

def update_prisma_columns(record_id: int,
                          updates: dict):
    """Update PRISMA filtering columns for a record."""
    conn = get_connection()
    cur  = conn.cursor()

    set_clause = ", ".join([
        f"{col} = %({col})s"
        for col in updates.keys()
    ])

    updates["id"] = record_id

    cur.execute(f"""
        UPDATE raw_combined_dataset
        SET {set_clause}
        WHERE id = %(id)s;
    """, updates)

    conn.commit()
    cur.close()
    conn.close()


def bulk_update_prisma(records: list):
    """
    Bulk update PRISMA columns for multiple records.
    records = list of dicts with 'id' and filter columns
    """
    if not records:
        return 0

    conn    = get_connection()
    cur     = conn.cursor()
    updated = 0

    for rec in records:
        try:
            rec_id  = rec.pop("id")
            set_str = ", ".join([
                f"{k} = %({k})s"
                for k in rec.keys()
            ])
            rec["id"] = rec_id

            cur.execute(f"""
                UPDATE raw_combined_dataset
                SET {set_str}
                WHERE id = %(id)s;
            """, rec)
            updated += cur.rowcount

        except Exception as e:
            print(f"  [DB] Update error id={rec_id}: {e}")
            conn.rollback()

    conn.commit()
    cur.close()
    conn.close()
    return updated


def get_all_raw_records() -> pd.DataFrame:
    """Fetch all raw records for PRISMA filtering."""
    engine = get_engine()
    return pd.read_sql("""
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


def get_prisma_stats() -> dict:
    """Get PRISMA stage counts for flow diagram."""
    conn = get_connection()
    cur  = conn.cursor()

    stats = {}

    # total identified
    cur.execute(
        "SELECT COUNT(*) FROM raw_combined_dataset;"
    )
    stats["n_identified"] = cur.fetchone()[0]

    # after duplicate removal
    cur.execute("""
        SELECT COUNT(*) FROM raw_combined_dataset
        WHERE is_duplicate = FALSE;
    """)
    stats["n_after_dedup"] = cur.fetchone()[0]

    # after bot removal
    cur.execute("""
        SELECT COUNT(*) FROM raw_combined_dataset
        WHERE is_duplicate = FALSE
        AND is_bot = FALSE;
    """)
    stats["n_after_bot"] = cur.fetchone()[0]

    # after language filter
    cur.execute("""
        SELECT COUNT(*) FROM raw_combined_dataset
        WHERE is_duplicate = FALSE
        AND is_bot = FALSE
        AND is_english = TRUE;
    """)
    stats["n_after_language"] = cur.fetchone()[0]

    # after keyword filter
    cur.execute("""
        SELECT COUNT(*) FROM raw_combined_dataset
        WHERE is_duplicate = FALSE
        AND is_bot = FALSE
        AND is_english = TRUE
        AND keyword_match = TRUE;
    """)
    stats["n_after_keyword"] = cur.fetchone()[0]

    # final included
    cur.execute("""
        SELECT COUNT(*) FROM raw_combined_dataset
        WHERE prisma_included = TRUE;
    """)
    stats["n_included"] = cur.fetchone()[0]

    # exclusion counts
    cur.execute("""
        SELECT exclusion_reason, COUNT(*) as count
        FROM raw_combined_dataset
        WHERE prisma_included = FALSE
        AND exclusion_reason IS NOT NULL
        GROUP BY exclusion_reason
        ORDER BY count DESC;
    """)
    exclusions = cur.fetchall()
    stats["exclusions"] = {
        row[0]: row[1] for row in exclusions
    }

    # per source breakdown
    cur.execute("""
        SELECT
            source,
            COUNT(*) as total,
            SUM(CASE WHEN prisma_included = TRUE
                THEN 1 ELSE 0 END) as included,
            SUM(CASE WHEN prisma_included = FALSE
                THEN 1 ELSE 0 END) as excluded
        FROM raw_combined_dataset
        GROUP BY source
        ORDER BY total DESC;
    """)
    source_stats = cur.fetchall()
    stats["per_source"] = [
        {
            "source"  : row[0],
            "total"   : row[1],
            "included": row[2],
            "excluded": row[3]
        }
        for row in source_stats
    ]

    cur.close()
    conn.close()
    return stats