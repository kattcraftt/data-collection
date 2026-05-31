import requests
import pandas as pd
from datetime import datetime
import time as time_lib
from tqdm import tqdm
from config import HACKERNEWS_CONFIG, HACKERNEWS_QUERIES, HACKERNEWS_FEEDS, HACKERNEWS_LIMIT
from database import bulk_insert, insert_combined_records, log_collection

# keep both URLs from config
BASE_URL   = HACKERNEWS_CONFIG["base_url"]   # Firebase
SEARCH_URL = HACKERNEWS_CONFIG["search_url"] # Algolia

# rate limit settings
DELAY_BETWEEN_QUERIES = 0.3   # seconds between Algolia calls
DELAY_BETWEEN_ITEMS   = 0.02  # reduced from 0.05 for Firebase
MAX_RETRIES           = 3     # retry attempts on failure

# safe integer conversion
def safe_int(value, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default

# safe datetime conversion
def safe_datetime(value) -> datetime:
    if value is None:
        return datetime.utcnow()
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )
        except Exception:
            return datetime.utcnow()
    return datetime.utcnow()

# safe ID extraction
def safe_hn_id(hit: dict) -> int:
    """
    Extract HN ID safely from multiple possible fields.
    objectID from Algolia can be a string like '12345'.
    """
    raw = (
        hit.get("objectID")   or
        hit.get("story_id")   or
        hit.get("id")         or
        hit.get("parent_id")  or
        0
    )
    try:
        return int(raw)
    except (TypeError, ValueError):
        # Hash the string ID as fallback
        return abs(hash(str(raw))) % (10 ** 9)

# get single item from Firebase (with retry)
def get_item(item_id: int) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(
                f"{BASE_URL}/item/{item_id}.json",
                timeout=10
            )
            if r.status_code == 200:
                return r.json()
            return None
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time_lib.sleep(1)
            continue
    return None

# algolia search (no tag filter = max results)
def search_hn(query: str,
              limit: int = 100,
              page:  int = 0) -> list:
    """
    Search via Algolia API.
    NOTE: No 'tags' filter — using tags caused 0 results.
    Keeping BASE_URL (Firebase) and SEARCH_URL (Algolia) both.
    """
    try:
        r = requests.get(
            f"{SEARCH_URL}/search",
            params={
                "query"      : query,
                "hitsPerPage": limit,
                "page"       : page
                # tags filter intentionally removed
                # it was causing 0 results
            },
            timeout=15
        )
        if r.status_code == 200:
            data = r.json()
            return data.get("hits", [])
        else:
            print(f"  [HN Algolia] Status {r.status_code} "
                  f"for '{query}'")
            return []
    except Exception as e:
        print(f"  [HN Algolia] Error: {e}")
        return []

# recent items via algolia search_by_date
# (uses SEARCH_URL — algolia)
def get_recent_algolia(item_type: str = "story",
                       limit: int = 200) -> list:
    try:
        r = requests.get(
            f"{SEARCH_URL}/search_by_date",
            params={
                "tags"       : item_type,
                "hitsPerPage": limit
            },
            timeout=15
        )
        if r.status_code == 200:
            return r.json().get("hits", [])
    except Exception as e:
        print(f"  [HN Recent Algolia] Error: {e}")
    return []

# process Algolia hit into records
def process_algolia_hit(hit: dict,
                        query: str,
                        feed_source: str) -> tuple:
    try:
        title        = str(hit.get("title", "")        or "")
        story_text   = str(hit.get("story_text", "")   or "")
        comment_text = str(hit.get("comment_text", "") or "")
        text_field   = str(hit.get("text", "")         or "")

        # Combine all available text fields
        full_text = " ".join(filter(None, [
            title,
            story_text,
            comment_text,
            text_field
        ])).strip()

        if len(full_text) < 20:
            return None, None

        hn_id   = safe_hn_id(hit)
        if hn_id == 0:
            return None, None

        tags      = hit.get("_tags", []) or []
        item_type = str(tags[0]) if tags else "story"
        created   = safe_datetime(hit.get("created_at"))
        score     = safe_int(hit.get("points"))
        comments  = safe_int(hit.get("num_comments"))
        author    = str(hit.get("author", "unknown") or
                        "unknown")[:100]
        url       = str(hit.get("url", "") or "")[:500]

        hn_rec = {
            "hn_id"        : hn_id,
            "item_type"    : item_type[:20],
            "title"        : title[:500],
            "text"         : (story_text or
                              comment_text or
                              text_field)[:5000],
            "full_text"    : full_text[:5000],
            "score"        : score,
            "comment_count": comments,
            "author"       : author,
            "created_at"   : created,
            "url"          : url,
            "feed_source"  : feed_source,
            "query_used"   : query
        }

        combined_rec = {
            "original_id"    : f"hn_{hn_id}",
            "source"         : "hackernews",
            "platform_weight": "tertiary",
            "text"           : full_text[:5000],
            "score"          : score,
            "engagement"     : comments,
            "created_date"   : created,
            "query_used"     : query,
            "category"       : item_type[:100],
            "text_length"    : len(full_text),
            "word_count"     : len(full_text.split())
        }

        return hn_rec, combined_rec

    except Exception as e:
        print(f"  [hackernews] process_hit error: {e}")
        return None, None

# process Firebase item into records
def process_firebase_item(item: dict,
                          feed: str) -> tuple:
    try:
        title     = str(item.get("title", "") or "")
        text      = str(item.get("text",  "") or "")
        full_text = (title + " " + text).strip()

        if len(full_text) < 20:
            return None, None

        hn_id   = safe_int(item.get("id"))
        if hn_id == 0:
            return None, None

        created = datetime.utcfromtimestamp(
            safe_int(item.get("time"))
        )
        score    = safe_int(item.get("score"))
        comments = safe_int(item.get("descendants"))
        author   = str(item.get("by", "unknown") or
                       "unknown")[:100]
        url      = str(item.get("url", "") or "")[:500]

        hn_rec = {
            "hn_id"        : hn_id,
            "item_type"    : str(item.get("type",
                                          "story"))[:20],
            "title"        : title[:500],
            "text"         : text[:5000],
            "full_text"    : full_text[:5000],
            "score"        : score,
            "comment_count": comments,
            "author"       : author,
            "created_at"   : created,
            "url"          : url,
            "feed_source"  : feed,
            "query_used"   : feed
        }

        combined_rec = {
            "original_id"    : f"hn_{hn_id}",
            "source"         : "hackernews",
            "platform_weight": "tertiary",
            "text"           : full_text[:5000],
            "score"          : score,
            "engagement"     : comments,
            "created_date"   : created,
            "query_used"     : feed,
            "category"       : feed,
            "text_length"    : len(full_text),
            "word_count"     : len(full_text.split())
        }

        return hn_rec, combined_rec

    except Exception as e:
        print(f"  [HN Firebase] process error: {e}")
        return None, None

def collect_hackernews() -> pd.DataFrame:
    print("\n" + "=" * 55)
    print(" [hackernews] TERTIARY SOURCE — Starting Collection")
    print(f"  BASE_URL   (Firebase): {BASE_URL}")
    print(f"  SEARCH_URL (Algolia) : {SEARCH_URL}")
    print("=" * 55)

    all_hn       = []
    all_combined = []

    # keyword search via algolia
    print("\n  [hackernews] Phase 1: Keyword search (Algolia)...")

    for query in tqdm(HACKERNEWS_QUERIES,
                      desc="HN Search"):
        start_time = time_lib.time()

        try:
            # two pages for more results
            hits_p0 = search_hn(query, limit=100, page=0)
            hits_p1 = search_hn(query, limit=100, page=1)
            all_hits = hits_p0 + hits_p1

            batch_hn       = []
            batch_combined = []

            for hit in all_hits:
                hn_rec, combined_rec = process_algolia_hit(
                    hit, query, "keyword_search"
                )
                if hn_rec and combined_rec:
                    batch_hn.append(hn_rec)
                    batch_combined.append(combined_rec)

            if batch_hn:
                bulk_insert(
                    "raw_hackernews_data",
                    batch_hn,
                    "hn_id"
                )
                insert_combined_records(batch_combined)
                all_hn.extend(batch_hn)
                all_combined.extend(batch_combined)

            duration = time_lib.time() - start_time
            log_collection(
                "hackernews", "tertiary",
                query, len(batch_hn),
                duration=duration
            )

            print(f"  ✓ '{query[:35]}': "
                  f"{len(batch_hn)} items "
                  f"({duration:.1f}s)")

            time_lib.sleep(DELAY_BETWEEN_QUERIES)

        except Exception as e:
            duration = time_lib.time() - start_time
            print(f"  ✗ Search error '{query}': {e}")
            log_collection(
                "hackernews", "tertiary",
                query, 0,
                status="error", error=str(e),
                duration=duration
            )
            continue

    # recent items via algolia
    print("\n  [hackernews] Phase 2: Recent items (Algolia)...")

    for item_type in ["story", "comment"]:
        start_time = time_lib.time()

        try:
            hits = get_recent_algolia(
                item_type = item_type,
                limit     = 200
            )

            batch_hn       = []
            batch_combined = []

            for hit in hits:
                hn_rec, combined_rec = process_algolia_hit(
                    hit,
                    f"recent_{item_type}",
                    "recent_algolia"
                )
                if hn_rec and combined_rec:
                    batch_hn.append(hn_rec)
                    batch_combined.append(combined_rec)

            if batch_hn:
                bulk_insert(
                    "raw_hackernews_data",
                    batch_hn,
                    "hn_id"
                )
                insert_combined_records(batch_combined)
                all_hn.extend(batch_hn)

            duration = time_lib.time() - start_time
            log_collection(
                "hackernews", "tertiary",
                f"recent_{item_type}",
                len(batch_hn),
                duration=duration
            )

            print(f"  ✓ Recent {item_type}s: "
                  f"{len(batch_hn)} items "
                  f"({duration:.1f}s)")

            time_lib.sleep(DELAY_BETWEEN_QUERIES)

        except Exception as e:
            print(f"  ✗ Recent {item_type} error: {e}")
            continue

    # feed collection via firebase
    # uses BASE_URL (firebase) for topstories/newstories feeds
    print("\n  [hackernews] Phase 3: Feed collection (Firebase)...")

    for feed in tqdm(HACKERNEWS_FEEDS, desc="HN Feeds"):
        start_time = time_lib.time()

        try:
            # get feed IDs from firebase
            r = requests.get(
                f"{BASE_URL}/{feed}.json",
                timeout=15
            )

            if r.status_code != 200:
                print(f"  ✗ Feed '{feed}' returned "
                      f"{r.status_code}")
                continue

            story_ids = r.json()[:HACKERNEWS_LIMIT]
            print(f"\n  Feed '{feed}': "
                  f"{len(story_ids)} IDs to fetch...")

            batch_hn       = []
            batch_combined = []

            for story_id in story_ids:
                # each item fetched from firebase individually
                item = get_item(story_id)

                if (not item or
                        item.get("deleted") or
                        item.get("dead")):
                    continue

                hn_rec, combined_rec = process_firebase_item(
                    item, feed
                )
                if hn_rec and combined_rec:
                    batch_hn.append(hn_rec)
                    batch_combined.append(combined_rec)

                # reduced delay — 0.02
                time_lib.sleep(DELAY_BETWEEN_ITEMS)

            if batch_hn:
                bulk_insert(
                    "raw_hackernews_data",
                    batch_hn,
                    "hn_id"
                )
                insert_combined_records(batch_combined)
                all_hn.extend(batch_hn)

            duration = time_lib.time() - start_time
            log_collection(
                "hackernews", "tertiary",
                feed, len(batch_hn),
                duration=duration
            )

            print(f"  ✓ Feed '{feed}': "
                  f"{len(batch_hn)} items "
                  f"({duration:.1f}s)")

        except Exception as e:
            print(f"  ✗ Feed error '{feed}': {e}")
            continue

    # top scored via algolia
    print("\n  [HN] Phase 4: Top scored (Algolia)...")

    top_queries = [
        "ecommerce",
        "consumer behavior",
        "digital market",
        "online shopping",
        "customer review"
    ]

    for q in tqdm(top_queries, desc="HN Top"):
        start_time = time_lib.time()

        try:
            r = requests.get(
                f"{SEARCH_URL}/search",
                params={
                    "query"         : q,
                    "hitsPerPage"   : 100,
                    "numericFilters": "points>5"
                },
                timeout=15
            )

            batch_hn       = []
            batch_combined = []

            if r.status_code == 200:
                hits = r.json().get("hits", [])
                for hit in hits:
                    hn_rec, combined_rec = process_algolia_hit(
                        hit, q, "top_scored"
                    )
                    if hn_rec and combined_rec:
                        batch_hn.append(hn_rec)
                        batch_combined.append(combined_rec)

            if batch_hn:
                bulk_insert(
                    "raw_hackernews_data",
                    batch_hn,
                    "hn_id"
                )
                insert_combined_records(batch_combined)
                all_hn.extend(batch_hn)

            duration = time_lib.time() - start_time
            log_collection(
                "hackernews", "tertiary",
                f"top_{q}", len(batch_hn),
                duration=duration
            )

            print(f"  ✓ Top '{q[:30]}': "
                  f"{len(batch_hn)} items "
                  f"({duration:.1f}s)")

            time_lib.sleep(DELAY_BETWEEN_QUERIES)

        except Exception as e:
            print(f"  ✗ Top error '{q}': {e}")
            continue

    # final summary
    total = len(all_hn)

    print("\n" + "=" * 55)
    print(f"  [hackernews] Collection Complete")
    print(f"  Total records : {total:,}")
    print(f"  BASE_URL used : Firebase feed collection")
    print(f"  SEARCH_URL used: Algolia keyword + recent")
    print("=" * 55)

    if total == 0:
        print("\n  [hackernews] WARNING: No records collected.")
        print("  Run: python debug_hackernews.py to diagnose")
        return pd.DataFrame()

    return pd.DataFrame(all_hn)