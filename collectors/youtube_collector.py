from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
from datetime import datetime
import time as time_lib
import random
from tqdm import tqdm
from config import YOUTUBE_CONFIG, YOUTUBE_SEARCH_QUERIES, YOUTUBE_MAX_VIDEOS_PER_QUERY, YOUTUBE_MAX_COMMENTS_PER_VIDEO
from database import bulk_insert, insert_combined_records, log_collection

# rate limit settings
DELAY_BETWEEN_QUERIES  = 5.0   # seconds between search calls
DELAY_BETWEEN_VIDEOS   = 1.0   # seconds between video calls
DELAY_BETWEEN_COMMENTS = 0.5   # seconds between comment calls
MAX_RETRIES            = 3     # retry attempts on 429
RETRY_BASE_DELAY       = 30.0  # base wait on 429

def api_call_with_retry(api_call_fn,
                        max_retries: int = MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            return api_call_fn()

        except HttpError as e:
            if e.resp.status == 429:
                wait   = RETRY_BASE_DELAY * (2 ** attempt)
                jitter = random.uniform(1, 5)
                total  = wait + jitter

                print(f"\n  [youTube] 429 Rate limit. "
                      f"Waiting {total:.0f}s "
                      f"(retry {attempt + 1}/{max_retries})...")
                time_lib.sleep(total)

                if attempt == max_retries - 1:
                    raise e
                continue

            elif e.resp.status == 403:
                raise e
            else:
                raise e

        except Exception as e:
            raise e

    return None

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

# fetch video statistics
def fetch_video_stats(youtube_client,
                      video_id: str) -> dict:
    default = {
        "category"        : "unknown",
        "view_count"      : 0,
        "video_like_count": 0
    }
    try:
        def _call():
            return youtube_client.videos().list(
                part = "statistics,snippet",
                id   = video_id
            ).execute()

        resp = api_call_with_retry(_call)

        if resp and resp.get("items"):
            item  = resp["items"][0]
            stats = item.get("statistics", {})
            return {
                "category"        : item["snippet"].get(
                                      "categoryId", "unknown"),
                "view_count"      : safe_int(
                                      stats.get("viewCount")),
                "video_like_count": safe_int(
                                      stats.get("likeCount"))
            }
    except Exception:
        pass
    return default

# fetch comments for a video
def fetch_video_comments(youtube_client,
                         video_id:    str,
                         video_title: str,
                         channel:     str,
                         video_stats: dict,
                         query:       str,
                         max_results: int = 100) -> tuple:
    batch_yt       = []
    batch_combined = []

    try:
        def _call():
            return youtube_client.commentThreads().list(
                part       = "snippet",
                videoId    = video_id,
                maxResults = max_results,
                order      = "relevance",
                textFormat = "plainText"
            ).execute()

        resp  = api_call_with_retry(_call)
        items = resp.get("items", []) if resp else []

        for item in items:
            try:
                snippet = (item["snippet"]
                               ["topLevelComment"]
                               ["snippet"])
                text    = str(
                    snippet.get("textDisplay", "") or ""
                ).strip()

                if len(text) < 15:
                    continue

                published   = safe_datetime(
                    snippet.get("publishedAt")
                )
                like_count  = safe_int(
                    snippet.get("likeCount")
                )
                reply_count = safe_int(
                    item["snippet"].get("totalReplyCount")
                )
                comment_id  = str(item.get("id", ""))

                if not comment_id:
                    continue

                yt_rec = {
                    "video_id"        : video_id,
                    "comment_id"      : comment_id,
                    "video_title"     : video_title,
                    "channel_name"    : channel,
                    "comment_text"    : text,
                    "like_count"      : like_count,
                    "reply_count"     : reply_count,
                    "published_at"    : published,
                    "query_used"      : query,
                    "video_category"  : video_stats[
                                          "category"],
                    "view_count"      : video_stats[
                                          "view_count"],
                    "video_like_count": video_stats[
                                          "video_like_count"]
                }

                combined_rec = {
                    "original_id"    : f"yt_{comment_id}",
                    "source"         : "youtube",
                    "platform_weight": "primary",
                    "text"           : text,
                    "score"          : like_count,
                    "engagement"     : reply_count,
                    "created_date"   : published,
                    "query_used"     : query,
                    "category"       : video_title[:200],
                    "text_length"    : len(text),
                    "word_count"     : len(text.split())
                }

                batch_yt.append(yt_rec)
                batch_combined.append(combined_rec)

            except Exception:
                continue

        time_lib.sleep(DELAY_BETWEEN_COMMENTS)

    except HttpError as e:
        if e.resp.status not in [403, 404]:
            print(f"\n    [YT Comments] "
                  f"HTTP {e.resp.status} on {video_id}")
    except Exception as e:
        print(f"\n    [YT Comments] Error: {e}")

    return batch_yt, batch_combined

def collect_youtube() -> pd.DataFrame:
    print("\n" + "=" * 55)
    print(" [youTube] PRIMARY SOURCE — Starting Collection")
    print("=" * 55)
    print(f"\n  Rate limit settings:")
    print(f"  Query delay    : {DELAY_BETWEEN_QUERIES}s")
    print(f"  Video delay    : {DELAY_BETWEEN_VIDEOS}s")
    print(f"  Comment delay  : {DELAY_BETWEEN_COMMENTS}s")
    print(f"  429 retries    : {MAX_RETRIES}x "
          f"(base {RETRY_BASE_DELAY}s backoff)")

    api_key = YOUTUBE_CONFIG.get("api_key")
    if not api_key:
        print("\n  [youTube] ERROR: YOUTUBE_API_KEY not in .env")
        return pd.DataFrame()

    try:
        youtube = build(
            "youtube", "v3",
            developerKey=api_key
        )
    except Exception as e:
        print(f"\n  [youTube] Build failed: {e}")
        return pd.DataFrame()

    all_youtube  = []
    all_combined = []
    quota_used   = 0
    skipped      = 0

    for idx, query in enumerate(
        tqdm(YOUTUBE_SEARCH_QUERIES,
             desc="Youtube Queries"), 1
    ):
        query_start = time_lib.time()

        try:
            print(f"\n  [{idx}/{len(YOUTUBE_SEARCH_QUERIES)}]"
                  f" Searching: '{query[:40]}'")

            # search with retry
            def _search():
                return youtube.search().list(
                    q = query,
                    part = "id,snippet",
                    maxResults = YOUTUBE_MAX_VIDEOS_PER_QUERY,
                    type = "video",
                    relevanceLanguage = "en",
                    order = "relevance"
                ).execute()

            search_response = api_call_with_retry(_search)
            quota_used     += 100

            videos = (search_response.get("items", [])
                      if search_response else [])

            if not videos:
                print(f"    No videos found")
                log_collection(
                    "youtube", "primary", query, 0,
                    duration=time_lib.time() - query_start
                )
                time_lib.sleep(DELAY_BETWEEN_QUERIES)
                continue

            print(f"    Found {len(videos)} videos")

            query_batch_yt       = []
            query_batch_combined = []

            for v_idx, video in enumerate(videos, 1):
                try:
                    video_id    = video["id"]["videoId"]
                    video_title = str(
                        video["snippet"].get("title", "")
                    )
                    channel = str(
                        video["snippet"].get(
                            "channelTitle", "")
                    )

                    stats = fetch_video_stats(
                        youtube, video_id
                    )
                    quota_used += 1

                    v_yt, v_combined = fetch_video_comments(
                        youtube_client = youtube,
                        video_id       = video_id,
                        video_title    = video_title,
                        channel        = channel,
                        video_stats    = stats,
                        query          = query,
                        max_results    = YOUTUBE_MAX_COMMENTS_PER_VIDEO
                    )
                    quota_used += 1

                    query_batch_yt.extend(v_yt)
                    query_batch_combined.extend(v_combined)

                    print(f"    Video {v_idx}/{len(videos)}: "
                          f"{len(v_yt)} comments — "
                          f"'{video_title[:30]}'")

                    time_lib.sleep(DELAY_BETWEEN_VIDEOS)

                except HttpError as e:
                    if e.resp.status == 429:
                        print(f"\n    429 on video. "
                              f"Waiting 30s...")
                        time_lib.sleep(30)
                    continue

                except Exception as e:
                    print(f"\n    Video error: {e}")
                    continue

            # save batch
            if query_batch_yt:
                try:
                    bulk_insert(
                        "raw_youtube_data",
                        query_batch_yt,
                        "comment_id"
                    )
                    insert_combined_records(
                        query_batch_combined
                    )
                    all_youtube.extend(query_batch_yt)
                    all_combined.extend(
                        query_batch_combined
                    )
                except Exception as e:
                    print(f"\n    DB error: {e}")

            # log
            duration = time_lib.time() - query_start
            log_collection(
                "youtube", "primary", query,
                len(query_batch_yt),
                duration=duration
            )

            print(f"\n  ✓ '{query[:35]}': "
                  f"{len(query_batch_yt)} comments "
                  f"| ~{quota_used} quota "
                  f"| {duration:.1f}s")

            if quota_used >= 9000:
                print("\n  [youTube] Near quota limit. Stopping.")
                break

            print(f"  Waiting {DELAY_BETWEEN_QUERIES}s "
                  f"before next query...")
            time_lib.sleep(DELAY_BETWEEN_QUERIES)

        except HttpError as e:
            duration = time_lib.time() - query_start

            if e.resp.status == 429:
                print(f"\n  [youTube] 429 on query. "
                      f"Waiting 60s...")
                time_lib.sleep(60)
                skipped += 1

            elif e.resp.status == 403:
                print(f"\n  [youTube] Daily quota exhausted.")
                log_collection(
                    "youtube", "primary", query, 0,
                    status="quota_exhausted",
                    error=str(e), duration=duration
                )
                break

            else:
                print(f"\n  ✗ HTTP {e.resp.status}: {e}")
                log_collection(
                    "youtube", "primary", query, 0,
                    status="error", error=str(e),
                    duration=duration
                )
            continue

        except Exception as e:
            duration = time_lib.time() - query_start
            print(f"\n  ✗ Error '{query}': {e}")
            log_collection(
                "youtube", "primary", query, 0,
                status="error", error=str(e),
                duration=duration
            )
            continue

    # summary
    total = len(all_youtube)

    print("\n" + "=" * 55)
    print(f"  [youTube] Collection Complete")
    print(f"  Comments : {total:,}")
    print(f"  Skipped  : {skipped}")
    print(f"  Quota    : ~{quota_used:,} / 10,000 units")
    print(f"  Remaining: ~{10000 - quota_used:,} units")
    print("=" * 55)

    if total == 0:
        print("\n  No records collected.")
        return pd.DataFrame()

    return pd.DataFrame(all_youtube)