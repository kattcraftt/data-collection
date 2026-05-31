from atproto import Client
import pandas as pd
from datetime import datetime
import time as time_lib
from tqdm import tqdm
from config import BLUESKY_CONFIG, BLUESKY_QUERIES, BLUESKY_LIMIT_PER_QUERY
from database import bulk_insert, insert_combined_records, log_collection

def collect_bluesky() -> pd.DataFrame:
    print("\n" + "=" * 55)
    print(" [bluesky] SECONDARY SOURCE - Starting Collection")
    print("=" * 55)

    client = Client()

    try:
        client.login(
            BLUESKY_CONFIG["handle"],
            BLUESKY_CONFIG["password"],
        )
        print(" ✓ Bluesky login successful")
    except Exception as e:
        print(f" ✗ Bluesky login failed: {e}")
        return pd.DataFrame()

    all_bluesky = []
    all_combined = []

    for query in tqdm(BLUESKY_QUERIES,
                      desc="Bluesky Queries"):
        start_time = time_lib.time()

        try:
            response = client.app.bsky.feed.search_posts(
                params={
                    "q": query,
                    "limit": BLUESKY_LIMIT_PER_QUERY,
                    "lang": "en"
                }
            )

            batch_bluesky = []
            batch_combined = []

            for post in response.posts:
                text = post.record.text.strip()

                if len(text) < 15:
                    continue

                created = post.record.created_at
                if isinstance(created, str):
                    try:
                        created = datetime.fromisoformat(
                            created.replace("Z", "+00:00")
                        )
                    except Exception:
                        created = datetime.utcnow()

                bsky_rec = {
                    "post_cid": post.cid,
                    "post_uri": post.uri,
                    "text": text,
                    "author_handle": post.author.handle,
                    "like_count": post.like_count or 0,
                    "repost_count": post.repost_count or 0,
                    "reply_count": post.reply_count or 0,
                    "created_at": created,
                    "query_used": query
                }

                combined_rec = {
                    "original_id": f"bluesky_{post.cid}",
                    "source": "bluesky",
                    "platform_weight": "secondary",
                    "text": text,
                    "score": post.like_count or 0,
                    "engagement": (
                            (post.reply_count or 0) +
                            (post.repost_count or 0)
                    ),
                    "created_date": created,
                    "query_used": query,
                    "category": None,
                    "text_length": len(text),
                    "word_count": len(text.split())
                }

                batch_bluesky.append(bsky_rec)
                batch_combined.append(combined_rec)

            bulk_insert(
                "raw_bluesky_data",
                batch_bluesky,
                "post_cid"
            )
            insert_combined_records(batch_combined)

            all_bluesky.extend(batch_bluesky)
            all_combined.extend(batch_combined)

            duration = time_lib.time() - start_time
            log_collection(
                "bluesky", "secondary", query,
                len(batch_bluesky), duration=duration
            )

            print(f"  ✓ '{query[:40]}': "
                  f"{len(batch_bluesky)} posts")

            time_libS.sleep(0.5)

        except Exception as e:
            print(f"\n  ✗ Bluesky error on '{query}': {e}")
            log_collection(
                "bluesky", "secondary", query,
                0, status="error", error=str(e)
            )
            continue

    print(f"\n[bluesky] Collection complete.")
    print(f"  Total records: {len(all_bluesky)}")
    return pd.DataFrame(all_bluesky)