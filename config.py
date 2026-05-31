import os
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_CONFIG = {
    "api_key": os.getenv("YOUTUBE_API_KEY")
}

YOUTUBE_SEARCH_QUERIES = [
    "digital market trends 2021",
    "digital market trends 2022",
    "digital market trends 2023",
    "digital market trends 2024",
    "digital market trends 2025",
    "digital market trends 2026",
    "online shopping experience review",
    "ecommerce product review",
    "digital marketplace customer experience",
    "best online store review",
    "product unboxing review",
    "customer service experience review",
    "product quality review online store",
    "online shopping complaint",
    "best deals online shopping",
    "digital product review",
    "consumer behavior online market",
    "amazon shopping experience",
    "ebay buying experience review",
    "etsy seller review customer experience",
    "online purchase review feedback",
    "digital market trends consumer",
    "ecommerce platform comparison",
    "consumer behavior online shopping",
    "digital market trends consumer",
    "online purchase review feedback",
    "why i stopped online shopping",
    "online shopping tips consumer advice"
]

YOUTUBE_MAX_VIDEOS_PER_QUERY = 10
YOUTUBE_MAX_COMMENTS_PER_VIDEO = 100
YOUTUBE_DAILY_QUOTA = 10000

BLUESKY_CONFIG = {
    "handle": os.getenv("BLUESKY_HANDLE"),
    "password": os.getenv("BLUESKY_PASSWORD")
}

BLUESKY_QUERIES = [
    "online shopping experience",
    "product review purchase",
    "customer service complaint",
    "ecommerce digital market",
    "brand recommendation buy",
    "digital marketplace review",
    "consumer behavior shopping",
    "online store feedback",
    "digital market consumer trend",
    "consumer product review",
    "best online store",
    "worst online shopping"
]

BLUESKY_LIMIT_PER_QUERY = 100

HACKERNEWS_CONFIG = {
    "base_url": "https://hacker-news.firebaseio.com/v0",
    "search_url": "https://hn.algolia.com/api/v1"
}

HACKERNEWS_QUERIES = [
    "ecommerce consumer behavior",
    "digital market trends",
    "online shopping experience",
    "product recommendation system",
    "customer analytics",
    "retail technology consumer",
    "marketplace platform review",
    "digital commerce trends",
    "customer analytics behavior",
    "product recommendation system",
    "consumer behavior prediction",
    "retail analytics machine learning",
    "ecommerce data analytics",
    "consumer insight digital market"
]

HACKERNEWS_FEEDS = ["topstories", "newstories",
                    "askstories", "showstories"]
HACKERNEWS_LIMIT = 300

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "customer_behavior_db"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

DATABASE_URL = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)

CSV_OUTPUT_DIR = "output_csv"