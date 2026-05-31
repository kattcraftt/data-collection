TRUNCATE TABLE raw_combined_dataset RESTART IDENTITY CASCADE;
TRUNCATE TABLE raw_youtube_data RESTART IDENTITY CASCADE;
TRUNCATE TABLE raw_bluesky_data RESTART IDENTITY CASCADE;
TRUNCATE TABLE raw_hackernews_data RESTART IDENTITY CASCADE;
TRUNCATE TABLE collection_log RESTART IDENTITY CASCADE;

SELECT 'raw_combined_dataset' AS table_name, COUNT(*) AS rows FROM raw_combined_dataset
UNION ALL
SELECT 'raw_youtube_data',    COUNT(*) FROM raw_youtube_data
UNION ALL
SELECT 'raw_bluesky_data',    COUNT(*) FROM raw_bluesky_data
UNION ALL
SELECT 'raw_hackernews_data', COUNT(*) FROM raw_hackernews_data
UNION ALL
SELECT 'collection_log',      COUNT(*) FROM collection_log
ORDER BY table_name;