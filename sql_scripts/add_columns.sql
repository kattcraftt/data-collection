ALTER TABLE raw_combined_dataset
ADD COLUMN IF NOT EXISTS sentiment_confidence FLOAT;

ALTER TABLE raw_combined_dataset
ADD COLUMN IF NOT EXISTS topic_label VARCHAR(500);

ALTER TABLE raw_combined_dataset
ADD COLUMN IF NOT EXISTS rf_prediction VARCHAR(50);

ALTER TABLE raw_combined_dataset
ADD COLUMN IF NOT EXISTS rf_confidence FLOAT;

ALTER TABLE raw_combined_dataset
ADD COLUMN IF NOT EXISTS rf_split VARCHAR(10);

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'raw_combined_dataset'
ORDER BY ordinal_position;