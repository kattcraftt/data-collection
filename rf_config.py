# Random Forest Model Configuration

RF_CONFIG = {

    # model hyperparameters
    "n_estimators"     : 200,   # number of trees
    "max_depth"        : None,  # unlimited depth
    "min_samples_split": 5,     # min samples to split node
    "min_samples_leaf" : 2,     # min samples at leaf
    "max_features"     : "sqrt",# features per split
    "class_weight"     : "balanced", # handle imbalanced data
    "random_state"     : 42,
    "n_jobs"           : -1,    # use all CPU cores
    "verbose"          : 0,

    # training settings
    "test_size"        : 0.2,   # 80/20 train/test split
    "cv_folds"         : 5,     # cross-validation folds
    "random_state_split": 42,

    # feature columns (from NLP extraction)
    "feature_columns"  : [
        # RoBERTa features
        "sentiment_negative",
        "sentiment_neutral",
        "sentiment_positive",

        # BERTopic features
        "topic_id",
        "topic_probability",

        # Engagement features
        "score",
        "engagement",
        "keyword_count"
    ],

    # target column
    "target_column"    : "predicted_behavior",

    # target classes
    "target_classes"   : [
        "purchase_intent",
        "passive_browsing",
        "churn_risk"
    ]
}

# output settings
RF_OUTPUT_DIR = "output_csv/random_forest"