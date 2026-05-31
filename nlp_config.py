# NLP Pipeline Configuration
# RoBERTa + BERTopic settings

# RoBERTa settings
ROBERTA_CONFIG = {
    # Twitter-RoBERTa trained on 124M tweets
    # best for social media consumer text
    "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "max_length": 512,
    "batch_size": 32,
    "device": "auto",   # auto-detect GPU/CPU

    # label mapping from model output
    "label_map": {
        0: "negative",
        1: "neutral",
        2: "positive"
    }
}

# BERTopic settings
BERTOPIC_CONFIG = {
    # sentence transformer for embeddings
    "embedding_model": "all-MiniLM-L6-v2",

    # UMAP dimensionality reduction
    "umap_config": {
        "n_neighbors": 15,
        "n_components": 5,
        "min_dist": 0.0,
        "metric": "cosine",
        "random_state": 42
    },

    # HDBSCAN clustering
    "hdbscan_config": {
        "min_cluster_size": 10,
        "metric": "euclidean",
        "cluster_selection_method": "eom",
        "prediction_data": True
    },

    # BERTopic general settings
    "nr_topics": "auto",  # auto-detect number of topics
    "top_n_words": 10,      # keywords per topic
    "min_topic_size": 10     # minimum docs per topic
}

# processing settings
NLP_CONFIG = {
    "batch_size": 32,    # texts per RoBERTa batch
    "min_text_length": 15,    # skip very short texts
    "max_text_length": 512,   # truncate long texts
    "db_update_batch": 500    # records per DB update
}

# output settings
NLP_OUTPUT_DIR = "output_csv/nlp"