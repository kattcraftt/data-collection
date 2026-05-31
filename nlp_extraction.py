# NLP Feature Extraction Pipeline
# Reads   : prisma_included records from PostgreSQL
# Stage 1 : RoBERTa — sentiment + behavioral intent scoring
# Stage 2 : BERTopic — discourse and topic extraction
# Writes  : NLP features back to PostgreSQL
# Exports : Feature dataset CSV for Random Forest

import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

warnings.filterwarnings("ignore")

# PyTorch and Transformers
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Sentence Transformers
from sentence_transformers import SentenceTransformer

# BERTopic
from bertopic import BERTopic
from umap     import UMAP
from hdbscan  import HDBSCAN

# database and config
from database   import get_engine, get_connection
from nlp_config import ROBERTA_CONFIG, BERTOPIC_CONFIG, NLP_CONFIG, NLP_OUTPUT_DIR
from config     import CSV_OUTPUT_DIR

# database helpers
def load_prisma_included() -> pd.DataFrame:
    """Load all PRISMA-included records for NLP processing."""
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
            keyword_count
        FROM raw_combined_dataset
        WHERE prisma_included = TRUE
        ORDER BY id ASC;
    """, engine)
    return df


def save_nlp_results(df: pd.DataFrame,
                     batch_size: int = 500) -> int:
    """Write NLP features back to PostgreSQL."""
    nlp_cols = [
        "id",
        "sentiment_negative",
        "sentiment_neutral",
        "sentiment_positive",
        "sentiment_label",
        "sentiment_confidence",
        "topic_id",
        "topic_probability",
        "topic_label",
        "predicted_behavior"
    ]

    available = [c for c in nlp_cols if c in df.columns]
    records   = df[available].copy()
    records   = records.where(pd.notnull(records), None)

    records_list = records.to_dict("records")
    total        = len(records_list)
    updated      = 0

    print(f"\n  Saving {total:,} NLP records to PostgreSQL...")

    conn = get_connection()
    cur  = conn.cursor()

    for i in tqdm(range(0, total, batch_size),
                  desc="  DB Update"):
        batch = records_list[i:i + batch_size]

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

# device setup
def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  ✓ GPU detected: "
              f"{torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"  ✓ Apple Silicon MPS detected")
    else:
        device = torch.device("cpu")
        print(f"  ✓ Using CPU")
    return device

# text preprocessing
def preprocess_text(text: str,
                    max_length: int = 512) -> str:
    """
    Clean and truncate text for NLP processing.
    Keeps semantic meaning while removing noise.
    """
    if not isinstance(text, str):
        return ""

    import re

    # remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # remove excessive punctuation
    text = re.sub(r"[!?]{3,}", "!", text)
    text = re.sub(r"\.{3,}", "...", text)

    # truncate to max length (by words to preserve meaning)
    words = text.split()
    if len(words) > max_length:
        text = " ".join(words[:max_length])

    return text

# Stage 1: RoBERTa sentiment extraction
class RoBERTaExtractor:

    def __init__(self, config: dict, device: torch.device):
        self.config    = config
        self.device    = device
        self.model     = None
        self.tokenizer = None

    def load(self):
        """Load RoBERTa model and tokenizer."""
        print(f"\n  Loading RoBERTa model...")
        print(f"  Model: {self.config['model_name']}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"]
        )
        self.model = AutoModelForSequenceClassification\
            .from_pretrained(
                self.config["model_name"]
            )
        self.model.to(self.device)
        self.model.eval()

        print(f"  ✓ RoBERTa loaded on {self.device}")
        return self

    def predict_batch(self,
                      texts: list) -> list:
        """
        Run sentiment prediction on a batch of texts.
        Returns list of dicts with scores per class.
        """
        results = []

        try:
            # tokenize
            encoded = self.tokenizer(
                texts,
                return_tensors = "pt",
                truncation     = True,
                padding        = True,
                max_length     = self.config["max_length"]
            )

            # move to device
            encoded = {
                k: v.to(self.device)
                for k, v in encoded.items()
            }

            # inference
            with torch.no_grad():
                outputs = self.model(**encoded)
                probs   = torch.nn.functional.softmax(
                    outputs.logits, dim=-1
                ).cpu().numpy()

            # build results
            label_map = self.config["label_map"]

            for prob in probs:
                label_idx  = int(np.argmax(prob))
                confidence = float(np.max(prob))

                results.append({
                    "sentiment_negative"  : float(prob[0]),
                    "sentiment_neutral"   : float(prob[1]),
                    "sentiment_positive"  : float(prob[2]),
                    "sentiment_label"     : label_map[label_idx],
                    "sentiment_confidence": confidence
                })

        except Exception as e:
            print(f"\n  [RoBERTa] Batch error: {e}")
            # return neutral defaults for failed batch
            for _ in texts:
                results.append({
                    "sentiment_negative"  : 0.333,
                    "sentiment_neutral"   : 0.334,
                    "sentiment_positive"  : 0.333,
                    "sentiment_label"     : "neutral",
                    "sentiment_confidence": 0.334
                })

        return results

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run RoBERTa on all texts in dataframe.
        Processes in batches for efficiency.
        """
        print("\n" + "=" * 55)
        print("  Stage 1 — RoBERTa SENTIMENT EXTRACTION")
        print("=" * 55)
        print(f"\n  Records to process : {len(df):,}")
        print(f"  Batch size         : "
              f"{self.config['batch_size']}")
        print(f"  Device             : {self.device}")

        texts      = df["text_clean"].tolist()
        batch_size = self.config["batch_size"]
        all_results= []

        for i in tqdm(range(0, len(texts), batch_size),
                      desc="  RoBERTa Batches"):
            batch   = texts[i:i + batch_size]
            results = self.predict_batch(batch)
            all_results.extend(results)

        # add results to dataframe
        results_df = pd.DataFrame(all_results)

        df["sentiment_negative"]  = results_df[
            "sentiment_negative"].values
        df["sentiment_neutral"]   = results_df[
            "sentiment_neutral"].values
        df["sentiment_positive"]  = results_df[
            "sentiment_positive"].values
        df["sentiment_label"]     = results_df[
            "sentiment_label"].values
        df["sentiment_confidence"]= results_df[
            "sentiment_confidence"].values

        # print distribution
        label_counts = df["sentiment_label"].value_counts()
        total        = len(df)

        print(f"\n  Sentiment distribution:")
        print(f"  {'Label':<12} {'Count':>8} {'Percent':>9}")
        print(f"  {'-'*12} {'-'*8} {'-'*9}")

        for label, count in label_counts.items():
            pct = count / total * 100
            print(f"  {label:<12} {count:>8,} {pct:>8.1f}%")

        print(f"\n  ✓ RoBERTa extraction complete")
        return df

# Stage 2: BERTopic discourse extraction
class BERTopicExtractor:

    def __init__(self, config: dict):
        self.config      = config
        self.topic_model = None
        self.topic_info  = None

    def build_model(self):
        """Build BERTopic model with configured components."""
        print(f"\n  Building BERTopic model...")
        print(f"  Embedding model: "
              f"{self.config['embedding_model']}")

        # sentence transformer for embeddings
        embedding_model = SentenceTransformer(
            self.config["embedding_model"]
        )

        # UMAP for dimensionality reduction
        umap_model = UMAP(
            **self.config["umap_config"]
        )

        # HDBSCAN for clustering
        hdbscan_model = HDBSCAN(
            **self.config["hdbscan_config"]
        )

        # build BERTopic
        self.topic_model = BERTopic(
            embedding_model = embedding_model,
            umap_model      = umap_model,
            hdbscan_model   = hdbscan_model,
            nr_topics       = self.config["nr_topics"],
            top_n_words     = self.config["top_n_words"],
            min_topic_size  = self.config["min_topic_size"],
            verbose         = False
        )

        print(f"  ✓ BERTopic model built")
        return self

    def extract(self,
                df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit BERTopic on texts and extract topic features.
        """
        print("\n" + "=" * 55)
        print("  Stage 2 — BERTopic DISCOURSE EXTRACTION")
        print("=" * 55)
        print(f"\n  Records to process : {len(df):,}")
        print(f"  Min topic size     : "
              f"{self.config['min_topic_size']}")
        print(f"  Nr topics          : "
              f"{self.config['nr_topics']}")

        texts = df["text_clean"].tolist()

        print(f"\n  Fitting BERTopic model...")
        print(f"  This may take several minutes...")

        try:
            topics, probs = self.topic_model.fit_transform(
                texts
            )

            # get topic info
            self.topic_info = self.topic_model.get_topic_info()

            n_topics   = len(self.topic_info) - 1
            n_outliers = sum(1 for t in topics if t == -1)

            print(f"\n  ✓ BERTopic fitting complete")
            print(f"  Topics discovered  : {n_topics}")
            print(f"  Outlier records    : {n_outliers:,}")

            # add topic assignments to df
            df["topic_id"] = topics

            # handle probability arrays
            if probs is not None and len(probs) > 0:
                if hasattr(probs[0], "__len__"):
                    # Array of probability arrays
                    df["topic_probability"] = [
                        float(max(p)) if len(p) > 0 else 0.0
                        for p in probs
                    ]
                else:
                    # single probability values
                    df["topic_probability"] = [
                        float(p) for p in probs
                    ]
            else:
                df["topic_probability"] = 0.0

            # add topic labels
            topic_label_map = {}
            for _, row in self.topic_info.iterrows():
                topic_id    = row["Topic"]
                topic_name  = str(row.get("Name", ""))
                topic_label_map[topic_id] = topic_name

            df["topic_label"] = df["topic_id"].map(
                topic_label_map
            ).fillna("outlier")

            # print top topics
            print(f"\n  Top 10 discovered topics:")
            print(f"  {'ID':>4} {'Size':>6} "
                  f"{'Topic Name':<40}")
            print(f"  {'-'*4} {'-'*6} {'-'*40}")

            top_topics = self.topic_info[
                self.topic_info["Topic"] != -1
            ].head(10)

            for _, row in top_topics.iterrows():
                print(f"  {row['Topic']:>4} "
                      f"{row['Count']:>6,} "
                      f"{str(row['Name'])[:40]:<40}")

        except Exception as e:
            print(f"\n  [BERTopic] Error: {e}")
            print(f"  Assigning default topic values...")
            df["topic_id"]          = -1
            df["topic_probability"] = 0.0
            df["topic_label"]       = "unknown"

        print(f"\n  ✓ BERTopic extraction complete")
        return df

    def save_topic_model(self,
                         path: str):
        """Save fitted BERTopic model for reuse."""
        try:
            self.topic_model.save(path)
            print(f"  ✓ BERTopic model saved: {path}")
        except Exception as e:
            print(f"  ✗ Model save error: {e}")

    def get_topic_keywords(self) -> pd.DataFrame:
        """Get all topic keywords for documentation."""
        if self.topic_model is None:
            return pd.DataFrame()

        rows = []
        for topic_id in self.topic_info["Topic"].unique():
            if topic_id == -1:
                continue
            try:
                keywords = self.topic_model.get_topic(
                    topic_id
                )
                if keywords:
                    kw_str = ", ".join([
                        w for w, _ in keywords[:10]
                    ])
                    rows.append({
                        "topic_id"  : topic_id,
                        "keywords"  : kw_str,
                        "top_word"  : keywords[0][0]
                                      if keywords else "",
                        "doc_count" : int(
                            self.topic_info[
                                self.topic_info["Topic"]
                                == topic_id
                            ]["Count"].values[0]
                        )
                    })
            except Exception:
                continue

        return pd.DataFrame(rows)

# Stage 3: BEHAVIORAL label assignment
# map NLP features to behavioral prediction classes
def assign_behavioral_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine RoBERTa sentiment and BERTopic discourse
    to assign a preliminary behavioral prediction label.

    Labels:
      purchase_intent  → positive sentiment + buying keywords
      churn_risk       → negative sentiment + complaint topics
      passive_browsing → neutral sentiment + browsing behavior
    """
    print("\n" + "=" * 55)
    print("  Stage 3 — BEHAVIORAL LABEL ASSIGNMENT")
    print("=" * 55)

    # purchase intent keywords
    intent_keywords = [
        "buy", "purchase", "order", "checkout",
        "recommend", "worth it", "would buy",
        "best purchase", "good deal", "love it",
        "excellent", "amazing", "satisfied",
        "repeat", "loyal", "will buy again"
    ]

    # churn risk keywords
    churn_keywords = [
        "refund", "return", "complaint", "scam",
        "fraud", "disappointed", "terrible",
        "never again", "waste", "regret",
        "worst", "horrible", "broken", "fake",
        "not as described", "chargeback"
    ]

    def get_behavioral_label(row) -> str:
        sentiment = str(row.get("sentiment_label", ""))
        text      = str(row.get("text", "")).lower()
        conf      = float(row.get(
            "sentiment_confidence", 0
        ))

        # check keyword signals
        has_intent = any(
            kw in text for kw in intent_keywords
        )
        has_churn  = any(
            kw in text for kw in churn_keywords
        )

        # decision logic
        if sentiment == "positive" and has_intent:
            return "purchase_intent"

        elif sentiment == "negative" and has_churn:
            return "churn_risk"

        elif sentiment == "positive" and conf >= 0.7:
            return "purchase_intent"

        elif sentiment == "negative" and conf >= 0.7:
            return "churn_risk"

        else:
            return "passive_browsing"

    tqdm.pandas(desc="  Assigning labels")
    df["predicted_behavior"] = df.progress_apply(
        get_behavioral_label, axis=1
    )

    # distribution
    behavior_counts = df["predicted_behavior"].value_counts()
    total           = len(df)

    print(f"\n  Behavioral label distribution:")
    print(f"  {'Label':<20} {'Count':>8} {'Percent':>9}")
    print(f"  {'-'*20} {'-'*8} {'-'*9}")

    for label, count in behavior_counts.items():
        pct = count / total * 100
        print(f"  {label:<20} {count:>8,} {pct:>8.1f}%")

    print(f"\n  ✓ Behavioral labels assigned")
    return df

# exportt results
def export_nlp_results(df: pd.DataFrame,
                       topic_keywords: pd.DataFrame,
                       timestamp: str):
    print("\n" + "=" * 55)
    print("  EXPORTING NLP RESULTS")
    print("=" * 55)

    os.makedirs(NLP_OUTPUT_DIR, exist_ok=True)

    # file 1: full NLP feature dataset
    feature_cols = [
        "id", "original_id", "source",
        "platform_weight", "text",
        "score", "engagement", "created_date",
        "keyword_count",

        # RoBERTa features
        "sentiment_negative",
        "sentiment_neutral",
        "sentiment_positive",
        "sentiment_label",
        "sentiment_confidence",

        # BERTopic features
        "topic_id",
        "topic_probability",
        "topic_label",

        # Behavioral label
        "predicted_behavior"
    ]

    available = [
        c for c in feature_cols
        if c in df.columns
    ]
    features_df = df[available].copy()

    path_features = (
        f"{NLP_OUTPUT_DIR}/"
        f"nlp_features_{timestamp}.csv"
    )
    features_df.to_csv(path_features, index=False)
    print(f"\n  ✓ NLP features dataset")
    print(f"    Records  : {len(features_df):,}")
    print(f"    Columns  : {len(features_df.columns)}")
    print(f"    File     : {path_features}")

    # file 2: topic keywords
    if not topic_keywords.empty:
        path_topics = (
            f"{NLP_OUTPUT_DIR}/"
            f"bertopic_topics_{timestamp}.csv"
        )
        topic_keywords.to_csv(path_topics, index=False)
        print(f"\n  ✓ BERTopic keywords")
        print(f"    Topics   : {len(topic_keywords):,}")
        print(f"    File     : {path_topics}")

    # file 3: random Forest ready dataset
    rf_cols = [
        "id",
        "sentiment_negative",
        "sentiment_neutral",
        "sentiment_positive",
        "topic_id",
        "topic_probability",
        "predicted_behavior",
        "source",
        "score",
        "engagement",
        "keyword_count"
    ]

    rf_available = [
        c for c in rf_cols
        if c in df.columns
    ]
    rf_df = df[rf_available].dropna(
        subset=["sentiment_negative",
                "topic_id"]
    ).copy()

    path_rf = (
        f"{NLP_OUTPUT_DIR}/"
        f"random_forest_input_{timestamp}.csv"
    )
    rf_df.to_csv(path_rf, index=False)
    print(f"\n  ✓ Random Forest input dataset")
    print(f"    Records  : {len(rf_df):,}")
    print(f"    Features : {len(rf_df.columns) - 2}")
    print(f"    File     : {path_rf}")

    # file 4: summary statistics
    summary = build_nlp_summary(df)
    path_summary = (
        f"{NLP_OUTPUT_DIR}/"
        f"nlp_summary_{timestamp}.csv"
    )
    summary.to_csv(path_summary, index=False)
    print(f"\n  ✓ NLP summary statistics")
    print(f"    File     : {path_summary}")

    return path_rf


def build_nlp_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build summary statistics for paper documentation."""
    rows = []

    # overall sentiment
    for label in ["negative", "neutral", "positive"]:
        col   = f"sentiment_{label}"
        count = int(
            (df["sentiment_label"] == label).sum()
        ) if "sentiment_label" in df.columns else 0
        rows.append({
            "metric"   : f"sentiment_{label}_count",
            "value"    : count,
            "source"   : "roberta"
        })
        if col in df.columns:
            rows.append({
                "metric": f"sentiment_{label}_mean_score",
                "value" : round(float(df[col].mean()), 4),
                "source": "roberta"
            })

    # behavioral labels
    if "predicted_behavior" in df.columns:
        for label in ["purchase_intent",
                      "churn_risk",
                      "passive_browsing"]:
            count = int(
                (df["predicted_behavior"] == label).sum()
            )
            pct   = count / len(df) * 100
            rows.append({
                "metric": f"behavior_{label}_count",
                "value" : count,
                "source": "behavioral"
            })
            rows.append({
                "metric": f"behavior_{label}_percent",
                "value" : round(pct, 2),
                "source": "behavioral"
            })

    # topic stats
    if "topic_id" in df.columns:
        n_topics   = int(
            df["topic_id"].nunique()
        )
        n_outliers = int(
            (df["topic_id"] == -1).sum()
        )
        rows.append({
            "metric": "bertopic_unique_topics",
            "value" : n_topics,
            "source": "bertopic"
        })
        rows.append({
            "metric": "bertopic_outlier_records",
            "value" : n_outliers,
            "source": "bertopic"
        })
        if "topic_probability" in df.columns:
            rows.append({
                "metric": "bertopic_mean_probability",
                "value" : round(float(
                    df["topic_probability"].mean()
                ), 4),
                "source": "bertopic"
            })

    return pd.DataFrame(rows)

# main
def main():
    print("\n" + "=" * 55)
    print("  NLP EXTRACTION PIPELINE")
    print("  RoBERTa + BERTopic")
    print("  Customer Behavior Analysis in Digital Markets")
    print("=" * 55)
    print(f"  Started  : "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(NLP_OUTPUT_DIR, exist_ok=True)

    # load PRISMA-included records
    print("\n[load] Loading PRISMA-included records...")
    df = load_prisma_included()

    if df is None or df.empty:
        print("\n  ✗ No PRISMA-included records found.")
        print("  Run python prisma_filter.py first.")
        return

    print(f"  ✓ Loaded {len(df):,} records")

    # preprocess text
    print("\n[preprocess] Cleaning text...")
    tqdm.pandas(desc="  Preprocessing")
    df["text_clean"] = df["text"].progress_apply(
        lambda x: preprocess_text(
            x,
            max_length=NLP_CONFIG["max_text_length"]
        )
    )

    # remove empty texts after preprocessing
    df = df[
        df["text_clean"].str.len() >=
        NLP_CONFIG["min_text_length"]
    ].reset_index(drop=True)

    print(f"  ✓ {len(df):,} records after preprocessing")

    # detect device
    print("\n[device] Detecting compute device...")
    device = get_device()

    # stage 1: RoBERTa
    print("\n[stage 1] Running RoBERTa sentiment extraction...")
    try:
        roberta = RoBERTaExtractor(
            config = ROBERTA_CONFIG,
            device = device
        ).load()
        df = roberta.extract(df)
    except Exception as e:
        print(f"\n  ✗ RoBERTa failed: {e}")
        print("  Check: pip install transformers torch")
        return

    # stage 2: BERTopic
    print("\n[stage 2] Running BERTopic discourse extraction...")
    try:
        bertopic_extractor = BERTopicExtractor(
            config = BERTOPIC_CONFIG
        ).build_model()
        df = bertopic_extractor.extract(df)

        # get topic keywords for export
        topic_keywords = bertopic_extractor.get_topic_keywords()

        # save fitted model for reuse
        model_path = f"{NLP_OUTPUT_DIR}/bertopic_model"
        bertopic_extractor.save_topic_model(model_path)

    except Exception as e:
        print(f"\n  ✗ BERTopic failed: {e}")
        print("  Check: pip install bertopic umap-learn hdbscan")
        df["topic_id"]          = -1
        df["topic_probability"] = 0.0
        df["topic_label"]       = "unknown"
        topic_keywords          = pd.DataFrame()

    # stage 3: behavioral labels
    print("\n[stage 3] Assigning behavioral labels...")
    df = assign_behavioral_labels(df)

    # save to PostgreSQL
    print("\n[save] Writing NLP features to PostgreSQL...")
    save_nlp_results(df, NLP_CONFIG["db_update_batch"])

    # export csv
    print("\n[export] Exporting NLP results...")
    path_rf = export_nlp_results(
        df, topic_keywords, timestamp
    )

    # final summary
    total = len(df)

    print("\n" + "=" * 55)
    print("  NLP EXTRACTION COMPLETE")
    print("=" * 55)
    print(f"  Records processed  : {total:,}")

    if "sentiment_label" in df.columns:
        for label in ["positive", "neutral", "negative"]:
            count = int(
                (df["sentiment_label"] == label).sum()
            )
            pct   = count / total * 100
            print(f"  Sentiment {label:<10}: "
                  f"{count:>6,} ({pct:.1f}%)")

    if "topic_id" in df.columns:
        n_topics = int(
            df[df["topic_id"] != -1]["topic_id"].nunique()
        )
        print(f"  Topics discovered  : {n_topics}")

    if "predicted_behavior" in df.columns:
        print(f"\n  Behavioral labels:")
        for label in ["purchase_intent",
                      "churn_risk",
                      "passive_browsing"]:
            count = int(
                (df["predicted_behavior"] == label).sum()
            )
            pct   = count / total * 100
            print(f"    {label:<20}: "
                  f"{count:>6,} ({pct:.1f}%)")

    print(f"\n  Output directory   : {NLP_OUTPUT_DIR}/")
    print(f"  RF input file      : {path_rf}")
    print(f"  Timestamp          : {timestamp}")
    print("=" * 55)
    print("\n  Next step → Run: python random_forest.py")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()