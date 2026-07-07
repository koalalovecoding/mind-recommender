#Step 3&4 Parse impression logs and user history
from pathlib import Path
import pandas as pd
Data_Dir = Path("../data/raw")
train_behaviors_path = Data_Dir / "MINDsmall_train" / "behaviors.tsv"
train_news_path = Data_Dir / "MINDsmall_train" / "news.tsv"
dev_behaviors_path = Data_Dir / "MINDsmall_dev" / "behaviors.tsv"
dev_news_path = Data_Dir / "MINDsmall_dev" / "news.tsv"

behaviors_cols=[
 "impression_id",
    "user_id",
    "time",
    "history",
    "impressions",
]

news_cols = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

train_behaviors = pd.read_csv(train_behaviors_path, sep = '\t', names=behaviors_cols, header=None)
dev_behaviors = pd.read_csv(dev_behaviors_path, sep='\t', names=behaviors_cols, header=None)
train_news = pd.read_csv(train_news_path, sep="\t", names=news_cols, header=None)
dev_news = pd.read_csv(dev_news_path, sep='\t', names=news_cols, header=None)

print("train_behaviors:", train_behaviors.shape)
print("dev_behaviors:", dev_behaviors.shape)
print("train_news:", train_news.shape)
print("dev_news:", dev_news.shape)

#I converted the raw time column into pandas datetime format so that I could sort user events chronologically and support time-based splitting. I used errors="coerce" to safely handle malformed timestamps by converting them into missing datetime values, which can then be inspected or filtered.
train_behaviors ['time'] = pd.to_datetime(train_behaviors['time'],errors="coerce")
dev_behaviors ['time'] = pd.to_datetime(dev_behaviors['time'], errors="coerce")

def parse_impressions_history(behaviors_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in behaviors_df.itertuples(index=False):
        impression_id = row.impression_id
        user_id = row.user_id
        time = row.time
        impressions = row.impressions
        history = row.history

        if pd.isna(impressions):
            continue

        impressions_items = impressions.split()

        for item in impressions_items:
            try:
                item_id, click = item.rsplit("-", 1)
            except ValueError:
                continue

            records.append({
                "user_id": user_id,
                "item_id": item_id,
                "click": int(click),
                "impression_id": impression_id,
                "time": time,
                "source": "impression"
            })

        if isinstance(history,str):
            history_items = history.split()
            for item in history_items:
                item_id, click = item, 1
                records.append({
                    "user_id": user_id,
                    "item_id": item_id,
                    "click": int(click),
                    "impression_id": impression_id,
                    "time": time,
                    "source": "history"
                })




    interactions_history_df = pd.DataFrame(records)
    return interactions_history_df


# Parse train and dev impression logs
train_impressions_history = parse_impressions_history(train_behaviors)
dev_impressions_history = parse_impressions_history(dev_behaviors)

# =========================
# Save parsed Step 3 & 4 interaction tables
# =========================

Processed_Dir = Path("../data/processed")
Processed_Dir.mkdir(parents=True, exist_ok=True)

# Since train_impressions_history contains both:
# 1. impression rows
# 2. history rows
# we save it as the general interaction table.

train_interactions_path = Processed_Dir / "interactions_train.parquet"
dev_interactions_path = Processed_Dir / "interactions_dev.parquet"
news_path = Processed_Dir / "news.parquet"

# Save train and dev interaction tables
train_impressions_history.to_parquet(
    train_interactions_path,
    index=False,
)

dev_impressions_history.to_parquet(
    dev_interactions_path,
    index=False,
)

# Combine train_news and dev_news, then remove duplicate news_id rows
news_all = pd.concat([train_news, dev_news], ignore_index=True)
news_all = news_all.drop_duplicates(subset=["news_id"])

news_all.to_parquet(
    news_path,
    index=False,
)

print("Saved processed files:")
print("Train interactions:", train_interactions_path)
print("Dev interactions:", dev_interactions_path)
print("News metadata:", news_path)

print("\nTrain interaction shape:", train_impressions_history.shape)
print("Dev interaction shape:", dev_impressions_history.shape)
print("News shape:", news_all.shape)

print("\nTrain source counts:")
print(train_impressions_history["source"].value_counts())

print("\nDev source counts:")
print(dev_impressions_history["source"].value_counts())

print("\nTrain click counts:")
print(train_impressions_history["click"].value_counts())

print("\nDev click counts:")
print(dev_impressions_history["click"].value_counts())


#sanity check
assert set(train_impressions_history["source"].unique()) == {"impression", "history"}
assert set(dev_impressions_history["source"].unique()) == {"impression", "history"}

assert train_impressions_history["user_id"].notna().all()
assert train_impressions_history["item_id"].notna().all()
assert train_impressions_history["click"].isin([0, 1]).all()

assert train_impressions_history[
    train_impressions_history["source"] == "history"
]["click"].eq(1).all()

assert not train_impressions_history["item_id"].eq("nan").any()
assert not dev_impressions_history["item_id"].eq("nan").any()