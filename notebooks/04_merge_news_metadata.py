# Step 5: Create cleaned interaction tables with news metadata.
# Merge parsed user-item interactions with news metadata from news.tsv, add category,
# subcategory, title, and abstract fields, and keep item_id as the standard item identifier.
# The resulting train/dev tables are the cleaned files used for downstream modeling.

from pathlib import Path
import pandas as pd
DATA_DIR = Path("../data/processed")
output_train_path = DATA_DIR / "interactions_history_train.parquet"
output_dev_path = DATA_DIR / "interactions_history_dev.parquet"
output_news_path = DATA_DIR / "news.parquet"
train_df = pd.read_parquet(output_train_path)
dev_df = pd.read_parquet(output_dev_path)
news_df = pd.read_parquet(output_news_path)

# Keep only the metadata columns needed for Phase 2.
news_meta = news_df[ [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",]
    ].drop_duplicates(subset=['news_id'])

## Use a left join to keep all interaction rows, even if some items do not have matching news metadata.
def interactions_news_merge(interactions_df, news_df):
    return interactions_df.merge(news_df,
        left_on = "item_id",
        right_on = 'news_id',
        how = 'left',
        )

train_with_news = interactions_news_merge(train_df, news_meta)
dev_with_news = interactions_news_merge(dev_df, news_meta)

# After merging, news_id duplicates item_id, so we drop news_id and keep item_id
# as the standard item identifier for downstream recommendation models.

# In pandas, drop(columns=["news_id"]) removes the column named "news_id".
# If we wrote drop(["news_id"]) without columns=..., pandas would try to drop a row/index label instead.
train_with_news = train_with_news.drop(columns=["news_id"])
dev_with_news = dev_with_news.drop(columns=["news_id"])

print("train before merge:", train_df.shape)
print("train after merge:", train_with_news.shape)

print("dev before merge:", dev_df.shape)
print("dev after merge:", dev_with_news.shape)

print(train_with_news.head().to_string())

train_with_news.to_parquet(DATA_DIR/"train_with_news.parquet", index=False)
dev_with_news.to_parquet(DATA_DIR/"dev_with_news.parquet", index=False)