#step 5: merge news metadata
#In this step, I merged parsed user-item interaction records with news metadata from news.tsv,
# enriching each interaction with category, subcategory, title, and abstract fields.
# This preserves the implicit-feedback interaction structure while preparing the data for
# later hybrid recommendation models such as LightFM and content-enhanced retrieval.

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


print("train before merge:", train_df.shape)
print("train after merge:", train_with_news.shape)

print("dev before merge:", dev_df.shape)
print("dev after merge:", dev_with_news.shape)

print(train_with_news.head().to_string())

train_with_news.to_parquet(DATA_DIR/"train_with_news.parquet", index=False)
dev_with_news.to_parquet(DATA_DIR/"dev_with_news.parquet", index=False)