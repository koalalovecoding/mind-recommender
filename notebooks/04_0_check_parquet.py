## Load processed Phase 2 parquet files and verify that they can be read successfully.
from pathlib import Path
import pandas as pd
DATA_DIR = Path("../data/processed")
train_path = DATA_DIR / "interactions_history_train.parquet"
dev_path = DATA_DIR / "interactions_history_dev.parquet"
news_path = DATA_DIR / "news.parquet"
train_df = pd.read_parquet(train_path)
dev_df = pd.read_parquet(dev_path)
news_df = pd.read_parquet(news_path)
print("interactions_history_train.parquet:", train_df.shape)
print("interactions_history_dev.parquet:", dev_df.shape)
print("news.parquet:", news_df.shape)

