# build_features.py

import pandas as pd
import numpy as np
import os

print("Starting feature engineering process...")

# =============================================================================
# ۱. تعریف مسیرها و ساخت پوشه خروجی
# =============================================================================
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

# اگر پوشه برای داده‌های پردازش‌شده وجود ندارد، آن را بساز
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# =============================================================================
# ۲. بارگذاری داده‌های خام
# =============================================================================
try:
    p_train_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "p_train.csv"), index_col=0)
    p_test_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "p_test.csv"), index_col=0)
    
    with open(os.path.join(RAW_DATA_DIR, 'banners.txt'), 'r', encoding='utf-8') as f:
        banner_captions = [line.strip() for line in f.readlines()]
    banners_df = pd.DataFrame(banner_captions, columns=['caption'])
    
    print("Raw data loaded successfully.")
    print(f"Train shape: {p_train_df.shape}")
    print(f"Test shape: {p_test_df.shape}")
    print(f"Banners count: {len(banners_df)}")

except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please make sure the raw data files are in the 'data/raw' directory.")
    exit()

# =============================================================================
# ۳. مهندسی ویژگی از متن بنرها
# =============================================================================
print("Engineering features from banner captions...")

banners_df['text_length'] = banners_df['caption'].str.len()
banners_df['word_count'] = banners_df['caption'].str.split().str.len()

# =============================================================================
# ۴. مهندسی ویژگی از داده‌های تعامل (فقط از p_train)
# =============================================================================
print("Engineering features from interaction data (using train set only)...")

# --- ویژگی‌های مرتبط با بنر (Banner-Centric) ---
banner_features = p_train_df.groupby('j')['p'].agg(['mean', 'count', 'std']).reset_index()
banner_features.columns = ['j', 'banner_avg_score', 'banner_view_count', 'banner_score_std']

# --- ویژگی‌های مرتبط با سگمنت کاربر (User-Segment-Centric) ---
segment_features = p_train_df.groupby('i')['p'].agg(['mean', 'count', 'std']).reset_index()
segment_features.columns = ['i', 'segment_avg_score', 'segment_interaction_count', 'segment_score_std']

# پر کردن مقادیر NaN در انحراف معیار (برای مواردی که فقط یک نمونه دارند)
banner_features.fillna(0, inplace=True)
segment_features.fillna(0, inplace=True)

# =============================================================================
# ۵. ادغام ویژگی‌ها با دیتافریم‌های اصلی
# =============================================================================
print("Merging new features into train and test dataframes...")

# --- ادغام برای داده‌های آموزشی ---
train_df_featured = p_train_df.merge(banners_df, left_on='j', right_index=True, how='left')
train_df_featured = train_df_featured.merge(banner_features, on='j', how='left')
train_df_featured = train_df_featured.merge(segment_features, on='i', how='left')

# --- ادغام برای داده‌های آزمون ---
test_df_featured = p_test_df.merge(banners_df, left_on='j', right_index=True, how='left')
test_df_featured = test_df_featured.merge(banner_features, on='j', how='left')
test_df_featured = test_df_featured.merge(segment_features, on='i', how='left')

# پر کردن مقادیر NaN برای داده‌های آزمون
# (ممکن است یک بنر یا سگمنت در مجموعه آزمون باشد که در آموزشی نبوده)
# ما اینجا با میانگین کلی پر می‌کنیم
test_df_featured['banner_avg_score'].fillna(p_train_df['p'].mean(), inplace=True)
test_df_featured['segment_avg_score'].fillna(p_train_df['p'].mean(), inplace=True)
test_df_featured.fillna(0, inplace=True) # بقیه موارد را با صفر پر می‌کنیم

print("Feature merging complete.")
print(f"New train shape: {train_df_featured.shape}")
print(f"New test shape: {test_df_featured.shape}")
print("\nColumns added:")
for col in train_df_featured.columns:
    if col not in p_train_df.columns:
        print(f"- {col}")

# =============================================================================
# ۶. ذخیره دیتافریم‌های جدید
# =============================================================================
train_output_path = os.path.join(PROCESSED_DATA_DIR, "train_featured.csv")
test_output_path = os.path.join(PROCESSED_DATA_DIR, "test_featured.csv")

train_df_featured.to_csv(train_output_path)
test_df_featured.to_csv(test_output_path)

print(f"\n✅ Processed data saved successfully!")
print(f"Train data saved to: {train_output_path}")
print(f"Test data saved to: {test_output_path}")