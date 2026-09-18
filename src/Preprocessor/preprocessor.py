# IMPORTS
import os
import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# SCIKIT-LEARN

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, RobustScaler, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# FUNCTION

from src.Analyze_data.export_analyse import load_features_from_db
from src.config import IMAGE_DIR


# =======================================================
# DATA
df = load_features_from_db()

def image_path(row):
    art_id = str(row['article_id']).zfill(10)
    img_name = art_id + '.jpg'
    idx = art_id[:3]
    return os.path.join(IMAGE_DIR, idx, img_name)




def data_preprocessor(df):
    df = df.copy()
    df['image_path'] = df.apply(image_path, axis=1)
    img_paths_arr = df['image_path'].values
    article_ids = df['article_id'].values
    y = df['target_price']
    X = df.drop(columns=['target_price', 'index_name', 'article_id', 'image_path'], errors='ignore')

    num_cols = X.select_dtypes(include='number').columns.drop(['total_count', 'desc_length'], errors='ignore')
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.drop(['clean_desc', 'prod_name'], errors='ignore')

    log_transformer = FunctionTransformer(np.log1p, validate=True)

    # =======================================================
    # PIPELINES
    # =======================================================

    cat = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
        ])

    num = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])

    text_desc_pipe = Pipeline(\
        steps=[
        ('vect_desc', TfidfVectorizer(stop_words='english'))
    ])

    text_prod_pipe = Pipeline(
        steps=[
        ('vect_prod', TfidfVectorizer(stop_words='english'))
    ])

    log = Pipeline(
        steps=[
            ('logn', log_transformer),
            ('scaler', StandardScaler()),
        ])
    # =======================================================
    # SPLIT
    # =======================================================

    X_train, X_test, y_train, y_test, paths_train, paths_test, ids_train, ids_test = train_test_split(
        X, y, img_paths_arr, article_ids,
        test_size=0.2,
        random_state=42
    )

    # =======================================================
    # TRANSFORMER
    # =======================================================
    preprocessor = ColumnTransformer(transformers=[
        ('log_features', log, ['total_count', 'desc_length']),
        ('num_features', num, num_cols),
        ('cat_features', cat, cat_cols),
        ('text_desc', text_desc_pipe, 'clean_desc'),
        ('text_prod_name', text_prod_pipe, 'prod_name'),
    ])

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()
        X_test_proc = X_test_proc.toarray()

    return (X_train_proc, X_test_proc,
            y_train, y_test,
            paths_train, paths_test,
            ids_train, ids_test, preprocessor)

    # =======================================================
    # =======================================================

