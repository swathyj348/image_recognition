from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = 'quality'
HIGH_QUALITY_THRESHOLD = 7


def load_wine_frame(csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame.columns = [column.strip().replace(' ', '_') for column in frame.columns]
    if 'type' in frame.columns:
        frame['type'] = frame['type'].fillna(frame['type'].mode(dropna=True).iloc[0])
    for column in frame.columns:
        if column != 'type':
            frame[column] = pd.to_numeric(frame[column], errors='coerce')
    numeric_columns = frame.select_dtypes(include=[np.number]).columns
    frame[numeric_columns] = frame[numeric_columns].replace([np.inf, -np.inf], np.nan)
    frame[numeric_columns] = frame[numeric_columns].fillna(frame[numeric_columns].median())
    return frame


def prepare_wine_data(frame: pd.DataFrame):
    working = frame.copy()
    working['high_quality'] = (working[TARGET_COLUMN] >= HIGH_QUALITY_THRESHOLD).astype(int)
    working = pd.get_dummies(working, columns=['type'], drop_first=True)

    target = working['high_quality'].astype(np.float32).values
    feature_frame = working.drop(columns=[TARGET_COLUMN, 'high_quality'])
    feature_names = list(feature_frame.columns)
    features = feature_frame.astype(np.float32).values
    return features, target, feature_names, feature_frame


def split_and_scale(features, target, test_size=0.2, val_size=0.2, random_state=42):
    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    relative_val_size = val_size / (1 - test_size)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_temp,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    return x_train, x_val, x_test, y_train, y_val, y_test, scaler
