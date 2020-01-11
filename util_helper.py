import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN


def resample_to_csv(X, y, randomState, path):
    smote_enn = SMOTEENN(random_state=randomState)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)
    X_resampled['BK'] = y_resampled
    X_resampled.to_csv(path)
