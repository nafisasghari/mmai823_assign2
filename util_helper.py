import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import BorderlineSMOTE, ADASYN


def resample_to_csv(X, y, random_state, path, method):
    """Re-samples dataset using desired method of oversampling and writes output to CSV.

    :param X: Original Features
    :param y: Original Labels
    :param randomState: Random intialization
    :param path: Path to output location and name of CSV
    :param method: Either SMOTEN-NN method or BorderLineSMOTE (borderline) method.
    See imbalanced-learn documentation for more information.
    https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.BorderlineSMOTE.html

    :return: none
    """

    if method == 'SMOTEN-NN':
        smote_enn = SMOTEENN(random_state=random_state)
        X_resampled, y_resampled = smote_enn.fit_resample(X, y)
        X_resampled['BK'] = y_resampled
        X_resampled.to_csv(path)
    elif method == 'borderline':
        borderlineSmote = BorderlineSMOTE(random_state=random_state)
        X_resampled, y_resampled = borderlineSmote.fit_resample(X, y)
        X_resampled['BK'] = y_resampled
        X_resampled.to_csv(path)

    elif method == 'adasyn':
        adasyn = ADASYN(random_state=random_state)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        X_resampled['BK'] = y_resampled
        X_resampled.to_csv(path)

    elif method == 'tomek':
        tomek = SMOTETomek(random_state=random_state)
        X_resampled, y_resampled = tomek.fit_resample(X, y)
        X_resampled['BK'] = y_resampled
        X_resampled.to_csv(path)