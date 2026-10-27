"""
stagbdt
=======
Self-Throttling Asymmetric Gradient Boosted Decision Tree (ST-AGBDT) Library.
Pure NumPy implementation; zero external machine learning dependencies.
"""

from stagbdt.loss import AsymmetricLeastSquaresLoss
from stagbdt.tree import Tree, FlatTree
from stagbdt.model import STAGBDT
from stagbdt.conformal import ConformalCalibrator, EarlyExitCalibrator
from stagbdt.features import (
    FEATURE_NAMES,
    extract_features_single,
    extract_features_sequence,
    StreamingArrivalFeatureExtractor,
)

__all__ = [
    "STAGBDT",
    "Tree",
    "FlatTree",
    "AsymmetricLeastSquaresLoss",
    "ConformalCalibrator",
    "EarlyExitCalibrator",
    "FEATURE_NAMES",
    "extract_features_single",
    "extract_features_sequence",
    "StreamingArrivalFeatureExtractor",
]
