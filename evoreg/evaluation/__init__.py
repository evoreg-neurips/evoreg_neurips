"""
Evaluation module for EvoReg.

Provides comprehensive evaluation metrics for point cloud registration
including Chamfer distance, Earth Mover's distance, Sliced Wasserstein
distance, point-to-point error metrics, Feature Matching Recall with KPConv,
Registration Recall metrics, Correspondence Error, and Geodesic Distance.
"""

from .point_to_point import Point_to_Point_Error, P2PError
from .earth_movers_distance import (
    earth_movers_distance,
    approximate_emd_sinkhorn,
    EarthMoversDistance
)
from .sliced_wasserstein_distance import (
    sliced_wasserstein_distance,
    max_sliced_wasserstein_distance,
    adaptive_sliced_wasserstein_distance,
    SlicedWassersteinDistance
)
from .feature_matching_recall import (
    FeatureMatchingRecall,
    compute_inlier_ratio_kpconv,
    KPConvFeatureExtractor
)
from .registration_recall import (
    RegistrationRecall,
    registration_recall_dataset,
    calculate_rmse_knn
)
from .registration_error import (
    registration_error,
    registration_error_with_transformation,
    RegistrationError
)
from .Correspondence_error import Correspondence_Error
from .Geodesic_Distance import Geodesic_Distance
from .rotation_error import rotation_error, RotationError
from .translation_error import translation_error, TranslationError

__all__ = [
    # Point-to-point metrics
    'Point_to_Point_Error',
    'P2PError',

    # Earth Mover's Distance
    'earth_movers_distance',
    'approximate_emd_sinkhorn',
    'EarthMoversDistance',

    # Sliced Wasserstein Distance
    'sliced_wasserstein_distance',
    'max_sliced_wasserstein_distance',
    'adaptive_sliced_wasserstein_distance',
    'SlicedWassersteinDistance',

    # Feature Matching Recall with KPConv
    'FeatureMatchingRecall',
    'compute_inlier_ratio_kpconv',
    'KPConvFeatureExtractor',

    # Registration Recall
    'RegistrationRecall',
    'registration_recall_dataset',
    'calculate_rmse_knn',

    # CPD Registration Error
    'registration_error',
    'registration_error_with_transformation',
    'RegistrationError',

    # Correspondence Error
    'Correspondence_Error',

    # Geodesic Distance
    'Geodesic_Distance',

    # Rotation Error
    'rotation_error',
    'RotationError',

    # Translation Error
    'translation_error',
    'TranslationError'
]