"""Label-free feature interactions shared by fitted constrained model pipelines."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from scripts.common import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


INTERACTION_NUMERIC = NUMERIC_FEATURES + [
    "log_telemetry_age", "log_rated_power", "health_is_zero", "health_squared",
    "log_age_x_health", "log_age_x_log_power", "health_x_log_power",
]
INTERACTION_CATEGORICAL = CATEGORICAL_FEATURES + ["region_device"]


class FaultInteractions(TransformerMixin, BaseEstimator):
    """Use only information available when the ticket is raised; learn no labels."""

    def fit(self, x, y=None):
        self.transform(x)
        self.n_features_in_ = len(FEATURES)
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        return self

    def transform(self, x):
        output = x.loc[:, FEATURES].copy()
        output["log_telemetry_age"] = np.log1p(output.telemetry_age_minutes)
        output["log_rated_power"] = np.log1p(output.rated_power_kw)
        output["health_is_zero"] = output.reported_health_score.eq(0).astype(int)
        output["health_squared"] = output.reported_health_score ** 2
        output["log_age_x_health"] = output.log_telemetry_age * output.reported_health_score
        output["log_age_x_log_power"] = output.log_telemetry_age * output.log_rated_power
        output["health_x_log_power"] = output.reported_health_score * output.log_rated_power
        # Length prefixes prevent collisions even if a category contains separators.
        region, device = output.region.astype(str), output.device_type.astype(str)
        output["region_device"] = region.str.len().astype(str) + ":" + region + device
        return output
