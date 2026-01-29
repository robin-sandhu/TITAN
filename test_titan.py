"""
TITAN Unit and Integration Tests

Run with: pytest test_titan.py -v
"""

# Import TITAN components
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from TITAN import (
    RANDOM_STATE,
    AuditLog,
    SyntheticDataGenerator,
    bootstrap_auc_ci,
    calibration_slope_intercept,
    infer_task_type,
    load_model,
    normalize_binary_target,
    profile_dataset,
    save_model,
    schema_dictionary,
)


class TestSyntheticDataGenerator:
    """Tests for synthetic data generation."""

    def test_binary_classification_basic(self):
        """Test basic synthetic data generation."""
        df = SyntheticDataGenerator.generate_binary_classification(n_samples=100)
        assert len(df) == 100
        assert "outcome" in df.columns
        assert df["outcome"].nunique() == 2

    def test_binary_classification_custom_params(self):
        """Test synthetic data with custom parameters."""
        df = SyntheticDataGenerator.generate_binary_classification(
            n_samples=500,
            n_features=30,
            n_informative=15,
            n_categorical=4,
            class_ratio=0.4,
            missing_rate=0.1,
        )
        assert len(df) == 500
        assert df.isnull().sum().sum() > 0  # Should have missing values

    def test_survival_data(self):
        """Test survival data generation."""
        df = SyntheticDataGenerator.generate_survival_data(n_samples=200)
        assert len(df) == 200
        assert "time" in df.columns
        assert "event" in df.columns
        assert df["event"].isin([0, 1]).all()


class TestNormalizeBinaryTarget:
    """Tests for binary target normalization."""

    def test_numeric_01(self):
        """Test already binary 0/1 target."""
        y = pd.Series([0, 1, 1, 0, 1])
        result = normalize_binary_target(y)
        assert result is not None
        assert set(result.unique()) == {0, 1}

    def test_string_labels(self):
        """Test string labels."""
        y = pd.Series(["No", "Yes", "Yes", "No", "Yes"])
        result = normalize_binary_target(y)
        assert result is not None
        assert set(result.unique()) == {0, 1}

    def test_with_nan(self):
        """Test handling of NaN values."""
        y = pd.Series([0, 1, np.nan, 0, 1])
        result = normalize_binary_target(y)
        assert result is not None

    def test_non_binary_returns_none(self):
        """Test that non-binary targets return None."""
        y = pd.Series([0, 1, 2, 3])
        result = normalize_binary_target(y)
        assert result is None


class TestCalibrationSlope:
    """Tests for calibration slope computation."""

    def test_basic_calibration(self):
        """Test basic calibration slope calculation."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.2, 0.6])
        slope, intercept = calibration_slope_intercept(y_true, y_prob)
        assert not np.isnan(slope)
        assert not np.isnan(intercept)

    def test_perfect_calibration(self):
        """Test with perfectly calibrated predictions."""
        np.random.seed(42)
        n = 1000
        y_true = np.random.binomial(1, 0.5, n)
        # Perfect calibration: p(y=1|prob) = prob
        y_prob = np.random.uniform(0.2, 0.8, n)
        y_true = (np.random.random(n) < y_prob).astype(int)

        slope, intercept = calibration_slope_intercept(y_true, y_prob)
        # Should be close to slope=1, intercept=0
        assert abs(slope - 1.0) < 0.5
        assert abs(intercept) < 0.5


class TestBootstrapAUC:
    """Tests for bootstrap AUC confidence intervals."""

    def test_basic_ci(self):
        """Test basic CI computation."""
        np.random.seed(42)
        y_true = np.random.choice([0, 1], size=200)
        y_prob = y_true * 0.6 + np.random.uniform(0, 0.4, 200)

        ci_low, ci_high = bootstrap_auc_ci(y_true, y_prob, n_bootstrap=100)

        assert ci_low < ci_high
        assert 0 <= ci_low <= 1
        assert 0 <= ci_high <= 1

    def test_small_sample_returns_nan(self):
        """Test that small samples return NaN."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])

        ci_low, ci_high = bootstrap_auc_ci(y_true, y_prob)

        assert np.isnan(ci_low)
        assert np.isnan(ci_high)


class TestDataProfiling:
    """Tests for data profiling functions."""

    def test_profile_dataset(self):
        """Test dataset profiling."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, np.nan],
                "b": ["x", "y", "x", "y"],
                "c": [1.0, 2.0, 3.0, 4.0],
            }
        )

        profile = profile_dataset(df)

        assert profile["rows"] == 4
        assert profile["cols"] == 3
        assert profile["missing_cells"] == 1
        assert profile["numeric_cols"] == 2

    def test_schema_dictionary(self):
        """Test schema dictionary generation."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        schema = schema_dictionary(df)

        assert len(schema) == 2
        assert "variable" in schema.columns
        assert "dtype" in schema.columns


class TestTaskInference:
    """Tests for task type inference."""

    def test_binary_detection(self):
        """Test binary classification detection."""
        df = pd.DataFrame({"target": [0, 1, 0, 1], "feature": [1.0, 2.0, 3.0, 4.0]})

        result = infer_task_type(df, "target")
        assert result["task_type"] == "binary"

    def test_multiclass_detection(self):
        """Test multiclass detection."""
        df = pd.DataFrame(
            {"target": ["a", "b", "c", "a", "b"], "feature": [1, 2, 3, 4, 5]}
        )

        result = infer_task_type(df, "target")
        assert result["task_type"] == "multiclass"


class TestModelSerialization:
    """Tests for model save/load."""

    def test_save_and_load_model(self):
        """Test model serialization round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            audit = AuditLog(tmpdir / "test_audit.jsonl")

            # Create mock model dict
            model_dict = {
                "model": None,
                "base": None,
                "preprocessor": None,
                "feature_names": ["f1", "f2", "f3"],
                "auc": 0.85,
                "brier": 0.15,
            }

            # Save
            model_path = save_model(model_dict, tmpdir, audit, use_joblib=False)
            assert model_path.exists()

            # Load
            loaded = load_model(model_path, audit)
            assert loaded["feature_names"] == ["f1", "f2", "f3"]
            assert loaded["auc"] == 0.85


class TestAuditLog:
    """Tests for audit logging."""

    def test_audit_log_creation(self):
        """Test audit log file creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_log.jsonl"
            audit = AuditLog(log_path)

            audit.log("TEST_EVENT", {"key": "value"})

            assert log_path.exists()
            with open(log_path) as f:
                content = f.read()
                assert "TEST_EVENT" in content


class TestIntegration:
    """Integration tests for full pipeline."""

    @pytest.mark.slow
    def test_full_pipeline_synthetic(self):
        """Test full pipeline with synthetic data."""
        from TITAN import run_infinity_on_file

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Generate synthetic data
            df = SyntheticDataGenerator.generate_binary_classification(n_samples=300)
            csv_path = tmpdir / "test_data.csv"
            df.to_csv(csv_path, index=False)

            # Run pipeline
            output_dir = tmpdir / "output"
            result = run_infinity_on_file(
                csv_path,
                output_dir,
                eda_level="basic",
                interactive_target=False,
                session_config=None,
            )

            assert result.get("status") == "success"
            assert Path(result["output_dir"]).exists()


# Fixtures for shared test data
@pytest.fixture
def sample_binary_df():
    """Create sample binary classification DataFrame."""
    np.random.seed(RANDOM_STATE)
    n = 200
    return pd.DataFrame(
        {
            "age": np.random.normal(60, 15, n),
            "biomarker": np.random.exponential(1, n),
            "sex": np.random.choice(["M", "F"], n),
            "outcome": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        }
    )


@pytest.fixture
def temp_audit_log():
    """Create temporary audit log."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield AuditLog(Path(tmpdir) / "audit.jsonl")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
