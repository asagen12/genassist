"""
Feature Engineering column names: only Custom Expression needs a New Column
Name. Every other strategy names its columns automatically when it has none
(log_price, pca_1, poly_year^2, price_bin, ...), never clashing with an
existing column or an earlier feature, and features saved with a name keep it.
"""

import pickle
import uuid

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions.exception_classes import AppException
from app.modules.workflow.engine.nodes.ml.ml_model_inference_node import _replay_feature_engineering
from app.modules.workflow.engine.nodes.ml.train_model_node import TrainModelNode
from app.modules.workflow.engine.workflow_state import WorkflowState


@pytest.fixture
def node() -> TrainModelNode:
    return TrainModelNode(node_id="test", node_config={}, state=None)


def _data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    make = lambda k: pd.DataFrame({
        "year": rng.integers(2020, 2025, k).astype(float),
        "month": rng.integers(1, 13, k).astype(float),
        "price": rng.uniform(1, 100, k),
        "income": rng.uniform(10, 50, k),
    })
    return make(n), make(10)


def _columns(node, features):
    X_train, X_val = _data()
    X_train, X_val, steps = node._engineer_features(X_train, X_val, features)
    return list(X_train.columns), steps, X_val


ORIGINAL = ["year", "month", "price", "income"]


class TestAutomaticNames:
    @pytest.mark.parametrize(
        "feature, expected",
        [
            ({"strategy": "log_transform", "sourceColumns": ["price", "income"]}, ["log_price", "log_income"]),
            ({"strategy": "log_transform", "sourceColumns": ["price"]}, ["log_price"]),  # column kept in the name
            ({"strategy": "quantile_transform", "sourceColumns": ["price"]}, ["quantile_price"]),
            ({"strategy": "power_transform", "sourceColumns": ["price", "income"]}, ["power_price", "power_income"]),
            ({"strategy": "bin_numeric", "binColumn": "price", "numBins": 3}, ["price_bin"]),
            ({"strategy": "polynomial", "polynomialColumns": ["year", "month"], "polynomialDegree": 2},
             ["poly_year^2", "poly_year month", "poly_month^2"]),
        ],
    )
    def test_each_strategy_names_its_columns(self, node, feature, expected):
        columns, _, _ = _columns(node, [feature])
        assert columns == ORIGINAL + expected

    def test_pca_numbers_its_components_and_replaces_sources(self, node):
        columns, _, _ = _columns(node, [{"strategy": "pca", "sourceColumns": ["price", "income"], "pcaComponents": 2}])
        assert columns == ["year", "month", "pca_1", "pca_2"]

    def test_an_empty_string_name_counts_as_no_name(self, node):
        columns, _, _ = _columns(node, [{"newColumnName": "", "strategy": "log_transform", "sourceColumns": ["price"]}])
        assert columns[-1] == "log_price"


class TestClashesAreNumbered:
    def test_two_pcas_get_separate_names(self, node):
        columns, _, _ = _columns(node, [
            {"strategy": "pca", "sourceColumns": ["price", "income"], "pcaComponents": 1, "replaceSourceColumns": False},
            {"strategy": "pca", "sourceColumns": ["year", "month"], "pcaComponents": 1, "replaceSourceColumns": False},
        ])
        assert columns[-2:] == ["pca_1", "pca2_1"]

    def test_the_same_log_twice(self, node):
        columns, _, _ = _columns(node, [
            {"strategy": "log_transform", "sourceColumns": ["price"]},
            {"strategy": "log_transform", "sourceColumns": ["price"]},
        ])
        assert columns[-2:] == ["log_price", "log2_price"]

    def test_bin_numeric_twice(self, node):
        columns, _, _ = _columns(node, [
            {"strategy": "bin_numeric", "binColumn": "price", "numBins": 3},
            {"strategy": "bin_numeric", "binColumn": "price", "numBins": 5},
        ])
        assert columns[-2:] == ["price_bin", "price_bin_2"]

    def test_a_data_column_already_using_the_name(self, node):
        X_train, X_val = _data()
        X_train["log_price"] = 1.0
        X_val["log_price"] = 1.0
        X_train, _, _ = node._engineer_features(X_train, X_val, [{"strategy": "log_transform", "sourceColumns": ["price"]}])
        assert "log2_price" in X_train.columns

    def test_a_later_feature_can_use_an_automatic_name(self, node):
        columns, _, _ = _columns(node, [
            {"strategy": "log_transform", "sourceColumns": ["price"]},
            {"strategy": "pca", "sourceColumns": ["log_price", "income"], "pcaComponents": 1},
        ])
        assert columns[-1] == "pca_1"


class TestNamedFeaturesUnchanged:
    def test_saved_names_are_kept(self, node):
        columns, _, _ = _columns(node, [
            {"newColumnName": "lp", "strategy": "log_transform", "sourceColumns": ["price"]},
            {"newColumnName": "q", "strategy": "quantile_transform", "sourceColumns": ["price", "income"]},
            {"newColumnName": "f", "strategy": "polynomial", "polynomialColumns": ["year"], "polynomialDegree": 2},
            {"newColumnName": "pb", "strategy": "bin_numeric", "binColumn": "price", "numBins": 3},
        ])
        assert columns == ORIGINAL + ["lp", "q_price", "q_income", "f_year^2", "pb"]

    def test_custom_expression_uses_its_name(self, node):
        columns, _, _ = _columns(node, [{"newColumnName": "ym", "strategy": "custom_expression", "expression": "year * month"}])
        assert columns[-1] == "ym"


class TestReplayUsesTheAutomaticNames:
    def test_inference_rebuilds_the_same_columns(self, node):
        _, steps, _ = _columns(node, [
            {"strategy": "log_transform", "sourceColumns": ["price"]},
            {"strategy": "bin_numeric", "binColumn": "income", "numBins": 3},
            {"strategy": "polynomial", "polynomialColumns": ["year", "month"], "polynomialDegree": 2},
            {"strategy": "pca", "sourceColumns": ["log_price", "income"], "pcaComponents": 1},
        ])
        replayed = _replay_feature_engineering(
            {"year": [2022.0], "month": [6.0], "price": [10.0], "income": [20.0]}, steps, 1
        )
        assert {"log_price", "income_bin", "poly_year^2", "poly_year month", "poly_month^2", "pca_1"} <= set(replayed)


class TestValidation:
    @pytest.mark.asyncio
    async def test_custom_expression_still_needs_a_name(self, tmp_path):
        with pytest.raises(AppException) as exc_info:
            await _train(tmp_path, [{"strategy": "custom_expression", "expression": "year * month"}])
        assert "Feature engineering #1 (Custom Expression) needs a New Column Name" in exc_info.value.error_detail

    def test_unnamed_feature_errors_refer_to_its_number(self):
        with pytest.raises(AppException) as exc_info:
            TrainModelNode._validate_column_transform_config({"strategy": "pca", "sourceColumns": ["a"]}, 2)
        assert exc_info.value.error_detail.startswith("Feature engineering #3 (PCA):")


async def _train(tmp_path, features):
    X, _ = _data(80)
    X["target"] = X["price"] * 0.5 + X["income"]
    path = tmp_path / f"auto_{uuid.uuid4().hex[:6]}.csv"
    X.to_csv(path, index=False)
    node = TrainModelNode(node_id=str(uuid.uuid4()), node_config={}, state=WorkflowState(workflow={}))
    return await node.process({
        "name": f"auto-{uuid.uuid4().hex[:8]}", "modelType": "linear_regression", "fileUrl": str(path),
        "targetColumn": "target", "featureColumns": ORIGINAL, "validationSplit": 0.25,
        "featureEngineering": features,
    })


@pytest.mark.asyncio
async def test_unnamed_features_train_end_to_end(tmp_path):
    result = await _train(tmp_path, [
        {"strategy": "log_transform", "sourceColumns": ["price"]},
        {"strategy": "polynomial", "polynomialColumns": ["year", "month"], "polynomialDegree": 2},
        {"newColumnName": "ratio", "strategy": "custom_expression", "expression": "price / income"},
    ])
    assert result["success"] is True
    with open(result["model_file_path"], "rb") as f:
        metadata = pickle.load(f)["metadata"]
    assert {"log_price", "poly_year month", "ratio"} <= set(metadata["model_input_columns"])


class TestShownDefaultsAreUsed:
    """The dialog shows degree 2 / 5 bins but used not to save them, so a
    Polynomial or Bin Numeric feature added without touching those fields
    failed training ("missing polynomialColumns or polynomialDegree")."""

    @pytest.mark.asyncio
    async def test_polynomial_without_a_degree_trains_with_degree_2(self, tmp_path):
        result = await _train(tmp_path, [{"strategy": "polynomial", "polynomialColumns": ["year", "month"]}])
        with open(result["model_file_path"], "rb") as f:
            cols = pickle.load(f)["metadata"]["model_input_columns"]
        assert {"poly_year^2", "poly_year month", "poly_month^2"} <= set(cols)
        assert not any(c.endswith("^3") for c in cols)

    @pytest.mark.asyncio
    async def test_bin_numeric_without_a_bin_count_uses_5_bins(self, tmp_path):
        result = await _train(tmp_path, [{"strategy": "bin_numeric", "binColumn": "price"}])
        with open(result["model_file_path"], "rb") as f:
            steps = pickle.load(f)["metadata"]["feature_engineering_steps"]
        assert len(steps[0]["bin_edges"]) == 6  # 5 bins

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "feature, message",
        [
            ({"strategy": "polynomial", "polynomialColumns": ["year"], "polynomialDegree": 1}, "degree must be a whole number of 2 or more"),
            ({"strategy": "polynomial"}, "(Polynomial): choose at least one column"),
            ({"strategy": "bin_numeric", "binColumn": "price", "numBins": 1}, "number of bins must be a whole number of 2 or more"),
            ({"strategy": "bin_numeric"}, "(Bin Numeric): choose a column to bin"),
        ],
    )
    async def test_invalid_settings_fail_clearly(self, tmp_path, feature, message):
        with pytest.raises(AppException) as exc_info:
            await _train(tmp_path, [feature])
        assert message in exc_info.value.error_detail
