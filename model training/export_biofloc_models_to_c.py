"""
Export biofloc soft-sensor models to a compact C/C++ header.

This script is a cleaned, standalone version of the Colab export workflow used
for the biofloc reactor. It trains candidate models for each target, selects the
best exportable model by cross-validated R2, retrains the selected model on all
available rows for that target, and writes one combined Arduino-compatible
header file.

Expected biofloc model input order:
    input_raw[0] = DO_B(mg/L)
    input_raw[1] = ORP_B(mV)
    input_raw[2] = EC_B(mS/cm)
    input_raw[3] = pH_B
    input_raw[4] = Temp_B(C)
    input_raw[5] = Turbidty_B(NTU)

Example:
    python model_export/export_biofloc_models_to_c.py \
        --input-excel "split tank full data.xlsx" \
        --sheet-name "Biofloc tank" \
        --output-dir "exports/biofloc_c_models"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor


FEATURES_BIO = [
    "DO_B(mg/L)",
    "ORP_B(mV)",
    "EC_B(mS/cm)",
    "pH_B",
    "Temp_B(C)",
    "Turbidty_B(NTU)",
]

TARGETS_BIO = [
    "TAN_B(mg/L)",
    "TOC_B(mg/L)",
    "DOC_B(mg/L)",
    "TN_B(mg/L)",
    "DN_B(mg/L)",
]

EXPORTABLE_MODELS = {
    "LinearRegression",
    "DecisionTree",
    "KNN_k1",
    "KNN_k3",
    "RandomForest",
    "AdaBoost",
    "XGBoost",
}


@dataclass
class TargetResult:
    target: str
    n_samples: int
    best_model_name: str
    cv_metrics: Dict[str, Dict[str, object]]
    cv_best_mae: float
    cv_best_rmse: float
    cv_best_r2: float
    y_true_cv: np.ndarray
    y_pred_cv: np.ndarray
    loo_mae: float
    loo_rmse: float
    loo_r2: float
    y_true_loo: np.ndarray
    y_pred_loo: np.ndarray
    final_model: object
    feature_cols: List[str]
    X_train_final: np.ndarray
    y_train_final: np.ndarray


def get_candidate_models(random_state: int = 42) -> Dict[str, object]:
    return {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "DecisionTree": DecisionTreeRegressor(random_state=random_state),
        "KNN_k1": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=1)),
        ]),
        "KNN_k3": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=3)),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        ),
        "AdaBoost": AdaBoostRegressor(
            n_estimators=300,
            random_state=random_state,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        ),
    }


def cv_predictions_for_model(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 10,
    random_state: int = 42,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    n_samples = len(y)
    effective_splits = min(n_splits, n_samples)

    if effective_splits < 2:
        raise ValueError("At least 2 samples are required for cross-validation.")

    kf = KFold(
        n_splits=effective_splits,
        shuffle=True,
        random_state=random_state,
    )

    y_true_all = np.array(y, copy=True, dtype=float)
    y_pred_all = np.zeros_like(y_true_all, dtype=float)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        fitted = clone(model)
        fitted.fit(X_train, y_train)
        y_pred_all[test_idx] = fitted.predict(X_test)

    mae = mean_absolute_error(y_true_all, y_pred_all)
    rmse = float(np.sqrt(mean_squared_error(y_true_all, y_pred_all)))
    r2 = r2_score(y_true_all, y_pred_all)

    return mae, rmse, r2, y_true_all, y_pred_all


def cv_metrics_for_models(
    models: Dict[str, object],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 10,
    random_state: int = 42,
) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}

    for name, model in models.items():
        try:
            mae, rmse, r2, _, _ = cv_predictions_for_model(
                model,
                X,
                y,
                n_splits=n_splits,
                random_state=random_state,
            )
            results[name] = {
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2),
                "failed": False,
                "error": "",
            }
        except Exception as exc:
            results[name] = {
                "mae": np.nan,
                "rmse": np.nan,
                "r2": -np.inf,
                "failed": True,
                "error": str(exc),
            }

    return results


def loo_evaluation(model: object, X: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    if len(y) < 2:
        raise ValueError("At least 2 samples are required for leave-one-out evaluation.")

    loo = LeaveOneOut()
    y_true_all: List[float] = []
    y_pred_all: List[float] = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fitted = clone(model)
        fitted.fit(X_train, y_train)
        y_pred = fitted.predict(X_test)

        y_true_all.append(float(y_test[0]))
        y_pred_all.append(float(y_pred[0]))

    y_true = np.array(y_true_all, dtype=float)
    y_pred = np.array(y_pred_all, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2, y_true, y_pred


def format_target_name_for_print(name: str) -> str:
    formatted = str(name)
    if "(" in formatted and ")" in formatted:
        formatted = formatted.replace("(", " (")
    return formatted.replace("NO3_B", "NO3_B").replace("NO2_B", "NO2_B")


def make_c_name(name: str) -> str:
    clean = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_")
    if not re.match(r"^[A-Za-z_]", clean):
        clean = "x_" + clean
    return clean.lower()


def c_float(x: float) -> str:
    value = float(x)

    if np.isnan(value):
        return "0.0f"
    if np.isposinf(value):
        return "3.4028235e38f"
    if np.isneginf(value):
        return "-3.4028235e38f"

    text = f"{value:.9g}"
    if "." not in text and "e" not in text.lower():
        text = text + ".0"
    return text + "f"


def c_int(x: int) -> str:
    return str(int(x))


def c_float_array(values: Iterable[float]) -> str:
    return ", ".join(c_float(v) for v in values)


def get_estimator_and_scaler(fitted_model: object) -> Tuple[object, Optional[StandardScaler]]:
    scaler: Optional[StandardScaler] = None
    estimator = fitted_model

    if isinstance(fitted_model, Pipeline):
        for _, step_obj in fitted_model.steps:
            if isinstance(step_obj, StandardScaler):
                scaler = step_obj
        estimator = fitted_model.steps[-1][1]

    return estimator, scaler


def sklearn_tree_to_nodes(tree_estimator: object) -> List[Dict[str, object]]:
    tree = tree_estimator.tree_
    value = tree.value.reshape(tree.value.shape[0], -1)[:, 0]

    nodes: List[Dict[str, object]] = []
    for i in range(tree.node_count):
        if tree.children_left[i] == tree.children_right[i]:
            nodes.append({
                "feature": -1,
                "threshold": 0.0,
                "left": -1,
                "right": -1,
                "value": float(value[i]),
            })
        else:
            nodes.append({
                "feature": int(tree.feature[i]),
                "threshold": float(tree.threshold[i]),
                "left": int(tree.children_left[i]),
                "right": int(tree.children_right[i]),
                "value": 0.0,
            })

    return nodes


def parse_xgb_base_score(xgb_model: XGBRegressor) -> float:
    booster = xgb_model.get_booster()
    config = json.loads(booster.save_config())
    raw = config["learner"]["learner_model_param"].get("base_score", "0")

    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]

    return float(raw)


def xgb_feature_index(split_name: str) -> int:
    split = str(split_name)
    if split.startswith("f"):
        return int(split[1:])
    raise ValueError(f"Unexpected XGBoost split name: {split}")


def xgb_json_tree_to_nodes(tree_dict: Dict[str, object]) -> List[Dict[str, object]]:
    nodes: List[Dict[str, object]] = []

    def build(node: Dict[str, object]) -> int:
        idx = len(nodes)
        nodes.append({
            "feature": -1,
            "threshold": 0.0,
            "left": -1,
            "right": -1,
            "value": 0.0,
        })

        if "leaf" in node:
            nodes[idx] = {
                "feature": -1,
                "threshold": 0.0,
                "left": -1,
                "right": -1,
                "value": float(node["leaf"]),
            }
            return idx

        children = {child["nodeid"]: child for child in node["children"]}
        left_idx = build(children[node["yes"]])
        right_idx = build(children[node["no"]])

        nodes[idx] = {
            "feature": xgb_feature_index(node["split"]),
            "threshold": float(node["split_condition"]),
            "left": left_idx,
            "right": right_idx,
            "value": 0.0,
        }
        return idx

    build(tree_dict)
    return nodes


def compact_tree_predict_python(nodes: List[Dict[str, object]], x: np.ndarray, less_than: bool = False) -> float:
    node = 0
    while int(nodes[node]["feature"]) >= 0:
        f = int(nodes[node]["feature"])
        threshold = float(nodes[node]["threshold"])

        if less_than:
            go_left = x[f] < threshold
        else:
            go_left = x[f] <= threshold

        node = int(nodes[node]["left"] if go_left else nodes[node]["right"])

    return float(nodes[node]["value"])


def write_tree_nodes(lines: List[str], array_name: str, nodes: List[Dict[str, object]]) -> None:
    lines.append(f"static const BioTreeNode {array_name}[] = {{")
    for node in nodes:
        lines.append(
            "    {"
            + f"{c_int(node['feature'])}, "
            + f"{c_float(node['threshold'])}, "
            + f"{c_int(node['left'])}, "
            + f"{c_int(node['right'])}, "
            + f"{c_float(node['value'])}"
            + "},"
        )
    lines.append("};")
    lines.append("")


def add_linear_model(lines: List[str], res: TargetResult) -> None:
    estimator, scaler = get_estimator_and_scaler(res.final_model)
    fn = f"predict_{make_c_name(res.target)}"
    prefix = fn
    n_features = len(res.feature_cols)

    mean = scaler.mean_.astype(float) if scaler is not None else np.zeros(n_features, dtype=float)
    scale = scaler.scale_.astype(float) if scaler is not None else np.ones(n_features, dtype=float)
    coef = estimator.coef_.astype(float)
    intercept = float(estimator.intercept_)

    lines.append(f"/* LinearRegression for {res.target} */")
    lines.append(f"#define {prefix.upper()}_N_FEATURES {n_features}")
    lines.append(f"static const float {prefix}_mean[{prefix.upper()}_N_FEATURES] = {{{c_float_array(mean)}}};")
    lines.append(f"static const float {prefix}_scale[{prefix.upper()}_N_FEATURES] = {{{c_float_array(scale)}}};")
    lines.append(f"static const float {prefix}_coef[{prefix.upper()}_N_FEATURES] = {{{c_float_array(coef)}}};")
    lines.append(f"static const float {prefix}_intercept = {c_float(intercept)};")
    lines.append("")
    lines.append(f"float {fn}(const float input_raw[{prefix.upper()}_N_FEATURES]) {{")
    lines.append(f"    float pred = {prefix}_intercept;")
    lines.append(f"    for (int i = 0; i < {prefix.upper()}_N_FEATURES; i++) {{")
    lines.append(f"        float xs = (input_raw[i] - {prefix}_mean[i]) / {prefix}_scale[i];")
    lines.append(f"        pred += {prefix}_coef[i] * xs;")
    lines.append("    }")
    lines.append("    if (pred < 0.0f) pred = 0.0f;")
    lines.append("    return pred;")
    lines.append("}")
    lines.append("")


def add_knn_model(lines: List[str], res: TargetResult) -> None:
    estimator, scaler = get_estimator_and_scaler(res.final_model)
    fn = f"predict_{make_c_name(res.target)}"
    prefix = fn

    if scaler is not None:
        X_model = scaler.transform(res.X_train_final).astype(float)
        mean = scaler.mean_.astype(float)
        scale = scaler.scale_.astype(float)
    else:
        X_model = res.X_train_final.astype(float)
        mean = np.zeros(X_model.shape[1], dtype=float)
        scale = np.ones(X_model.shape[1], dtype=float)

    y_train = res.y_train_final.astype(float)
    k = int(estimator.n_neighbors)
    n_samples, n_features = X_model.shape

    lines.append(f"/* KNN for {res.target} */")
    lines.append(f"#define {prefix.upper()}_N_FEATURES {n_features}")
    lines.append(f"#define {prefix.upper()}_N_SAMPLES {n_samples}")
    lines.append(f"#define {prefix.upper()}_K {k}")
    lines.append(f"static const float {prefix}_mean[{prefix.upper()}_N_FEATURES] = {{{c_float_array(mean)}}};")
    lines.append(f"static const float {prefix}_scale[{prefix.upper()}_N_FEATURES] = {{{c_float_array(scale)}}};")
    lines.append(f"static const float {prefix}_x[{prefix.upper()}_N_SAMPLES][{prefix.upper()}_N_FEATURES] = {{")
    for row in X_model:
        lines.append("    {" + c_float_array(row) + "},")
    lines.append("};")
    lines.append(f"static const float {prefix}_y[{prefix.upper()}_N_SAMPLES] = {{{c_float_array(y_train)}}};")
    lines.append("")
    lines.append(f"float {fn}(const float input_raw[{prefix.upper()}_N_FEATURES]) {{")
    lines.append(f"    float input[{prefix.upper()}_N_FEATURES];")
    lines.append(f"    for (int j = 0; j < {prefix.upper()}_N_FEATURES; j++) {{")
    lines.append(f"        input[j] = (input_raw[j] - {prefix}_mean[j]) / {prefix}_scale[j];")
    lines.append("    }")
    lines.append(f"    float best_dist[{prefix.upper()}_K];")
    lines.append(f"    float best_y[{prefix.upper()}_K];")
    lines.append(f"    for (int k_idx = 0; k_idx < {prefix.upper()}_K; k_idx++) {{")
    lines.append("        best_dist[k_idx] = FLT_MAX;")
    lines.append("        best_y[k_idx] = 0.0f;")
    lines.append("    }")
    lines.append(f"    for (int i = 0; i < {prefix.upper()}_N_SAMPLES; i++) {{")
    lines.append("        float dist = 0.0f;")
    lines.append(f"        for (int j = 0; j < {prefix.upper()}_N_FEATURES; j++) {{")
    lines.append(f"            float diff = input[j] - {prefix}_x[i][j];")
    lines.append("            dist += diff * diff;")
    lines.append("        }")
    lines.append("        int worst_index = 0;")
    lines.append("        float worst_dist = best_dist[0];")
    lines.append(f"        for (int k_idx = 1; k_idx < {prefix.upper()}_K; k_idx++) {{")
    lines.append("            if (best_dist[k_idx] > worst_dist) {")
    lines.append("                worst_dist = best_dist[k_idx];")
    lines.append("                worst_index = k_idx;")
    lines.append("            }")
    lines.append("        }")
    lines.append("        if (dist < worst_dist) {")
    lines.append("            best_dist[worst_index] = dist;")
    lines.append(f"            best_y[worst_index] = {prefix}_y[i];")
    lines.append("        }")
    lines.append("    }")
    lines.append("    float pred = 0.0f;")
    lines.append(f"    for (int k_idx = 0; k_idx < {prefix.upper()}_K; k_idx++) pred += best_y[k_idx];")
    lines.append(f"    pred = pred / (float){prefix.upper()}_K;")
    lines.append("    if (pred < 0.0f) pred = 0.0f;")
    lines.append("    return pred;")
    lines.append("}")
    lines.append("")


def add_decision_tree_model(lines: List[str], res: TargetResult) -> None:
    estimator, scaler = get_estimator_and_scaler(res.final_model)
    if scaler is not None:
        raise ValueError("DecisionTree export does not expect a scaler.")

    fn = f"predict_{make_c_name(res.target)}"
    nodes = sklearn_tree_to_nodes(estimator)

    lines.append(f"/* DecisionTree for {res.target} */")
    write_tree_nodes(lines, f"{fn}_tree", nodes)
    lines.append(f"float {fn}(const float input_raw[BIOFLOC_N_FEATURES]) {{")
    lines.append(f"    float pred = bio_eval_tree_le({fn}_tree, input_raw);")
    lines.append("    if (pred < 0.0f) pred = 0.0f;")
    lines.append("    return pred;")
    lines.append("}")
    lines.append("")


def add_random_forest_model(lines: List[str], res: TargetResult) -> None:
    estimator, scaler = get_estimator_and_scaler(res.final_model)
    if scaler is not None:
        raise ValueError("RandomForest export does not expect a scaler.")

    fn = f"predict_{make_c_name(res.target)}"
    trees = [sklearn_tree_to_nodes(t) for t in estimator.estimators_]

    lines.append(f"/* RandomForest for {res.target} */")
    lines.append(f"#define {fn.upper()}_N_TREES {len(trees)}")
    for i, nodes in enumerate(trees):
        write_tree_nodes(lines, f"{fn}_tree_{i}", nodes)

    lines.append(f"static const BioTreeNode* const {fn}_trees[{fn.upper()}_N_TREES] = {{")
    for i in range(len(trees)):
        lines.append(f"    {fn}_tree_{i},")
    lines.append("};")
    lines.append("")
    lines.append(f"float {fn}(const float input_raw[BIOFLOC_N_FEATURES]) {{")
    lines.append("    float sum = 0.0f;")
    lines.append(f"    for (int i = 0; i < {fn.upper()}_N_TREES; i++) {{")
    lines.append(f"        sum += bio_eval_tree_le({fn}_trees[i], input_raw);")
    lines.append("    }")
    lines.append(f"    float pred = sum / (float){fn.upper()}_N_TREES;")
    lines.append("    if (pred < 0.0f) pred = 0.0f;")
    lines.append("    return pred;")
    lines.append("}")
    lines.append("")


def add_adaboost_model(lines: List[str], res: TargetResult) -> None:
    estimator, scaler = get_estimator_and_scaler(res.final_model)
    if scaler is not None:
        raise ValueError("AdaBoost export does not expect a scaler.")

    fn = f"predict_{make_c_name(res.target)}"
    trees = [sklearn_tree_to_nodes(t) for t in estimator.estimators_]
    weights = estimator.estimator_weights_.astype(float)

    lines.append(f"/* AdaBoostRegressor for {res.target} */")
    lines.append(f"#define {fn.upper()}_N_TREES {len(trees)}")
    for i, nodes in enumerate(trees):
        write_tree_nodes(lines, f"{fn}_tree_{i}", nodes)

    lines.append(f"static const BioTreeNode* const {fn}_trees[{fn.upper()}_N_TREES] = {{")
    for i in range(len(trees)):
        lines.append(f"    {fn}_tree_{i},")
    lines.append("};")
    lines.append(f"static const float {fn}_weights[{fn.upper()}_N_TREES] = {{{c_float_array(weights)}}};")
    lines.append("")
    lines.append(f"float {fn}(const float input_raw[BIOFLOC_N_FEATURES]) {{")
    lines.append(f"    float preds[{fn.upper()}_N_TREES];")
    lines.append(f"    float weights[{fn.upper()}_N_TREES];")
    lines.append(f"    for (int i = 0; i < {fn.upper()}_N_TREES; i++) {{")
    lines.append(f"        preds[i] = bio_eval_tree_le({fn}_trees[i], input_raw);")
    lines.append(f"        weights[i] = {fn}_weights[i];")
    lines.append("    }")
    lines.append(f"    for (int i = 1; i < {fn.upper()}_N_TREES; i++) {{")
    lines.append("        float p = preds[i];")
    lines.append("        float w = weights[i];")
    lines.append("        int j = i - 1;")
    lines.append("        while (j >= 0 && preds[j] > p) {")
    lines.append("            preds[j + 1] = preds[j];")
    lines.append("            weights[j + 1] = weights[j];")
    lines.append("            j--;")
    lines.append("        }")
    lines.append("        preds[j + 1] = p;")
    lines.append("        weights[j + 1] = w;")
    lines.append("    }")
    lines.append("    float total_weight = 0.0f;")
    lines.append(f"    for (int i = 0; i < {fn.upper()}_N_TREES; i++) total_weight += weights[i];")
    lines.append("    float threshold = 0.5f * total_weight;")
    lines.append("    float cumulative = 0.0f;")
    lines.append("    float pred = preds[0];")
    lines.append(f"    for (int i = 0; i < {fn.upper()}_N_TREES; i++) {{")
    lines.append("        cumulative += weights[i];")
    lines.append("        if (cumulative >= threshold) {")
    lines.append("            pred = preds[i];")
    lines.append("            break;")
    lines.append("        }")
    lines.append("    }")
    lines.append("    if (pred < 0.0f) pred = 0.0f;")
    lines.append("    return pred;")
    lines.append("}")
    lines.append("")


def add_xgboost_model(lines: List[str], res: TargetResult) -> None:
    estimator, scaler = get_estimator_and_scaler(res.final_model)
    if scaler is not None:
        raise ValueError("XGBoost export does not expect a scaler.")

    fn = f"predict_{make_c_name(res.target)}"
    booster = estimator.get_booster()
    dumps = booster.get_dump(dump_format="json")
    trees = [xgb_json_tree_to_nodes(json.loads(dump)) for dump in dumps]
    base_score = parse_xgb_base_score(estimator)

    X = res.X_train_final
    py_pred = estimator.predict(X[:min(20, len(X))])
    compact_base: List[float] = []
    compact_zero: List[float] = []

    for x in X[:min(20, len(X))]:
        s_base = base_score
        s_zero = 0.0
        for nodes in trees:
            leaf = compact_tree_predict_python(nodes, x, less_than=True)
            s_base += leaf
            s_zero += leaf
        compact_base.append(s_base)
        compact_zero.append(s_zero)

    err_base = np.mean(np.abs(np.array(compact_base) - py_pred))
    err_zero = np.mean(np.abs(np.array(compact_zero) - py_pred))
    used_base = 0.0 if err_zero < err_base else base_score

    lines.append(f"/* XGBoost for {res.target} */")
    lines.append(f"#define {fn.upper()}_N_TREES {len(trees)}")
    lines.append(f"static const float {fn}_base_score = {c_float(used_base)};")
    lines.append("")
    for i, nodes in enumerate(trees):
        write_tree_nodes(lines, f"{fn}_tree_{i}", nodes)

    lines.append(f"static const BioTreeNode* const {fn}_trees[{fn.upper()}_N_TREES] = {{")
    for i in range(len(trees)):
        lines.append(f"    {fn}_tree_{i},")
    lines.append("};")
    lines.append("")
    lines.append(f"float {fn}(const float input_raw[BIOFLOC_N_FEATURES]) {{")
    lines.append(f"    float pred = {fn}_base_score;")
    lines.append(f"    for (int i = 0; i < {fn.upper()}_N_TREES; i++) {{")
    lines.append(f"        pred += bio_eval_tree_lt({fn}_trees[i], input_raw);")
    lines.append("    }")
    lines.append("    if (pred < 0.0f) pred = 0.0f;")
    lines.append("    return pred;")
    lines.append("}")
    lines.append("")


def add_header_preamble(lines: List[str]) -> None:
    lines.extend([
        "#ifndef BIOFLOC_COMPACT_MODELS_H",
        "#define BIOFLOC_COMPACT_MODELS_H",
        "",
        "#include <float.h>",
        "#include <stdint.h>",
        "",
        "/*",
        "Compact array-based embedded models.",
        "Generated by model_export/export_biofloc_models_to_c.py.",
        "",
        "Input order for all prediction functions:",
        "input_raw[0] = DO_B(mg/L)",
        "input_raw[1] = ORP_B(mV)",
        "input_raw[2] = EC_B(mS/cm)",
        "input_raw[3] = pH_B",
        "input_raw[4] = Temp_B(C)",
        "input_raw[5] = Turbidty_B(NTU)",
        "",
        "Targets included:",
        "TAN_B(mg/L)",
        "TOC_B(mg/L)",
        "DOC_B(mg/L)",
        "TN_B(mg/L)",
        "DN_B(mg/L)",
        "",
        "NO3_B and TSS are not included as output targets.",
        "*/",
        "",
        "#define BIOFLOC_N_FEATURES 6",
        "",
        "typedef struct {",
        "    int16_t feature;",
        "    float threshold;",
        "    int16_t left;",
        "    int16_t right;",
        "    float value;",
        "} BioTreeNode;",
        "",
        "static float bio_eval_tree_le(const BioTreeNode *nodes, const float *input) {",
        "    int16_t node = 0;",
        "    while (nodes[node].feature >= 0) {",
        "        int16_t f = nodes[node].feature;",
        "        if (input[f] <= nodes[node].threshold) {",
        "            node = nodes[node].left;",
        "        } else {",
        "            node = nodes[node].right;",
        "        }",
        "    }",
        "    return nodes[node].value;",
        "}",
        "",
        "static float bio_eval_tree_lt(const BioTreeNode *nodes, const float *input) {",
        "    int16_t node = 0;",
        "    while (nodes[node].feature >= 0) {",
        "        int16_t f = nodes[node].feature;",
        "        if (input[f] < nodes[node].threshold) {",
        "            node = nodes[node].left;",
        "        } else {",
        "            node = nodes[node].right;",
        "        }",
        "    }",
        "    return nodes[node].value;",
        "}",
        "",
    ])


def create_combined_header(results: List[TargetResult], output_dir: Path, header_name: str) -> Path:
    header_path = output_dir / header_name
    lines: List[str] = []
    add_header_preamble(lines)

    for res in results:
        model_name = res.best_model_name
        lines.extend([
            "",
            "/* ============================================================",
            f"   Target: {res.target}",
            f"   Selected model: {model_name}",
            "   ============================================================ */",
            "",
        ])

        if model_name == "LinearRegression":
            add_linear_model(lines, res)
        elif model_name in {"KNN_k1", "KNN_k3"}:
            add_knn_model(lines, res)
        elif model_name == "DecisionTree":
            add_decision_tree_model(lines, res)
        elif model_name == "RandomForest":
            add_random_forest_model(lines, res)
        elif model_name == "AdaBoost":
            add_adaboost_model(lines, res)
        elif model_name == "XGBoost":
            add_xgboost_model(lines, res)
        else:
            raise ValueError(f"No compact exporter for model: {model_name}")

    lines.extend(["", "#endif", ""])
    header_path.write_text("\n".join(lines), encoding="utf-8")
    return header_path


def repair_invalid_float_literals(header_path: Path) -> None:
    text = header_path.read_text(encoding="utf-8")
    fixed = re.sub(r"(?<![A-Za-z0-9_.+\-])(-?\d+)f\b", r"\1.0f", text)
    header_path.write_text(fixed, encoding="utf-8")


def validate_header(header_path: Path) -> None:
    repair_invalid_float_literals(header_path)
    text = header_path.read_text(encoding="utf-8")

    n_ifndef = text.count("#ifndef")
    n_endif = text.count("#endif")
    bad_predict_guards = re.findall(r"#ifndef\s+PREDICT_[A-Z0-9_]+_H", text)
    bad_float_literals = re.findall(r"(?<![A-Za-z0-9_.+\-])(-?\d+)f\b", text)

    print("\nHeader validation:")
    print(f"Header path: {header_path}")
    print(f"Number of #ifndef: {n_ifndef}")
    print(f"Number of #endif: {n_endif}")
    print(f"Bad nested PREDICT guards: {len(bad_predict_guards)}")
    print(f"Bad float literals like 0f: {len(bad_float_literals)}")

    if n_ifndef != 1 or n_endif != 1:
        raise ValueError(f"Header guard problem. #ifndef={n_ifndef}, #endif={n_endif}")
    if bad_predict_guards:
        raise ValueError("Old nested PREDICT header guards found.")
    if bad_float_literals:
        raise ValueError("Invalid float literals found.")

    size_mb = header_path.stat().st_size / (1024 * 1024)
    print(f"Header size: {size_mb:.2f} MB")
    print(f"Header lines: {len(text.splitlines())}")
    if size_mb > 5:
        print("Warning: header is large. Use an ESP32 partition scheme with enough app space.")
    print("Header validation passed.")


def select_best_model_by_r2(cv_metrics: Dict[str, Dict[str, object]]) -> str:
    valid_models = {
        name: vals
        for name, vals in cv_metrics.items()
        if name in EXPORTABLE_MODELS
        and not bool(vals["failed"])
        and np.isfinite(float(vals["r2"]))
    }

    if not valid_models:
        raise ValueError("No valid exportable model was available.")

    return max(valid_models, key=lambda name: float(valid_models[name]["r2"]))


def run_pipeline_target(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    n_splits: int = 10,
    random_state: int = 42,
) -> TargetResult:
    sub_df = df[feature_cols + [target_col]].dropna()

    if sub_df.empty:
        raise ValueError(f"No valid rows found for target: {target_col}")

    X = sub_df[feature_cols].values.astype(float)
    y = sub_df[target_col].values.astype(float)

    if len(y) < 2:
        raise ValueError(f"Not enough samples for target: {target_col}")

    models = get_candidate_models(random_state=random_state)
    cv_metrics = cv_metrics_for_models(
        models,
        X,
        y,
        n_splits=n_splits,
        random_state=random_state,
    )

    formatted_target = format_target_name_for_print(target_col)
    print("\n" + "=" * 70)
    print(f"Target: {formatted_target}")
    print(f"Valid samples: {len(y)}")
    print("=" * 70)
    print(f"\n{n_splits}-fold CV metrics per model for {formatted_target}:")
    print("  Model            |  MAE   |  RMSE  |   R2")
    print("  " + "-" * 52)

    for name, metrics in sorted(cv_metrics.items(), key=lambda item: float(item[1]["r2"]), reverse=True):
        if metrics["failed"]:
            print(f"  {name:16s} |   NA   |   NA   |   NA")
            print(f"      Error: {metrics['error']}")
        else:
            print(
                f"  {name:16s} | "
                f"{float(metrics['mae']):6.3f} | "
                f"{float(metrics['rmse']):6.3f} | "
                f"{float(metrics['r2']):6.3f}"
            )

    best_name = select_best_model_by_r2(cv_metrics)
    best_model = models[best_name]
    print(f"\nBest exportable model for {formatted_target}: {best_name}")

    mae_cv, rmse_cv, r2_cv, y_true_cv, y_pred_cv = cv_predictions_for_model(
        best_model,
        X,
        y,
        n_splits=n_splits,
        random_state=random_state,
    )

    print(f"\nSelected model {n_splits}-fold CV for {formatted_target}:")
    print(f"  MAE  = {mae_cv:.3f}")
    print(f"  RMSE = {rmse_cv:.3f}")
    print(f"  R2   = {r2_cv:.3f}")

    mae_loo, rmse_loo, r2_loo, y_true_loo, y_pred_loo = loo_evaluation(best_model, X, y)
    print(f"\nLOO performance for {formatted_target}:")
    print(f"  MAE  = {mae_loo:.3f}")
    print(f"  RMSE = {rmse_loo:.3f}")
    print(f"  R2   = {r2_loo:.3f}")

    final_model = clone(best_model)
    final_model.fit(X, y)

    return TargetResult(
        target=target_col,
        n_samples=len(y),
        best_model_name=best_name,
        cv_metrics=cv_metrics,
        cv_best_mae=float(mae_cv),
        cv_best_rmse=float(rmse_cv),
        cv_best_r2=float(r2_cv),
        y_true_cv=y_true_cv,
        y_pred_cv=y_pred_cv,
        loo_mae=float(mae_loo),
        loo_rmse=float(rmse_loo),
        loo_r2=float(r2_loo),
        y_true_loo=y_true_loo,
        y_pred_loo=y_pred_loo,
        final_model=final_model,
        feature_cols=feature_cols,
        X_train_final=X,
        y_train_final=y,
    )


def write_summary_outputs(results: List[TargetResult], output_dir: Path) -> None:
    summary_rows = []
    for res in results:
        summary_rows.append({
            "target": res.target,
            "n_samples": res.n_samples,
            "selected_model": res.best_model_name,
            "cv_mae": res.cv_best_mae,
            "cv_rmse": res.cv_best_rmse,
            "cv_r2": res.cv_best_r2,
            "loo_mae": res.loo_mae,
            "loo_rmse": res.loo_rmse,
            "loo_r2": res.loo_r2,
            "c_function": f"predict_{make_c_name(res.target)}",
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "biofloc_compact_model_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\nModel summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSummary CSV saved to: {summary_path}")


def write_test_predictions(results: List[TargetResult], bio_df: pd.DataFrame, output_dir: Path) -> None:
    test_df = bio_df[FEATURES_BIO].dropna()
    if test_df.empty:
        print("\nNo complete test input row found for feature columns.")
        return

    test_input = test_df.iloc[0].values.astype(float)
    print("\nTest input used for Python prediction check:")
    for name, value in zip(FEATURES_BIO, test_input):
        print(f"  {name}: {value}")

    test_rows = []
    for res in results:
        pred = float(res.final_model.predict(test_input.reshape(1, -1))[0])
        if pred < 0.0:
            pred = 0.0
        test_rows.append({
            "target": res.target,
            "selected_model": res.best_model_name,
            "c_function": f"predict_{make_c_name(res.target)}",
            "python_prediction_from_final_model": pred,
        })

    test_pred_df = pd.DataFrame(test_rows)
    test_pred_path = output_dir / "python_final_model_test_predictions.csv"
    test_pred_df.to_csv(test_pred_path, index=False)
    print("\nPython final-model predictions for one test input:")
    print(test_pred_df.to_string(index=False))
    print(f"\nTest prediction CSV saved to: {test_pred_path}")


def prepare_output_dir(output_dir: Path, clear_old_exports: bool) -> None:
    if clear_old_exports and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export biofloc soft-sensor models to one compact C/C++ header."
    )
    parser.add_argument(
        "--input-excel",
        required=True,
        help="Path to the Excel workbook containing the biofloc data sheet.",
    )
    parser.add_argument(
        "--sheet-name",
        default="Biofloc tank",
        help="Excel sheet name to load. Default: Biofloc tank.",
    )
    parser.add_argument(
        "--output-dir",
        default="biofloc_c_models",
        help="Output directory for the generated header and CSV summaries.",
    )
    parser.add_argument(
        "--header-name",
        default="biofloc_compact_models.h",
        help="Name of the generated combined header file.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=10,
        help="Number of CV splits. Default: 10.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for cross-validation and model fitting. Default: 42.",
    )
    parser.add_argument(
        "--keep-old-exports",
        action="store_true",
        help="Do not delete the output directory before exporting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_excel = Path(args.input_excel)
    output_dir = Path(args.output_dir)

    if not input_excel.exists():
        raise FileNotFoundError(f"Input Excel file not found: {input_excel}")

    prepare_output_dir(output_dir, clear_old_exports=not args.keep_old_exports)

    print("Clean export folder:")
    print(output_dir)

    bio_df = pd.read_excel(input_excel, sheet_name=args.sheet_name)
    print("\nLoaded data:")
    print(bio_df.shape)

    missing_features = [col for col in FEATURES_BIO if col not in bio_df.columns]
    missing_targets = [col for col in TARGETS_BIO if col not in bio_df.columns]

    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    results: List[TargetResult] = []
    for target in TARGETS_BIO:
        result = run_pipeline_target(
            bio_df,
            FEATURES_BIO,
            target,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        results.append(result)

    if not results:
        raise ValueError("No results were generated. Check targets and input data.")

    header_path = create_combined_header(results, output_dir, args.header_name)
    print("\nCompact combined deployment header created:")
    print(header_path)

    validate_header(header_path)
    write_summary_outputs(results, output_dir)
    write_test_predictions(results, bio_df, output_dir)

    print("\nExport complete.")
    print(f"Generated files are in: {output_dir}")
    print(f"Use this single file for Arduino deployment: {header_path}")
    print("\nArduino include line:")
    print(f'#include "{args.header_name}"')
    print("\nArduino input must be float:")
    print("float input_raw[6];")
    print("\nArduino function names:")
    for res in results:
        print(f"  {res.target} -> predict_{make_c_name(res.target)}()")


if __name__ == "__main__":
    main()
