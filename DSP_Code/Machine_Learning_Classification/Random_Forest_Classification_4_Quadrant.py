import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis
from scipy.signal import detrend

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)


# ============================================================
# User settings
# ============================================================

DATA_DIR = Path(r"EIP_4quadrant_dataset")

FS = 20.0
TRANSIENT_SECONDS = 5.0
WINDOW_SECONDS = 10.0
STEP_SECONDS = 5.0

ACTIVE_VOLUME_FOR_CLASSIFICATION = "medium"
SYMMETRIC_VOLUME_FOR_CLASSIFICATION = "medium"

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 20

VOLUME_MAP = {
    "small": 33.3,
    "medium": 44.4,
    "large": 66.7,
    "NA": 44.4,
    "na": 44.4
}

CLASS_ORDER = [
    "symmetric",
    "Q1_active",
    "Q2_active",
    "Q3_active",
    "Q4_active",
    "left_weak",
    "right_weak",
    "upper_weak",
    "lower_weak"
]


# ============================================================
# Filename parsing
# ============================================================

def parse_filename(filename: str):
    name = Path(filename).stem

    m_active = re.match(
        r"(Q[1-4]_active)_(small|medium|large|NA|na)_(?:(rocking|static|rocking_IMU_ANC|rocking_noANC)_)?trial(\d+)_",
        name
    )
    if m_active:
        return m_active.group(1), m_active.group(2), m_active.group(3) or "static", m_active.group(4)

    m_weak = re.match(
        r"(left_weak|right_weak|upper_weak|lower_weak)_(small|medium|large|NA|na)_(?:(rocking|static|rocking_IMU_ANC|rocking_noANC)_)?trial(\d+)_",
        name
    )
    if m_weak:
        return m_weak.group(1), m_weak.group(2), m_weak.group(3) or "static", m_weak.group(4)

    m_sym = re.match(
        r"(symmetric)_(small|medium|large|NA|na)_(?:(rocking|static|rocking_IMU_ANC|rocking_noANC)_)?trial(\d+)_",
        name
    )
    if m_sym:
        return m_sym.group(1), m_sym.group(2), m_sym.group(3) or "static", m_sym.group(4)

    return None, None, None, None


def include_file_for_main_classification(case_name, volume_level, condition):
    if condition not in ["static", "rocking"]:
        return False

    if case_name == "symmetric":
        return volume_level.lower() == SYMMETRIC_VOLUME_FOR_CLASSIFICATION.lower()

    if case_name in ["Q1_active", "Q2_active", "Q3_active", "Q4_active"]:
        return volume_level.lower() == ACTIVE_VOLUME_FOR_CLASSIFICATION.lower()

    if case_name in ["left_weak", "right_weak", "upper_weak", "lower_weak"]:
        return True

    return False


# ============================================================
# Ground-truth quadrant volume and asymmetry index
# ============================================================

def get_quadrant_volumes(case_name: str, volume_level: str):
    """
    Q1 = upper-right, Q2 = upper-left, Q3 = lower-left, Q4 = lower-right.

    Weak-region cases use all four balloons:
    weak region = small-volume excitation, reference region = medium-volume excitation.
    """
    V = VOLUME_MAP.get(volume_level, 44.4)
    S = VOLUME_MAP["small"]
    M = VOLUME_MAP["medium"]

    q = np.zeros(4, dtype=float)

    if case_name == "symmetric":
        q[:] = M
    elif case_name == "Q1_active":
        q[0] = V
    elif case_name == "Q2_active":
        q[1] = V
    elif case_name == "Q3_active":
        q[2] = V
    elif case_name == "Q4_active":
        q[3] = V
    elif case_name == "left_weak":
        q[0] = M
        q[1] = S
        q[2] = S
        q[3] = M
    elif case_name == "right_weak":
        q[0] = S
        q[1] = M
        q[2] = M
        q[3] = S
    elif case_name == "upper_weak":
        q[0] = S
        q[1] = S
        q[2] = M
        q[3] = M
    elif case_name == "lower_weak":
        q[0] = M
        q[1] = M
        q[2] = S
        q[3] = S
    else:
        raise ValueError(f"Unknown case name: {case_name}")

    return q


def compute_asymmetry_indices(q):
    Q1, Q2, Q3, Q4 = q
    total = Q1 + Q2 + Q3 + Q4

    if total == 0:
        return np.nan, np.nan

    left = Q2 + Q3
    right = Q1 + Q4
    upper = Q1 + Q2
    lower = Q3 + Q4

    AI_LR = (left - right) / total
    AI_UL = (upper - lower) / total

    return AI_LR, AI_UL


# ============================================================
# Signal loading and feature extraction
# ============================================================

def load_csv_signal(csv_path: Path):
    df = pd.read_csv(
        csv_path,
        usecols=lambda c: c in ["time_s", "filtered_adc"]
    )

    if "filtered_adc" not in df.columns:
        raise ValueError(f"{csv_path.name} does not contain filtered_adc column.")

    y = df["filtered_adc"].to_numpy(dtype=float)
    y = y[~np.isnan(y)]

    return y


def dominant_frequency_features(y, fs):
    y = np.asarray(y, dtype=float)

    if len(y) < 4:
        return np.nan, np.nan

    y_d = detrend(y)
    n = len(y_d)

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals = np.fft.rfft(y_d)
    power = np.abs(fft_vals) ** 2

    band = (freqs >= 0.1) & (freqs <= 2.0)
    if not np.any(band):
        return np.nan, np.nan

    freqs_band = freqs[band]
    power_band = power[band]

    idx = np.argmax(power_band)
    dom_freq = freqs_band[idx]

    total_power = np.sum(power_band)
    dom_power = power_band[idx] / total_power if total_power > 0 else np.nan

    return dom_freq, dom_power


def extract_features_from_window(y, fs=20.0):
    y = np.asarray(y, dtype=float)

    mean_adc = np.mean(y)
    median_adc = np.median(y)

    min_adc = np.min(y)
    max_adc = np.max(y)

    p05_adc = np.percentile(y, 5)
    p95_adc = np.percentile(y, 95)

    peak_to_peak = max_adc - min_adc
    robust_peak_to_peak = p95_adc - p05_adc

    rms_adc = np.sqrt(np.mean(y ** 2))
    std_adc = np.std(y)
    iqr_adc = np.percentile(y, 75) - np.percentile(y, 25)

    skew_adc = skew(y, bias=False)
    kurtosis_adc = kurtosis(y, bias=False)

    x = np.arange(len(y)) / fs
    slope_adc = np.polyfit(x, y, 1)[0]

    dom_freq_hz, dom_freq_power = dominant_frequency_features(y, fs)

    n = len(y)

    idx_min = int(np.argmin(y))
    idx_max = int(np.argmax(y))

    idx_min_norm = idx_min / max(n - 1, 1)
    idx_max_norm = idx_max / max(n - 1, 1)

    time_min_s = idx_min / fs
    time_max_s = idx_max / fs

    if idx_max >= idx_min:
        rise_time_s = (idx_max - idx_min) / fs
        fall_time_s = (n - 1 - idx_max + idx_min) / fs
    else:
        rise_time_s = (n - 1 - idx_min + idx_max) / fs
        fall_time_s = (idx_min - idx_max) / fs

    first_half = y[: n // 2]
    second_half = y[n // 2:]

    mean_first_half = np.mean(first_half)
    mean_second_half = np.mean(second_half)
    first_second_diff = mean_first_half - mean_second_half

    y_z = y - np.mean(y)
    y_std = np.std(y_z) + 1e-9
    y_z = y_z / y_std

    theta = 2 * np.pi * np.arange(n) / n
    sine_ref = np.sin(theta)
    cos_ref = np.cos(theta)

    corr_with_sine = float(np.mean(y_z * sine_ref))
    corr_with_cosine = float(np.mean(y_z * cos_ref))

    q1 = y[: n // 4]
    q2 = y[n // 4: n // 2]
    q3 = y[n // 2: 3 * n // 4]
    q4 = y[3 * n // 4:]

    qmean_1 = np.mean(q1)
    qmean_2 = np.mean(q2)
    qmean_3 = np.mean(q3)
    qmean_4 = np.mean(q4)

    qdiff_12 = qmean_1 - qmean_2
    qdiff_23 = qmean_2 - qmean_3
    qdiff_34 = qmean_3 - qmean_4
    qdiff_41 = qmean_4 - qmean_1

    return {
        "mean_adc": mean_adc,
        "median_adc": median_adc,
        "min_adc": min_adc,
        "max_adc": max_adc,
        "p05_adc": p05_adc,
        "p95_adc": p95_adc,
        "peak_to_peak": peak_to_peak,
        "robust_peak_to_peak": robust_peak_to_peak,
        "rms_adc": rms_adc,
        "std_adc": std_adc,
        "iqr_adc": iqr_adc,
        "skew_adc": skew_adc,
        "kurtosis_adc": kurtosis_adc,
        "slope_adc": slope_adc,
        "dom_freq_hz": dom_freq_hz,
        "dom_freq_power": dom_freq_power,

        "idx_min_norm": idx_min_norm,
        "idx_max_norm": idx_max_norm,
        "time_min_s": time_min_s,
        "time_max_s": time_max_s,
        "rise_time_s": rise_time_s,
        "fall_time_s": fall_time_s,
        "mean_first_half": mean_first_half,
        "mean_second_half": mean_second_half,
        "first_second_diff": first_second_diff,
        "corr_with_sine": corr_with_sine,
        "corr_with_cosine": corr_with_cosine,

        "qmean_1": qmean_1,
        "qmean_2": qmean_2,
        "qmean_3": qmean_3,
        "qmean_4": qmean_4,
        "qdiff_12": qdiff_12,
        "qdiff_23": qdiff_23,
        "qdiff_34": qdiff_34,
        "qdiff_41": qdiff_41,
    }


def build_feature_dataset(data_dir: Path):
    rows = []
    csv_files = sorted(data_dir.glob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    win_len = int(WINDOW_SECONDS * FS)
    step_len = int(STEP_SECONDS * FS)
    transient_len = int(TRANSIENT_SECONDS * FS)

    print(f"\nWindow length samples: {win_len}")
    print(f"Step length samples  : {step_len}")

    if step_len < win_len:
        overlap = 100 * (1 - step_len / win_len)
        print(f"Window overlap       : {overlap:.1f}%")
    else:
        print("Window overlap       : 0%")

    for csv_path in csv_files:
        case_name, volume_level, condition, trial_id = parse_filename(csv_path.name)

        if case_name is None:
            print(f"[Skip] Filename not recognised: {csv_path.name}")
            continue

        if not include_file_for_main_classification(case_name, volume_level, condition):
            print(f"[Skip] Not used in main classification: {csv_path.name}")
            continue

        y_full = load_csv_signal(csv_path)
        n_full = len(y_full)

        if n_full <= transient_len + win_len:
            print(f"[Skip] {csv_path.name}: too short after transient removal, N = {n_full}")
            continue

        y = y_full[transient_len:]
        n = len(y)

        q = get_quadrant_volumes(case_name, volume_level)
        AI_LR, AI_UL = compute_asymmetry_indices(q)

        split_key = f"{condition}_trial{str(trial_id).zfill(2)}"
        window_idx = 0

        for start in range(0, n - win_len + 1, step_len):
            end = start + win_len
            y_win = y[start:end]

            feat = extract_features_from_window(y_win, FS)

            row = {
                "file": csv_path.name,
                "case": case_name,
                "volume_level": volume_level,
                "condition": condition,
                "trial_id": str(trial_id).zfill(2),
                "split_key": split_key,
                "window_idx_in_file": window_idx,
                "window_start_s": TRANSIENT_SECONDS + start / FS,
                "window_end_s": TRANSIENT_SECONDS + end / FS,

                "Q1_mL": q[0],
                "Q2_mL": q[1],
                "Q3_mL": q[2],
                "Q4_mL": q[3],
                "AI_LR": AI_LR,
                "AI_UL": AI_UL,
            }
            row.update(feat)
            rows.append(row)

            window_idx += 1

    feature_df = pd.DataFrame(rows)

    if feature_df.empty:
        raise RuntimeError("No valid feature rows were generated.")

    feature_df["case"] = pd.Categorical(
        feature_df["case"],
        categories=CLASS_ORDER,
        ordered=True
    )

    feature_df = feature_df.sort_values(
        ["case", "condition", "trial_id", "volume_level", "file", "window_start_s"]
    ).reset_index(drop=True)

    return feature_df


def get_feature_columns():
    return [
        "mean_adc",
        "median_adc",
        "min_adc",
        "max_adc",
        "p05_adc",
        "p95_adc",
        "peak_to_peak",
        "robust_peak_to_peak",
        "rms_adc",
        "std_adc",
        "iqr_adc",
        "skew_adc",
        "kurtosis_adc",
        "slope_adc",
        "dom_freq_hz",
        "dom_freq_power",

        "idx_min_norm",
        "idx_max_norm",
        "time_min_s",
        "time_max_s",
        "rise_time_s",
        "fall_time_s",
        "mean_first_half",
        "mean_second_half",
        "first_second_diff",
        "corr_with_sine",
        "corr_with_cosine",

        "qmean_1",
        "qmean_2",
        "qmean_3",
        "qmean_4",
        "qdiff_12",
        "qdiff_23",
        "qdiff_34",
        "qdiff_41",
    ]


# ============================================================
# Model
# ============================================================

def make_classifier(random_state):
    return RandomForestClassifier(
        n_estimators=140,
        max_depth=5,
        min_samples_leaf=8,
        max_features="sqrt",
        random_state=random_state,
        class_weight="balanced_subsample"
    )


# ============================================================
# Repeated stratified CV
# ============================================================

def repeated_cv_evaluation(feature_df):
    feature_cols = get_feature_columns()

    X = feature_df[feature_cols].to_numpy()
    y = feature_df["case"].astype(str).to_numpy()

    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE
    )

    cm_total = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=float)

    all_y_true = []
    all_y_pred = []

    fold_rows = []
    fold_idx = 0

    for train_idx, test_idx in rskf.split(X, y):
        fold_idx += 1

        X_train = X[train_idx]
        y_train = y[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]

        clf = make_classifier(RANDOM_STATE + fold_idx)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
        cm_total += cm

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(
            y_test,
            y_pred,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0
        )

        fold_rows.append({
            "fold_index": fold_idx,
            "accuracy": acc,
            "macro_f1": macro_f1
        })

        all_y_true.extend(list(y_test))
        all_y_pred.extend(list(y_pred))

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv("repeated_cv_fold_metrics_RF_medium_strength.csv", index=False)

    print("\n========== Repeated stratified CV ==========")
    print(f"n_splits  = {N_SPLITS}")
    print(f"n_repeats = {N_REPEATS}")
    print(f"Total evaluations = {N_SPLITS * N_REPEATS}")
    print(f"Mean accuracy = {fold_df['accuracy'].mean():.3f} ± {fold_df['accuracy'].std():.3f}")
    print(f"Mean macro-F1 = {fold_df['macro_f1'].mean():.3f} ± {fold_df['macro_f1'].std():.3f}")
    print("Saved fold metrics to: repeated_cv_fold_metrics_RF_medium_strength.csv")
    print("============================================\n")

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    report = classification_report(
        all_y_true,
        all_y_pred,
        labels=CLASS_ORDER,
        zero_division=0
    )
    print("\nPooled repeated-CV classification report:")
    print(report)

    report_dict = classification_report(
        all_y_true,
        all_y_pred,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv("classification_report_RF_medium_strength_repeated_cv.csv")

    final_clf = make_classifier(RANDOM_STATE)
    final_clf.fit(X, y)

    feature_df.to_csv("EIP_feature_dataset_used_RF_medium_strength_repeated_cv.csv", index=False)

    return final_clf, feature_cols, cm_total, all_y_true, all_y_pred, fold_df


# ============================================================
# Plotting
# ============================================================

def plot_normalised_confusion_matrix_from_cm(
    cm_total,
    output_path="confusion_matrix_RF_medium_strength_repeated_cv_normalised_percent.png"
):
    labels = CLASS_ORDER

    row_sum = cm_total.sum(axis=1, keepdims=True)

    cm_norm = np.divide(
        cm_total,
        row_sum,
        out=np.zeros_like(cm_total, dtype=float),
        where=row_sum != 0
    ) * 100.0

    fig, ax = plt.subplots(figsize=(11, 8.5))

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Percentage (%)")

    ax.set_title("Normalised confusion matrix for main spatial-condition classification", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            value = cm_norm[i, j]
            text_color = "white" if value > 50 else "black"
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color=text_color, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)

    cm_norm_df = pd.DataFrame(cm_norm, index=labels, columns=labels)
    cm_norm_df.to_csv("confusion_matrix_RF_medium_strength_repeated_cv_normalised_percent.csv")

    cm_count_df = pd.DataFrame(cm_total, index=labels, columns=labels)
    cm_count_df.to_csv("confusion_matrix_RF_medium_strength_repeated_cv_counts.csv")

    print(f"Saved normalised confusion matrix to: {output_path}")


def plot_random_forest_feature_importance(
    clf,
    feature_cols,
    output_path="random_forest_feature_importance_RF_medium_strength_repeated_cv.png"
):
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]

    importance_df = pd.DataFrame({
        "feature": np.array(feature_cols)[order],
        "importance": importances[order]
    })

    importance_df.to_csv("random_forest_feature_importance_RF_medium_strength_repeated_cv.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(importance_df["feature"], importance_df["importance"])
    ax.set_ylabel("Feature importance")
    ax.set_title("Random Forest feature importance")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"Saved feature importance figure to: {output_path}")


def select_top_features_for_fig_3_15(
    feature_df,
    clf,
    feature_cols,
    top_n=5,
    corr_threshold=0.92
):
    """
    Updated selection rule for Fig. 3.15:
    - Plot only five features.
    - Force A_min to replace A_RMS because both are magnitude/offset-related,
      but A_min is more interpretable for spatial encoding.
    - Exclude A_RMS and f_dom.
    - Select remaining features from RF importance ranking with high-correlation
      redundancy removal.
    """

    importances = pd.Series(
        clf.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    numeric_df = feature_df[feature_cols].copy()
    numeric_df = numeric_df.apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))

    selected = ["std_adc", "min_adc"]

    excluded = {
        "mean_adc",
        "rms_adc",
        "dom_freq_hz",
        "std_adc",
        "min_adc"
    }

    for feat in importances.index:
        if feat in excluded:
            continue

        if numeric_df[feat].std() == 0:
            continue

        is_redundant = False
        for chosen in selected:
            corr = numeric_df[[feat, chosen]].corr().iloc[0, 1]
            if np.isfinite(corr) and abs(corr) >= corr_threshold:
                is_redundant = True
                break

        if not is_redundant:
            selected.append(feat)

        if len(selected) >= top_n:
            break

    selected_importance_df = pd.DataFrame({
        "feature": selected,
        "importance": [importances[f] for f in selected]
    })

    selected_importance_df.to_csv("selected_features_for_Fig_3_15.csv", index=False)

    print("\nSelected features for Fig. 3.15:")
    print("Five features only; A_RMS is replaced by A_min; f_dom is excluded.")
    print(selected_importance_df.to_string(index=False))

    return selected


def plot_feature_grouped_bar_chart(
    feature_df,
    clf,
    feature_cols,
    output_path="Fig_3_15_top_discriminative_features.png",
    top_n=5
):
    selected_features = select_top_features_for_fig_3_15(
        feature_df=feature_df,
        clf=clf,
        feature_cols=feature_cols,
        top_n=top_n,
        corr_threshold=0.92
    )

    feature_label_map = {
        "std_adc": r"$\sigma_{\mathrm{ADC}}$",
        "robust_peak_to_peak": r"$A_{\mathrm{pp,rob}}$",
        "iqr_adc": r"$\mathrm{IQR}_{\mathrm{ADC}}$",
        "peak_to_peak": r"$A_{\mathrm{pp}}$",
        "rms_adc": r"$A_{\mathrm{RMS}}$",
        "min_adc": r"$A_{\min}$",
        "max_adc": r"$A_{\max}$",
        "median_adc": r"$A_{\mathrm{median}}$",
        "p05_adc": r"$P_{5,\mathrm{ADC}}$",
        "p95_adc": r"$P_{95,\mathrm{ADC}}$",
        "mean_first_half": r"$\bar{A}_{\mathrm{H1}}$",
        "mean_second_half": r"$\bar{A}_{\mathrm{H2}}$",
        "first_second_diff": r"$\Delta \bar{A}_{\mathrm{H1-H2}}$",
        "qmean_1": r"$\bar{A}_{\mathrm{QW1}}$",
        "qmean_2": r"$\bar{A}_{\mathrm{QW2}}$",
        "qmean_3": r"$\bar{A}_{\mathrm{QW3}}$",
        "qmean_4": r"$\bar{A}_{\mathrm{QW4}}$",
        "qdiff_12": r"$\Delta\bar{A}_{12}$",
        "qdiff_23": r"$\Delta\bar{A}_{23}$",
        "qdiff_34": r"$\Delta\bar{A}_{34}$",
        "qdiff_41": r"$\Delta\bar{A}_{41}$",
        "idx_min_norm": r"$t_{\min}/T$",
        "idx_max_norm": r"$t_{\max}/T$",
        "time_min_s": r"$t_{\min}$",
        "time_max_s": r"$t_{\max}$",
        "rise_time_s": r"$t_{\mathrm{rise}}$",
        "fall_time_s": r"$t_{\mathrm{fall}}$",
        "corr_with_sine": r"$r_{\sin}$",
        "corr_with_cosine": r"$r_{\cos}$",
        "dom_freq_power": r"$P_{f,\mathrm{dom}}$",
        "dom_freq_hz": r"$f_{\mathrm{dom}}$",
        "skew_adc": r"$\mathrm{skew}$",
        "kurtosis_adc": r"$\mathrm{kurtosis}$",
        "slope_adc": r"$\mathrm{slope}$",
    }

    plot_df = feature_df.copy()

    # Z-score normalisation per feature.
    for feat in selected_features:
        mu = plot_df[feat].mean()
        sigma = plot_df[feat].std()
        if sigma == 0 or not np.isfinite(sigma):
            plot_df[feat + "_z"] = 0.0
        else:
            plot_df[feat + "_z"] = (plot_df[feat] - mu) / sigma

    summary_rows = []
    for feat in selected_features:
        z_col = feat + "_z"
        for cls in CLASS_ORDER:
            values = plot_df.loc[
                plot_df["case"].astype(str) == cls,
                z_col
            ].dropna().to_numpy()

            summary_rows.append({
                "feature": feat,
                "case": cls,
                "mean_z": np.mean(values) if len(values) > 0 else np.nan,
                "std_z": np.std(values) if len(values) > 0 else np.nan,
                "n_windows": len(values)
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("Fig_3_15_top_discriminative_feature_summary.csv", index=False)

    n_features = len(selected_features)
    n_classes = len(CLASS_ORDER)

    x = np.arange(n_features)
    total_width = 0.86
    bar_width = total_width / n_classes

    fig, ax = plt.subplots(figsize=(14, 6))

    for class_idx, cls in enumerate(CLASS_ORDER):
        means = []
        for feat in selected_features:
            row = summary_df[
                (summary_df["feature"] == feat)
                & (summary_df["case"] == cls)
            ]
            means.append(row["mean_z"].values[0])

        offset = (class_idx - (n_classes - 1) / 2) * bar_width

        ax.bar(
            x + offset,
            means,
            width=bar_width,
            label=cls,
            linewidth=0.4,
            edgecolor="black"
        )

    feature_labels = [feature_label_map.get(feat, feat) for feat in selected_features]

    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, fontsize=11)
    ax.set_ylabel("Normalised feature value (z-score)", fontsize=12)
    ax.set_title("Top discriminative EIP features across main spatial-condition cases", fontsize=12)

    # Horizontal y = 0 axis.
    ax.axhline(y=0, color="black", linewidth=1.0)

    # Vertical dashed separators between feature groups.
    for sep in np.arange(0.5, n_features - 0.5, 1.0):
        ax.axvline(x=sep, color="black", linestyle="--", linewidth=0.8, alpha=0.75)

    ax.grid(axis="y", alpha=0.3)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=5,
        fontsize=8.5,
        frameon=True
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"Saved updated Fig. 3.15 to: {output_path}")


def plot_ai_target_space(
    feature_df,
    output_path="asymmetry_index_target_space_with_symmetric.png"
):
    gt = feature_df[
        ["case", "AI_LR", "AI_UL", "Q1_mL", "Q2_mL", "Q3_mL", "Q4_mL"]
    ].drop_duplicates()

    gt = gt.drop_duplicates(subset=["case", "AI_LR", "AI_UL"])
    gt["case"] = pd.Categorical(gt["case"], categories=CLASS_ORDER, ordered=True)
    gt = gt.sort_values("case")

    gt.to_csv("asymmetry_index_target_space_table_with_symmetric.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))

    for _, row in gt.iterrows():
        ax.scatter(row["AI_LR"], row["AI_UL"], s=110)
        ax.text(row["AI_LR"] + 0.04, row["AI_UL"] + 0.04, str(row["case"]), fontsize=9)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)

    ax.set_xlabel(r"$AI_{L/R} = \frac{(Q2+Q3)-(Q1+Q4)}{Q1+Q2+Q3+Q4}$")
    ax.set_ylabel(r"$AI_{U/L} = \frac{(Q1+Q2)-(Q3+Q4)}{Q1+Q2+Q3+Q4}$")
    ax.set_title("Defined asymmetry-index target space")

    ax.set_xlim([-1.25, 1.25])
    ax.set_ylim([-1.25, 1.25])
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)

    print(f"Saved asymmetry index plot to: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("Building EIP feature dataset...")
    print(f"Data folder: {DATA_DIR.resolve()}")
    print("CSV columns used: time_s, filtered_adc")
    print("Dataset: static + rocking")
    print("Classes: symmetric + Q1-Q4 active + left/right/upper/lower weak")
    print(f"Active volume included: {ACTIVE_VOLUME_FOR_CLASSIFICATION}")
    print(f"Symmetric volume included: {SYMMETRIC_VOLUME_FOR_CLASSIFICATION}")
    print(f"Window length: {WINDOW_SECONDS:.1f} s")
    print(f"Step length: {STEP_SECONDS:.1f} s")
    print(f"Transient removed: first {TRANSIENT_SECONDS:.1f} s")
    print(f"Evaluation: repeated stratified CV, {N_SPLITS}-fold × {N_REPEATS} repeats")
    print("Classifier: medium-strength Random Forest")

    feature_df = build_feature_dataset(DATA_DIR)

    feature_df.to_csv("EIP_feature_dataset_RF_medium_strength_repeated_cv_ready.csv", index=False)
    print("\nSaved feature dataset to: EIP_feature_dataset_RF_medium_strength_repeated_cv_ready.csv")

    print("\nIncluded files:")
    file_table = feature_df[
        ["file", "case", "volume_level", "condition", "trial_id", "split_key"]
    ].drop_duplicates()
    print(file_table.to_string(index=False))
    file_table.to_csv("included_files_RF_medium_strength_repeated_cv.csv", index=False)

    print("\nDataset class distribution:")
    print(feature_df["case"].value_counts().reindex(CLASS_ORDER))

    print("\nDataset condition/trial distribution:")
    print(feature_df["split_key"].value_counts())

    gt_cols = [
        "file",
        "case",
        "volume_level",
        "condition",
        "trial_id",
        "Q1_mL",
        "Q2_mL",
        "Q3_mL",
        "Q4_mL",
        "AI_LR",
        "AI_UL"
    ]
    gt_table = feature_df[gt_cols].drop_duplicates().reset_index(drop=True)
    gt_table.to_csv("EIP_ground_truth_AI_table_RF_medium_strength_repeated_cv.csv", index=False)

    final_clf, feature_cols, cm_total, all_y_true, all_y_pred, fold_df = repeated_cv_evaluation(feature_df)

    plot_feature_grouped_bar_chart(
        feature_df,
        final_clf,
        feature_cols,
        output_path="Fig_3_15_top_discriminative_features.png",
        top_n=5
    )

    plot_ai_target_space(feature_df)
    plot_normalised_confusion_matrix_from_cm(cm_total)
    plot_random_forest_feature_importance(final_clf, feature_cols)

    print("\nGenerated figures:")
    print("  Fig_3_15_top_discriminative_features.png")
    print("  asymmetry_index_target_space_with_symmetric.png")
    print("  confusion_matrix_RF_medium_strength_repeated_cv_normalised_percent.png")
    print("  random_forest_feature_importance_RF_medium_strength_repeated_cv.png")

    print("\nGenerated tables:")
    print("  EIP_feature_dataset_RF_medium_strength_repeated_cv_ready.csv")
    print("  included_files_RF_medium_strength_repeated_cv.csv")
    print("  EIP_ground_truth_AI_table_RF_medium_strength_repeated_cv.csv")
    print("  repeated_cv_fold_metrics_RF_medium_strength.csv")
    print("  classification_report_RF_medium_strength_repeated_cv.csv")
    print("  EIP_feature_dataset_used_RF_medium_strength_repeated_cv.csv")
    print("  confusion_matrix_RF_medium_strength_repeated_cv_normalised_percent.csv")
    print("  confusion_matrix_RF_medium_strength_repeated_cv_counts.csv")
    print("  random_forest_feature_importance_RF_medium_strength_repeated_cv.csv")
    print("  selected_features_for_Fig_3_15.csv")
    print("  Fig_3_15_top_discriminative_feature_summary.csv")
    print("  asymmetry_index_target_space_table_with_symmetric.csv")

    print("\nDone. Displaying figures now.")
    plt.show()


if __name__ == "__main__":
    main()
