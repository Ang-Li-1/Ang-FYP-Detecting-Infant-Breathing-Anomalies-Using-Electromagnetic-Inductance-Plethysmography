import matplotlib.pyplot as plt
import socket
import collections
import time
import numpy as np
from scipy.signal import butter, lfilter, find_peaks
from sklearn.mixture import GaussianMixture
import select
import csv
import datetime
import os

# =========================================================
# 2Tx2Rx streaming + MUX-synchronised paired sampling
# + IMU-based adaptive filtering via UDP
#
# ESP32 socket line format:
#     t_us,adc
#
# IMU UDP format from MUX+IMU ESP32:
#     t_us,pitch,roll,yaw
#
# Python logic:
#   - Only process when BOTH rightlung and leftlung have new samples.
#   - Do NOT duplicate previous sample if one side skips a sample.
#   - Feature window = 2.5 s.
#   - Prediction update = 1.0 s.
#   - IMU motion is computed from about 10 IMU samples over 1 second.
#   - ANC enabled when motion_level >= 0.05.
# =========================================================

# ===================== Experiment metadata =====================
CONDITION = "C0_static"   # C0_static / C1_rocking_1Hz / C2_vibration_10Hz
BR_TRUE_HZ = np.nan
LABEL_TIME_OFFSET_S = 0.0

# ===================== Logging setup =====================
LOG_DIR = "logs_2Tx2Rx_imu_anc_mux"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_ROTATE_SEC = 480.0

LOG_HEADER = [
    "t_unix", "t_rel_s", "window_idx",
    "condition",

    "t_right_us",
    "t_left_us",
    "mux_dt_us",

    "imu_pitch_deg", "imu_roll_deg", "imu_yaw_deg",
    "motion_level", "anc_mu",

    "A_rightlung", "A_leftlung", "D", "logD",

    "snr_r_db", "snr_l_db",
    "br_hz_est", "br_hz_true", "br_abs_err_hz",

    "spike_replaced_r",
    "spike_replaced_l",
    "spike_replaced_any",
    "spike_count_r",
    "spike_count_l",
    "spike_count_any",

    "true_is_apnea", "pred_is_apnea", "apnea_score",

    "z_low_fixed", "z_high_fixed",
    "z_low_gmm", "z_high_gmm",
    "z_low_final", "z_high_final",
    "gmm_ready", "conf",

    "true_label",
    "pred_label_raw",
    "pred_label_smooth",
    "final_output"
]

log_fp = None
log_writer = None
current_log_path = None
log_file_start_unix = None

def open_new_log_file():
    global log_fp, log_writer, current_log_path, log_file_start_unix

    if log_fp is not None:
        try:
            log_fp.flush()
            log_fp.close()
        except:
            pass

    ts = datetime.datetime.now().strftime("dominance_LR_imuANC_MUX_%Y%m%d_%H%M%S.csv")
    current_log_path = os.path.join(LOG_DIR, ts)
    log_fp = open(current_log_path, "w", newline="")
    log_writer = csv.writer(log_fp)
    log_writer.writerow(LOG_HEADER)
    log_file_start_unix = time.time()
    print(f"[Log] New file: {current_log_path}")

def maybe_rotate_log_file(now_unix: float):
    global log_file_start_unix

    if log_file_start_unix is None:
        open_new_log_file()
        return

    if (now_unix - log_file_start_unix) >= LOG_ROTATE_SEC:
        open_new_log_file()

open_new_log_file()

# ===================== Confusion matrix =====================
GMM_LABELS = ["Right Lung Collapse", "Balanced", "Left Lung Collapse"]
cm = {t: {p: 0 for p in GMM_LABELS} for t in GMM_LABELS}

def update_confusion(true_lab: str, pred_lab: str):
    if true_lab in cm and pred_lab in cm[true_lab]:
        cm[true_lab][pred_lab] += 1

def save_confusion_matrix():
    cm_name = datetime.datetime.now().strftime("confusion_GMM_%Y%m%d_%H%M%S.csv")
    cm_path = os.path.join(LOG_DIR, cm_name)

    try:
        with open(cm_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["True\\Pred"] + GMM_LABELS)
            for t in GMM_LABELS:
                w.writerow([t] + [cm[t][p] for p in GMM_LABELS])
        print(f"[CM] Saved GMM confusion matrix to: {cm_path}")
    except Exception as e:
        print(f"[CM] Save failed: {e}")

    print("\n[GMM Confusion Matrix] rows=True, cols=Pred")
    header = " " * 16 + "  ".join([f"{c[:12]:>12}" for c in GMM_LABELS])
    print(header)

    for t in GMM_LABELS:
        row = [cm[t][p] for p in GMM_LABELS]
        print(f"{t[:16]:<16}" + "  ".join([f"{v:>12d}" for v in row]))

    print("")

# ===================== Apnea performance =====================
apnea_tp = apnea_fp = apnea_tn = apnea_fn = 0

def update_apnea_cm(true_is_apnea: bool, pred_is_apnea: bool):
    global apnea_tp, apnea_fp, apnea_tn, apnea_fn

    if true_is_apnea and pred_is_apnea:
        apnea_tp += 1
    elif (not true_is_apnea) and pred_is_apnea:
        apnea_fp += 1
    elif (not true_is_apnea) and (not pred_is_apnea):
        apnea_tn += 1
    else:
        apnea_fn += 1

def print_apnea_metrics():
    total = apnea_tp + apnea_fp + apnea_tn + apnea_fn

    if total == 0:
        print("[Apnea] No samples to score.")
        return

    acc = (apnea_tp + apnea_tn) / total
    se = apnea_tp / (apnea_tp + apnea_fn) if (apnea_tp + apnea_fn) > 0 else np.nan
    sp = apnea_tn / (apnea_tn + apnea_fp) if (apnea_tn + apnea_fp) > 0 else np.nan

    print(
        f"[Apnea] Acc={acc*100:.1f}%  "
        f"Se={se*100:.1f}%  "
        f"Sp={sp*100:.1f}%  "
        f"(TP={apnea_tp}, FP={apnea_fp}, TN={apnea_tn}, FN={apnea_fn})"
    )

# =========================================================
# Network
# =========================================================
ESP32_RIGHTLUNG_IP = "10.56.110.130"
ESP32_LEFTLUNG_IP  = "10.56.110.110"
PORT = 3333

SOCKET_RECV_TIMEOUT = 0.01
RECONNECT_BACKOFF_S = 1.0
SELECT_TIMEOUT = 0.01

# =========================================================
# IMU UDP + robust motion estimation
# =========================================================
USE_IMU_ANC = True
IMU_UDP_PORT = 5006

# Use about 10 IMU samples over 1 second for robust motion.
IMU_SAMPLE_INTERVAL_SEC = 0.1
IMU_MOTION_WINDOW_SEC = 1.0
IMU_MOTION_MAXLEN = int(IMU_MOTION_WINDOW_SEC / IMU_SAMPLE_INTERVAL_SEC)

# ANC enable threshold
MOTION_ENABLE_TH = 0.05

def open_imu_udp_socket():
    if not USE_IMU_ANC:
        return None

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", IMU_UDP_PORT))
        s.setblocking(False)
        print(f"[IMU] Listening UDP on port {IMU_UDP_PORT}")
        return s
    except Exception as e:
        print(f"[IMU] WARNING: cannot open UDP port {IMU_UDP_PORT}: {e}")
        print("[IMU] ANC disabled fallback.")
        return None

def read_all_imu_udp(sock):
    """
    Read all available IMU UDP packets.

    Expected UDP format:
        t_us,pitch,roll,yaw

    Returns:
        list of (t_us, pitch, roll, yaw)
    """
    if sock is None:
        return []

    samples = []

    while True:
        try:
            data, _ = sock.recvfrom(256)
            line = data.decode(errors="ignore").strip()
            parts = line.split(",")

            if len(parts) != 4:
                continue

            t_us = int(parts[0])
            pitch = float(parts[1])
            roll = float(parts[2])
            yaw = float(parts[3])

            samples.append((t_us, pitch, roll, yaw))

        except BlockingIOError:
            break
        except Exception:
            break

    return samples

def unwrap_yaw_sequence(yaw_deg_list):
    """
    Unwrap yaw angle sequence to avoid 359 -> 0 discontinuity.
    """
    if len(yaw_deg_list) == 0:
        return np.array([], dtype=float)

    y = np.asarray(yaw_deg_list, dtype=float)
    y_unwrap = [y[0]]

    for i in range(1, len(y)):
        dy = y[i] - y[i - 1]

        if dy > 180.0:
            dy -= 360.0
        elif dy < -180.0:
            dy += 360.0

        y_unwrap.append(y_unwrap[-1] + dy)

    return np.asarray(y_unwrap, dtype=float)

def compute_motion_from_imu_window(imu_window):
    """
    Compute robust motion level from around 10 IMU samples within 1 second.

    imu_window item format:
        (t_us, pitch, roll, yaw)

    Returns:
        motion_level in [0, 1]
    """
    if len(imu_window) < 3:
        return 0.0

    arr = list(imu_window)

    t_us = np.asarray([x[0] for x in arr], dtype=float)
    p = np.asarray([x[1] for x in arr], dtype=float)
    r = np.asarray([x[2] for x in arr], dtype=float)
    y = unwrap_yaw_sequence([x[3] for x in arr])

    # ESP32 micros timestamp to seconds.
    t = (t_us - t_us[0]) * 1e-6

    # Remove static posture by subtracting median.
    p_dev = p - np.median(p)
    r_dev = r - np.median(r)
    y_dev = y - np.median(y)

    # Yaw weighted less because it is often noisier / less relevant.
    angle_rms = np.sqrt(
        np.mean(
            p_dev**2 +
            r_dev**2 +
            0.25 * y_dev**2
        )
    )

    # Robust derivative using ESP32 IMU timestamps.
    dt = np.diff(t)
    valid = dt > 1e-3

    if np.sum(valid) < 2:
        rate_rms = 0.0
    else:
        dp = np.diff(p)[valid] / dt[valid]
        dr = np.diff(r)[valid] / dt[valid]
        dy = np.diff(y)[valid] / dt[valid]

        # Clip unrealistic derivative spikes from packet timing/jitter.
        dp = np.clip(dp, -200.0, 200.0)
        dr = np.clip(dr, -200.0, 200.0)
        dy = np.clip(dy, -200.0, 200.0)

        rate_rms = np.sqrt(
            np.mean(
                dp**2 +
                dr**2 +
                0.25 * dy**2
            )
        )

    # Normalize to [0,1].
    # Tuned to avoid static IMU jitter saturating motion.
    angle_score = angle_rms / 5.0       # 5 deg RMS -> strong motion
    rate_score = rate_rms / 80.0        # 80 deg/s RMS -> strong motion

    motion = 0.6 * angle_score + 0.4 * rate_score

    return float(np.clip(motion, 0.0, 1.0))

class IMUReferenceBuilder:
    """
    Build motion reference vector from pitch/roll/yaw:
    [p_hp, r_hp, y_hp, dp, dr, dy]

    The returned local motion is not used as the main motion_level.
    Main motion_level is computed from the robust 1 s IMU window.
    """

    def __init__(self, tau_s=1.5):
        self.tau_s = tau_s
        self.initialized = False

    def update(self, pitch_deg, roll_deg, yaw_deg, dt):
        dt = max(dt, 1e-3)

        if not self.initialized:
            self.p_prev = pitch_deg
            self.r_prev = roll_deg
            self.y_prev_raw = yaw_deg
            self.y_unwrap = yaw_deg
            self.y_prev_unwrap = yaw_deg

            self.ema_p = pitch_deg
            self.ema_r = roll_deg
            self.ema_y = yaw_deg

            self.initialized = True
            return np.zeros(6, dtype=float), 0.0

        dy_raw = yaw_deg - self.y_prev_raw

        if dy_raw > 180.0:
            dy_raw -= 360.0
        elif dy_raw < -180.0:
            dy_raw += 360.0

        self.y_unwrap += dy_raw
        self.y_prev_raw = yaw_deg

        alpha = dt / (self.tau_s + dt)

        self.ema_p += alpha * (pitch_deg - self.ema_p)
        self.ema_r += alpha * (roll_deg - self.ema_r)
        self.ema_y += alpha * (self.y_unwrap - self.ema_y)

        p_hp = pitch_deg - self.ema_p
        r_hp = roll_deg - self.ema_r
        y_hp = self.y_unwrap - self.ema_y

        dp = (pitch_deg - self.p_prev) / dt
        dr = (roll_deg - self.r_prev) / dt
        dy = (self.y_unwrap - self.y_prev_unwrap) / dt

        self.p_prev = pitch_deg
        self.r_prev = roll_deg
        self.y_prev_unwrap = self.y_unwrap

        # Clip derivatives first to prevent UDP burst timing from creating huge values.
        dp = np.clip(dp, -200.0, 200.0)
        dr = np.clip(dr, -200.0, 200.0)
        dy = np.clip(dy, -200.0, 200.0)

        # Conservative reference scaling.
        ref = np.array([
            p_hp / 25.0,
            r_hp / 25.0,
            y_hp / 60.0,
            dp / 150.0,
            dr / 150.0,
            dy / 250.0
        ], dtype=float)

        ref = np.clip(ref, -3.0, 3.0)

        local_motion = float(np.clip(np.sqrt(np.mean(ref**2)) / 1.2, 0.0, 1.0))

        return ref, local_motion

class MultiLagNLMS:
    """
    Multi-reference, multi-lag normalized LMS adaptive filter.

    Less aggressive version:
    - fewer lags
    - lower mu
    - stronger leakage
    - correction clamp
    """

    def __init__(
        self,
        n_features,
        n_lags=4,
        mu_static=0.0,
        mu_motion=0.03,
        leak=1e-3,
        eps=1e-6,
        max_correction=40.0
    ):
        self.n_features = n_features
        self.n_lags = n_lags
        self.mu_static = mu_static
        self.mu_motion = mu_motion
        self.leak = leak
        self.eps = eps
        self.max_correction = max_correction

        self.hist = collections.deque(
            [np.zeros(n_features, dtype=float) for _ in range(n_lags)],
            maxlen=n_lags
        )

        self.w = np.zeros(n_features * n_lags, dtype=float)

    def update(self, d, ref_vec, motion_level):
        ref_vec = np.asarray(ref_vec, dtype=float)

        self.hist.appendleft(ref_vec.copy())
        x = np.concatenate(list(self.hist))

        y_hat = float(np.dot(self.w, x))
        y_hat = float(np.clip(y_hat, -self.max_correction, self.max_correction))

        e = float(d - y_hat)

        mu = self.mu_static + (self.mu_motion - self.mu_static) * motion_level
        denom = self.eps + float(np.dot(x, x))

        self.w = (1.0 - self.leak) * self.w + (mu / denom) * e * x

        return e, y_hat, mu

imu_udp_sock = open_imu_udp_socket()
imu_builder = IMUReferenceBuilder(tau_s=1.5)

anc_right = MultiLagNLMS(
    n_features=6,
    n_lags=2,
    mu_static=0.0,
    mu_motion=0.01,
    leak=5e-3,
    eps=1e-6,
    max_correction=10.0
)

anc_left = MultiLagNLMS(
    n_features=6,
    n_lags=2,
    mu_static=0.0,
    mu_motion=0.01,
    leak=5e-3,
    eps=1e-6,
    max_correction=10.0
)

latest_imu = None
latest_imu_ref_vec = np.zeros(6, dtype=float)
latest_motion_level = 0.0

last_imu_sample_t_us = None
last_imu_downsample_time_s = 0.0

# Store about 10 IMU samples over 1 second.
imu_motion_window = collections.deque(maxlen=IMU_MOTION_MAXLEN)

# =========================================================
# Signal processing
# =========================================================
FS = 20.0
CUTOFF = 1.5
FILTER_ORDER = 2

BUFFER_LEN = 200
UPDATE_INTERVAL = 2

MIN_BREATH_DELTA = 20
MIN_BREATH_DURATION = 0.1
MAX_BREATH_DURATION = 10.0

def butter_lowpass(cutoff, fs, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype="low", analog=False)

b, a = butter_lowpass(CUTOFF, FS, FILTER_ORDER)

zi_rightlung = np.zeros(max(len(a), len(b)) - 1)
zi_leftlung  = np.zeros(max(len(a), len(b)) - 1)

def hampel_last_value_with_flag(dq, window_size=9, n_sigmas=3.0):
    if len(dq) < window_size:
        return int(dq[-1]), False

    w = np.asarray(list(dq)[-window_size:], dtype=float)

    x0 = w[-1]
    med = np.median(w)
    mad = np.median(np.abs(w - med)) + 1e-6
    sigma = 1.4826 * mad

    if np.abs(x0 - med) > n_sigmas * sigma:
        return int(med), True

    return int(x0), False

# ===================== Window features =====================
WINDOW_SEC = 2.5
WINDOW_SAMPLES = int(WINDOW_SEC * FS)
WINDOW_UPDATE_SEC = 1.0

EPS = 1e-6

def mean_breath_pp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size

    if n < 3:
        return np.nan

    min_distance = max(1, int(0.3 * FS))
    prom = max(5.0, MIN_BREATH_DELTA * 0.5)

    trough_idx, _ = find_peaks(-x, distance=min_distance, prominence=prom)

    if trough_idx is None or len(trough_idx) < 2:
        return float(np.max(x) - np.min(x))

    trough_idx = np.sort(trough_idx)

    pp_list = []

    for i in range(len(trough_idx) - 1):
        a_i = trough_idx[i]
        b_i = trough_idx[i + 1]

        if b_i <= a_i + 1:
            continue

        seg = x[a_i:b_i + 1]
        pp = float(np.max(seg) - np.min(seg))

        if np.isfinite(pp) and pp > 0:
            pp_list.append(pp)

    if len(pp_list) == 0:
        return float(np.max(x) - np.min(x))

    return float(np.mean(pp_list))

def compute_A_and_D(rightlung_window, leftlung_window):
    x_r = np.asarray(rightlung_window, dtype=float)
    x_l = np.asarray(leftlung_window, dtype=float)

    A_r = mean_breath_pp(x_r)
    A_l = mean_breath_pp(x_l)

    if not np.isfinite(A_r) or not np.isfinite(A_l) or A_l < EPS:
        return np.nan, np.nan, np.nan, np.nan

    D = A_r / (A_l + EPS)
    z = float(np.log(D + EPS))

    return A_r, A_l, float(D), z

# ===================== SNR =====================
def bandpower(x, fs, f1, f2):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)

    n = len(x)

    if n < 8:
        return np.nan

    win = np.hanning(n)
    X = np.fft.rfft(x * win)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    psd = (np.abs(X) ** 2)

    idx = (freqs >= f1) & (freqs <= f2)

    if not np.any(idx):
        return np.nan

    return float(np.sum(psd[idx]))

def snr_db_window(x, fs):
    pb = bandpower(x, fs, 0.2, 1.2)
    pn = bandpower(x, fs, 1.5, min(8.0, fs / 2 - 0.1))

    if not np.isfinite(pb) or not np.isfinite(pn) or pn <= 1e-9:
        return np.nan

    return float(10.0 * np.log10(pb / pn))

# ===================== Apnea detection diagnostics =====================
APNEA_A_TH = 40.0

def apnea_score_from_A(A_r, A_l, th=APNEA_A_TH):
    if not np.isfinite(A_r) or not np.isfinite(A_l):
        return 0.0

    s_r = max(0.0, min(1.0, 1.0 - A_r / (th + 1e-6)))
    s_l = max(0.0, min(1.0, 1.0 - A_l / (th + 1e-6)))

    return float(min(s_r, s_l))

# ===================== Classification thresholds =====================
D_LOW  = 0.8
D_HIGH = 1.70

Z_LOW_FIXED  = float(np.log(D_LOW))
Z_HIGH_FIXED = float(np.log(D_HIGH))

WARMUP_SEC = 0
MIN_WINDOWS_FOR_GMM = 18
GMM_REFIT_EVERY = 3
Z_HISTORY_MAX = 240
ALPHA_FIXED = 0.3

MIN_Z_STD_FOR_GMM = 0.12
MIN_CENTER_GAP = 0.10

ENABLE_TRANSITION_GUARD = True
Z_JUMP_SKIP = 0.6

def label_from_z(z, z_low, z_high):
    if not np.isfinite(z):
        return "Unknown"

    if z < z_low:
        return "Right Lung Collapse"
    elif z > z_high:
        return "Left Lung Collapse"
    else:
        return "Balanced"

# ===================== True label schedule =====================
MODE_INTERVAL_S = 60.0

def true_label_from_time(t_rel_s: float) -> str:
    mode = int((t_rel_s // MODE_INTERVAL_S) % 4)

    if mode == 0:
        return "Balanced"
    elif mode == 1:
        return "Right Lung Collapse"
    elif mode == 2:
        return "Left Lung Collapse"
    else:
        return "Apnea"

def majority_vote(labels):
    counts = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    best = max(counts.items(), key=lambda kv: (kv[1], kv[0] != "Unknown"))[0]
    return best

# ===================== Socket helpers =====================
def connect_socket(ip, port, timeout):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5.0)
    s.connect((ip, port))
    s.settimeout(timeout)
    return s

def safe_close(s):
    try:
        if s is not None:
            s.close()
    except:
        pass

def recv_lines(sock, buf_state):
    lines = []

    while True:
        try:
            data = sock.recv(4096)

            if not data:
                break

            text = buf_state["buf"] + data.decode(errors="ignore")
            parts = text.splitlines()

            if text.endswith("\n") or text.endswith("\r"):
                buf_state["buf"] = ""
                lines.extend(parts)
            else:
                if len(parts) == 0:
                    buf_state["buf"] = text
                else:
                    buf_state["buf"] = parts[-1]
                    lines.extend(parts[:-1])

        except socket.timeout:
            break
        except BlockingIOError:
            break
        except Exception:
            break

    return lines

def parse_latest_sample(lines):
    """
    Parse latest MUX-synchronised sample.

    Expected line format:
        t_us,adc
    """
    for s in reversed(lines):
        s = s.strip()

        if "," not in s:
            continue

        parts = s.split(",", 1)

        if len(parts) != 2:
            continue

        t_str = parts[0].strip()
        v_str = parts[1].strip()

        if t_str.isdigit() and v_str.isdigit():
            return int(t_str), int(v_str)

    return None

# ===================== Plot =====================
plt.ion()
fig, ax = plt.subplots()

ax.set_title("2Tx2Rx Filtered Signals — Left vs Right Lung (MUX + IMU ANC)")
ax.set_xlabel("Samples")
ax.set_ylabel("ADC Value")
ax.set_xlim(0, BUFFER_LEN)
ax.set_ylim(3000, 4000)

line_right, = ax.plot(
    range(BUFFER_LEN),
    [0] * BUFFER_LEN,
    label="Rx_rightlung",
    linewidth=1.8
)

line_left, = ax.plot(
    range(BUFFER_LEN),
    [0] * BUFFER_LEN,
    label="Rx_leftlung",
    linewidth=1.8
)

ax.legend(loc="upper right")

info_text = ax.text(
    0.5,
    1.08,
    "",
    transform=ax.transAxes,
    ha="center",
    fontsize=10,
    color="black"
)

# ===================== Buffers =====================
raw_right = collections.deque([0] * BUFFER_LEN, maxlen=BUFFER_LEN)
raw_left  = collections.deque([0] * BUFFER_LEN, maxlen=BUFFER_LEN)

pre_right = collections.deque([0] * BUFFER_LEN, maxlen=BUFFER_LEN)
pre_left  = collections.deque([0] * BUFFER_LEN, maxlen=BUFFER_LEN)

filt_right = collections.deque([0] * BUFFER_LEN, maxlen=BUFFER_LEN)
filt_left  = collections.deque([0] * BUFFER_LEN, maxlen=BUFFER_LEN)

win_right = collections.deque(maxlen=WINDOW_SAMPLES)
win_left  = collections.deque(maxlen=WINDOW_SAMPLES)

z_hist = collections.deque(maxlen=Z_HISTORY_MAX)

gmm = None
z_low_gmm = None
z_high_gmm = None
last_gmm_refit_window = -999

label_hist = collections.deque(maxlen=3)

# ===================== BR estimation =====================
BR_AVG_SEC = 5.0

breath_times_r = collections.deque()
breath_times_l = collections.deque()

def _prune_times(dq, now, horizon_s):
    while len(dq) > 0 and (now - dq[0]) > horizon_s:
        dq.popleft()

def avg_freq_from_times(dq, now, horizon_s):
    _prune_times(dq, now, horizon_s)

    if len(dq) < 2:
        return np.nan

    dt = dq[-1] - dq[0]

    if dt <= 1e-6:
        return np.nan

    return float((len(dq) - 1) / dt)

# ===================== Breathing detector states =====================
state_r = "WAIT_MIN"
min_r = 4096
max_r = 0
tmin_r = time.time()

state_l = "WAIT_MIN"
min_l = 4096
max_l = 0
tmin_l = time.time()

def update_breathing_detector(fc, now, state, min_val, max_val, t_min, times_deque):
    if state == "WAIT_MIN":
        if fc < min_val:
            min_val = fc
            t_min = now
        elif fc - min_val > MIN_BREATH_DELTA:
            max_val = fc
            state = "WAIT_MAX"

    elif state == "WAIT_MAX":
        if fc > max_val:
            max_val = fc
        elif max_val - fc > MIN_BREATH_DELTA:
            duration = now - t_min

            if MIN_BREATH_DURATION <= duration <= MAX_BREATH_DURATION:
                times_deque.append(now)

            state = "WAIT_MIN"
            min_val = 4096
            max_val = 0
            t_min = now

    if now - t_min > MAX_BREATH_DURATION:
        state = "WAIT_MIN"
        min_val = 4096
        max_val = 0
        t_min = now

    return state, min_val, max_val, t_min

# ===================== Main sockets =====================
sock_right = None
sock_left  = None

buf_right = {"buf": ""}
buf_left  = {"buf": ""}

def ensure_connected():
    global sock_right, sock_left, buf_right, buf_left

    while True:
        try:
            if sock_right is None:
                sock_right = connect_socket(
                    ESP32_RIGHTLUNG_IP,
                    PORT,
                    SOCKET_RECV_TIMEOUT
                )
                buf_right = {"buf": ""}
                print(f"[Socket] Connected right lung: {ESP32_RIGHTLUNG_IP}:{PORT}")

            if sock_left is None:
                sock_left = connect_socket(
                    ESP32_LEFTLUNG_IP,
                    PORT,
                    SOCKET_RECV_TIMEOUT
                )
                buf_left = {"buf": ""}
                print(f"[Socket] Connected left lung: {ESP32_LEFTLUNG_IP}:{PORT}")

            return

        except Exception as e:
            print(f"[Connect] failed: {e}. retrying...")

            safe_close(sock_right)
            safe_close(sock_left)

            sock_right = None
            sock_left = None

            time.sleep(RECONNECT_BACKOFF_S)

try:
    print("[Info] Connecting sockets...")
    ensure_connected()

    print("[Info] Connected.")
    print("[Info] MUX-synchronised paired-sample logic enabled.")
    print("[Info] Expected socket format is 't_us,adc'.")
    print("[Info] IMU UDP format is 't_us,pitch,roll,yaw'.")
    print("[Info] Python only processes when both channels have new samples.")
    print("[Info] IMU-based adaptive filtering inserted before low-pass.")
    print("[Info] ANC less aggressive mode enabled.")
    print(f"[Info] IMU motion window = {IMU_MOTION_WINDOW_SEC:.1f} s, about {IMU_MOTION_MAXLEN} IMU samples.")
    print(f"[Info] MOTION_ENABLE_TH = {MOTION_ENABLE_TH}")
    print("[Info] Motion1s is computed from robust 1 s IMU window, not max spike.")
    print("[Info] ANC: n_lags=4, mu_static=0.0, mu_motion=0.03, leak=1e-3, max_correction=40 ADC counts.")
    print(f"[Info] Feature window = {WINDOW_SEC:.1f} s, prediction update = {WINDOW_UPDATE_SEC:.1f} s.")
    print("[Info] GMM enable after MIN_WINDOWS_FOR_GMM prediction windows.")
    print(f"[Info] Condition tag: {CONDITION}")
    print(f"[Info] Apnea diagnostic rule: A_leftlung < {APNEA_A_TH:.0f} and A_rightlung < {APNEA_A_TH:.0f}.")
    print(f"[Info] Log rotation: every {LOG_ROTATE_SEC/60:.1f} minutes.")

    sample_counter = 0
    start_time = time.time()

    total_windows = 0
    last_window_emit_time = -1e9

    last_seen_t_r = None
    last_seen_t_l = None

    pending_r = None
    pending_l = None

    paired_sample_count = 0
    skipped_wait_count = 0

    spike_count_r_win = 0
    spike_count_l_win = 0

    while True:
        now = time.time()
        maybe_rotate_log_file(now)

        # ---------------- IMU UDP update ----------------
        imu_samples = read_all_imu_udp(imu_udp_sock)

        for imu_t_us, p, r, y in imu_samples:
            # Use ESP32 IMU timestamp for dt, not Python packet receive time.
            if last_imu_sample_t_us is None:
                imu_dt = 1.0 / 25.0   # IMU is sent at about 25 Hz
            else:
                dt_us = int(imu_t_us) - int(last_imu_sample_t_us)

                # Handle possible micros() wrap, though unlikely within short tests.
                if dt_us < 0:
                    dt_us += 2**32

                imu_dt = max(dt_us * 1e-6, 1e-3)

            last_imu_sample_t_us = imu_t_us

            # Update ANC reference using every valid IMU packet.
            ref_vec_tmp, _ = imu_builder.update(p, r, y, imu_dt)

            # Downsample IMU motion window to about 10 Hz.
            current_time_s = time.time()
            if (current_time_s - last_imu_downsample_time_s) >= IMU_SAMPLE_INTERVAL_SEC:
                imu_motion_window.append((imu_t_us, p, r, y))
                last_imu_downsample_time_s = current_time_s
                latest_imu = (p, r, y)
                latest_imu_ref_vec = ref_vec_tmp

        # Compute robust motion from about 10 IMU values within 1 second.
        if len(imu_motion_window) >= 3:
            latest_motion_level = compute_motion_from_imu_window(imu_motion_window)
        else:
            latest_motion_level = 0.0

        if latest_imu is not None:
            imu_pitch_deg, imu_roll_deg, imu_yaw_deg = latest_imu
            imu_ref_vec = latest_imu_ref_vec
            motion_level = latest_motion_level
        else:
            imu_pitch_deg = np.nan
            imu_roll_deg = np.nan
            imu_yaw_deg = np.nan
            imu_ref_vec = np.zeros(6, dtype=float)
            motion_level = 0.0

        # ---------------- socket select ----------------
        try:
            rlist, _, _ = select.select(
                [sock_right, sock_left],
                [],
                [],
                SELECT_TIMEOUT
            )

        except Exception as e:
            print(f"[Select] error: {e}. reconnecting...")

            safe_close(sock_right)
            safe_close(sock_left)

            sock_right = None
            sock_left = None

            ensure_connected()
            plt.pause(0.001)
            continue

        # ---------------- read MUX-synchronised socket samples ----------------
        try:
            if sock_right in rlist:
                lines_r = recv_lines(sock_right, buf_right)
                sr = parse_latest_sample(lines_r)

                if sr is not None:
                    t_r_us, v_r_new = sr

                    if last_seen_t_r is None or t_r_us != last_seen_t_r:
                        last_seen_t_r = t_r_us
                        pending_r = (t_r_us, v_r_new)

            if sock_left in rlist:
                lines_l = recv_lines(sock_left, buf_left)
                sl = parse_latest_sample(lines_l)

                if sl is not None:
                    t_l_us, v_l_new = sl

                    if last_seen_t_l is None or t_l_us != last_seen_t_l:
                        last_seen_t_l = t_l_us
                        pending_l = (t_l_us, v_l_new)

        except Exception as e:
            print(f"[Socket] error: {e}. reconnecting...")

            safe_close(sock_right)
            safe_close(sock_left)

            sock_right = None
            sock_left = None

            ensure_connected()
            plt.pause(0.001)
            continue

        # Only process one data point when BOTH channels have new samples.
        if pending_r is None or pending_l is None:
            skipped_wait_count += 1
            plt.pause(0.001)
            continue

        t_r_us, v_r = pending_r
        t_l_us, v_l = pending_l

        pending_r = None
        pending_l = None

        paired_sample_count += 1
        now = time.time()

        if t_r_us is not None and t_l_us is not None:
            mux_dt_us = abs(int(t_r_us) - int(t_l_us))
        else:
            mux_dt_us = np.nan

        # ---------------- raw + spike suppression ----------------
        raw_right.append(v_r)
        raw_left.append(v_l)

        pre_vr, spike_r_this_sample = hampel_last_value_with_flag(
            raw_right,
            window_size=9,
            n_sigmas=3.0
        )

        pre_vl, spike_l_this_sample = hampel_last_value_with_flag(
            raw_left,
            window_size=9,
            n_sigmas=3.0
        )

        if spike_r_this_sample:
            spike_count_r_win += 1

        if spike_l_this_sample:
            spike_count_l_win += 1

        pre_right.append(pre_vr)
        pre_left.append(pre_vl)

        # ---------------- IMU-based adaptive ANC ----------------
        if USE_IMU_ANC and latest_imu is not None and motion_level >= MOTION_ENABLE_TH:
            anc_out_r, art_hat_r, anc_mu = anc_right.update(
                pre_vr,
                imu_ref_vec,
                motion_level
            )

            anc_out_l, art_hat_l, _ = anc_left.update(
                pre_vl,
                imu_ref_vec,
                motion_level
            )
        else:
            anc_out_r = float(pre_vr)
            art_hat_r = 0.0
            anc_mu = 0.0

            anc_out_l = float(pre_vl)
            art_hat_l = 0.0

        # ---------------- low-pass ----------------
        filtered_array_r, zi_rightlung = lfilter(
            b,
            a,
            [anc_out_r],
            zi=zi_rightlung
        )

        filtered_array_l, zi_leftlung = lfilter(
            b,
            a,
            [anc_out_l],
            zi=zi_leftlung
        )

        fr = int(filtered_array_r[-1])
        fl = int(filtered_array_l[-1])

        filt_right.append(fr)
        filt_left.append(fl)

        win_right.append(fr)
        win_left.append(fl)

        # ---------------- breathing detector ----------------
        state_r, min_r, max_r, tmin_r = update_breathing_detector(
            fr,
            now,
            state_r,
            min_r,
            max_r,
            tmin_r,
            breath_times_r
        )

        state_l, min_l, max_l, tmin_l = update_breathing_detector(
            fl,
            now,
            state_l,
            min_l,
            max_l,
            tmin_l,
            breath_times_l
        )

        # ===================== Window update =====================
        if len(win_right) == WINDOW_SAMPLES and len(win_left) == WINDOW_SAMPLES:
            if (now - last_window_emit_time) >= WINDOW_UPDATE_SEC:
                last_window_emit_time = now
                total_windows += 1

                t_rel = now - start_time + LABEL_TIME_OFFSET_S
                true_lab = true_label_from_time(t_rel)
                true_is_apnea = (true_lab == "Apnea")

                A_r, A_l, D, z = compute_A_and_D(win_right, win_left)

                pred_is_apnea = False
                if np.isfinite(A_r) and np.isfinite(A_l):
                    pred_is_apnea = (A_r < APNEA_A_TH) and (A_l < APNEA_A_TH)

                apnea_score = apnea_score_from_A(A_r, A_l, APNEA_A_TH)
                update_apnea_cm(true_is_apnea, pred_is_apnea)

                snr_r_db = snr_db_window(np.array(win_right, dtype=float), FS)
                snr_l_db = snr_db_window(np.array(win_left, dtype=float), FS)

                if np.isfinite(z):
                    if not ENABLE_TRANSITION_GUARD or len(z_hist) == 0:
                        z_hist.append(z)
                    else:
                        if abs(z - z_hist[-1]) < Z_JUMP_SKIP:
                            z_hist.append(z)

                warmup_done = (now - start_time) >= WARMUP_SEC
                enough_windows = total_windows >= MIN_WINDOWS_FOR_GMM
                time_to_refit = (total_windows - last_gmm_refit_window) >= GMM_REFIT_EVERY

                gmm_ready = False
                z_low_final = Z_LOW_FIXED
                z_high_final = Z_HIGH_FIXED
                conf_str = "--"

                if warmup_done and enough_windows and time_to_refit and len(z_hist) >= MIN_WINDOWS_FOR_GMM:
                    z_arr = np.array(z_hist, dtype=float).reshape(-1, 1)
                    z_std = float(np.std(z_arr))

                    if z_std >= MIN_Z_STD_FOR_GMM:
                        try:
                            gmm = GaussianMixture(
                                n_components=3,
                                covariance_type="full",
                                random_state=42,
                                reg_covar=1e-4,
                                max_iter=200
                            )

                            gmm.fit(z_arr)

                            means = np.sort(gmm.means_.flatten())

                            if (
                                (means[1] - means[0] >= MIN_CENTER_GAP)
                                and
                                (means[2] - means[1] >= MIN_CENTER_GAP)
                            ):
                                z_low_gmm = float(0.5 * (means[0] + means[1]))
                                z_high_gmm = float(0.5 * (means[1] + means[2]))
                                last_gmm_refit_window = total_windows
                                gmm_ready = True

                        except Exception as e:
                            print(f"[GMM] fit failed: {e}")
                            gmm_ready = False

                if (z_low_gmm is not None) and (z_high_gmm is not None):
                    gmm_ready = True

                if gmm_ready and (z_low_gmm is not None) and (z_high_gmm is not None):
                    z_low_final = float(
                        ALPHA_FIXED * Z_LOW_FIXED
                        +
                        (1 - ALPHA_FIXED) * z_low_gmm
                    )

                    z_high_final = float(
                        ALPHA_FIXED * Z_HIGH_FIXED
                        +
                        (1 - ALPHA_FIXED) * z_high_gmm
                    )

                    if gmm is not None and np.isfinite(z):
                        try:
                            p = gmm.predict_proba(np.array([[z]], dtype=float))[0]
                            conf_str = f"{float(np.max(p)):.2f}"
                        except:
                            conf_str = "--"

                # ===================== Final output =====================
                spike_r_win = 1 if spike_count_r_win > 0 else 0
                spike_l_win = 1 if spike_count_l_win > 0 else 0
                spike_any_win = 1 if (spike_count_r_win > 0 or spike_count_l_win > 0) else 0
                spike_count_any_win = spike_count_r_win + spike_count_l_win

                if pred_is_apnea:
                    pred_label_raw = "Apnea"
                    pred_label_smooth = "Apnea"
                    final_output = "Apnea"
                else:
                    pred_label_raw = label_from_z(
                        z,
                        z_low_final,
                        z_high_final
                    )

                    label_hist.append(pred_label_raw)
                    pred_label_smooth = majority_vote(list(label_hist))
                    final_output = pred_label_smooth

                freq_r = avg_freq_from_times(breath_times_r, now, BR_AVG_SEC)
                freq_l = avg_freq_from_times(breath_times_l, now, BR_AVG_SEC)

                if np.isfinite(A_r) and np.isfinite(A_l):
                    use_right = (A_r >= A_l)
                else:
                    use_right = True

                if use_right:
                    br_hz_est = freq_r
                else:
                    br_hz_est = freq_l

                br_hz_true = float(BR_TRUE_HZ) if np.isfinite(BR_TRUE_HZ) else np.nan

                br_abs_err = (
                    float(abs(br_hz_est - br_hz_true))
                    if (np.isfinite(br_hz_est) and np.isfinite(br_hz_true))
                    else np.nan
                )

                if (
                    (true_lab in GMM_LABELS)
                    and
                    (pred_label_smooth in GMM_LABELS)
                    and
                    (not pred_is_apnea)
                ):
                    update_confusion(true_lab, pred_label_smooth)

                zlg = z_low_gmm if (z_low_gmm is not None) else ""
                zhg = z_high_gmm if (z_high_gmm is not None) else ""

                log_writer.writerow([
                    f"{now:.6f}",
                    f"{t_rel:.3f}",
                    total_windows,
                    CONDITION,

                    t_r_us if t_r_us is not None else "",
                    t_l_us if t_l_us is not None else "",
                    mux_dt_us if np.isfinite(mux_dt_us) else "",

                    f"{imu_pitch_deg:.4f}" if np.isfinite(imu_pitch_deg) else "",
                    f"{imu_roll_deg:.4f}" if np.isfinite(imu_roll_deg) else "",
                    f"{imu_yaw_deg:.4f}" if np.isfinite(imu_yaw_deg) else "",
                    f"{motion_level:.4f}",
                    f"{anc_mu:.5f}",

                    f"{A_r:.6f}" if np.isfinite(A_r) else "",
                    f"{A_l:.6f}" if np.isfinite(A_l) else "",
                    f"{D:.6f}"   if np.isfinite(D) else "",
                    f"{z:.6f}"   if np.isfinite(z) else "",

                    f"{snr_r_db:.3f}" if np.isfinite(snr_r_db) else "",
                    f"{snr_l_db:.3f}" if np.isfinite(snr_l_db) else "",
                    f"{br_hz_est:.4f}" if np.isfinite(br_hz_est) else "",
                    f"{br_hz_true:.4f}" if np.isfinite(br_hz_true) else "",
                    f"{br_abs_err:.4f}" if np.isfinite(br_abs_err) else "",

                    spike_r_win,
                    spike_l_win,
                    spike_any_win,
                    spike_count_r_win,
                    spike_count_l_win,
                    spike_count_any_win,

                    1 if true_is_apnea else 0,
                    1 if pred_is_apnea else 0,
                    f"{apnea_score:.3f}",

                    f"{Z_LOW_FIXED:.6f}",
                    f"{Z_HIGH_FIXED:.6f}",
                    zlg,
                    zhg,
                    f"{z_low_final:.6f}",
                    f"{z_high_final:.6f}",
                    1 if gmm_ready else 0,
                    conf_str,

                    true_lab,
                    pred_label_raw,
                    pred_label_smooth,
                    final_output
                ])

                if (total_windows % 6) == 0:
                    try:
                        log_fp.flush()
                    except:
                        pass

                Ar_str = f"{A_r:.1f}" if np.isfinite(A_r) else "--"
                Al_str = f"{A_l:.1f}" if np.isfinite(A_l) else "--"
                D_str  = f"{D:.3f}" if np.isfinite(D) else "--"
                br_str = f"{br_hz_est:.3f} Hz" if np.isfinite(br_hz_est) else "--"

                mux_dt_str = f"{mux_dt_us} us" if np.isfinite(mux_dt_us) else "--"

                imu_str = (
                    f"IMU=({imu_pitch_deg:.1f},{imu_roll_deg:.1f},{imu_yaw_deg:.1f})"
                    if np.isfinite(imu_pitch_deg) and np.isfinite(imu_roll_deg) and np.isfinite(imu_yaw_deg)
                    else "IMU=(--,--,--)"
                )

                if pred_is_apnea:
                    info_text.set_text(
                        f"[APNEA] "
                        f"A_L={Al_str}  A_R={Ar_str}  "
                        f"score={apnea_score:.2f}  "
                        f"True={true_lab}  "
                        f"{imu_str}  "
                        f"Motion1s={motion_level:.2f}  "
                        f"mu={anc_mu:.3f}  "
                        f"MUXdt={mux_dt_str}  "
                        f"Pairs={paired_sample_count}  "
                        f"Cond={CONDITION}"
                    )
                else:
                    info_text.set_text(
                        f"BR={br_str}  "
                        f"A_L={Al_str}  A_R={Ar_str}  "
                        f"D={D_str}  "
                        f"Pred={final_output}  "
                        f"True={true_lab}  "
                        f"{imu_str}  "
                        f"Motion1s={motion_level:.2f}  "
                        f"mu={anc_mu:.3f}  "
                        f"MUXdt={mux_dt_str}  "
                        f"Pairs={paired_sample_count}  "
                        f"Cond={CONDITION}"
                    )

                spike_count_r_win = 0
                spike_count_l_win = 0

                if (total_windows % 12) == 0:
                    print_apnea_metrics()

        # ===================== Plot update =====================
        sample_counter += 1

        if sample_counter >= UPDATE_INTERVAL:
            line_right.set_ydata(filt_right)
            line_left.set_ydata(filt_left)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            sample_counter = 0

        plt.pause(0.001)

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    safe_close(sock_right)
    safe_close(sock_left)

    try:
        if imu_udp_sock is not None:
            imu_udp_sock.close()
            print("[IMU] UDP socket closed.")
    except:
        pass

    try:
        if log_fp is not None:
            log_fp.flush()
            log_fp.close()
            print(f"[Log] Closed: {current_log_path}")
    except:
        pass

    save_confusion_matrix()
    print_apnea_metrics()
    
    
    
    
    
    
    
    
    
    
    