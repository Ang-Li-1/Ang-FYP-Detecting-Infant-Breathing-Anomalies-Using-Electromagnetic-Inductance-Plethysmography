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

# ===================== Logging setup (rotate every 9 minutes) =====================
LOG_DIR = "logs_2Tx2Rx"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_ROTATE_SEC = 540.0  # 9 minutes

LOG_HEADER = [
    "t_unix", "t_rel_s", "window_idx",
    "A_rightlung", "A_leftlung", "D", "logD",
    "z_low_fixed", "z_high_fixed",
    "z_low_gmm", "z_high_gmm",
    "z_low_final", "z_high_final",
    "gmm_ready", "conf",
    "true_label",
    "pred_label_raw", "pred_label_smooth"
]

log_fp = None
log_writer = None
current_log_path = None
log_file_start_unix = None

def open_new_log_file():
    """Close current file (if any) and open a new CSV with header."""
    global log_fp, log_writer, current_log_path, log_file_start_unix

    if log_fp is not None:
        try:
            log_fp.flush()
            log_fp.close()
        except:
            pass

    ts = datetime.datetime.now().strftime("dominance_LR_%Y%m%d_%H%M%S.csv")
    current_log_path = os.path.join(LOG_DIR, ts)
    log_fp = open(current_log_path, "w", newline="")
    log_writer = csv.writer(log_fp)
    log_writer.writerow(LOG_HEADER)
    log_file_start_unix = time.time()

    print(f"[Log] New file: {current_log_path}")

def maybe_rotate_log_file(now_unix: float):
    """Rotate log file every LOG_ROTATE_SEC."""
    global log_file_start_unix
    if log_file_start_unix is None:
        open_new_log_file()
        return
    if (now_unix - log_file_start_unix) >= LOG_ROTATE_SEC:
        open_new_log_file()

open_new_log_file()

# ===================== Confusion matrix =====================
LABELS = ["Left-dominant", "Balanced", "Right-dominant"]
cm = {t: {p: 0 for p in LABELS} for t in LABELS}

def update_confusion(true_lab: str, pred_lab: str):
    if true_lab in cm and pred_lab in cm[true_lab]:
        cm[true_lab][pred_lab] += 1

def save_confusion_matrix():
    cm_name = datetime.datetime.now().strftime("confusion_LR_%Y%m%d_%H%M%S.csv")
    cm_path = os.path.join(LOG_DIR, cm_name)
    try:
        with open(cm_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["True\\Pred"] + LABELS)
            for t in LABELS:
                w.writerow([t] + [cm[t][p] for p in LABELS])
        print(f"[CM] Saved confusion matrix to: {cm_path}")
    except Exception as e:
        print(f"[CM] Save failed: {e}")

    print("\n[Confusion Matrix] rows=True, cols=Pred")
    header = " " * 16 + "  ".join([f"{c[:12]:>12}" for c in LABELS])
    print(header)
    for t in LABELS:
        row = [cm[t][p] for p in LABELS]
        print(f"{t[:16]:<16}" + "  ".join([f"{v:>12d}" for v in row]))
    print("")

# =========================================================
# 2Tx2Rx + 2 ESP32 streaming (Rx_rightlung + Rx_leftlung)
# =========================================================

# ===================== Network =====================
ESP32_LEFTLUNG_IP = "10.56.110.110"
ESP32_RIGHTLUNG_IP  = "10.56.110.130"
PORT = 3333

SOCKET_RECV_TIMEOUT = 0.01
RECONNECT_BACKOFF_S = 1.0
SELECT_TIMEOUT = 0.01

# ===================== Signal processing =====================
FS = 20.0
CUTOFF = 1.5
FILTER_ORDER = 2

BUFFER_LEN = 200
UPDATE_INTERVAL = 5

# Breathing detection (kept; now applied per-channel)
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

def suppress_trough(signal_list, window_size=5, threshold=300):
    if len(signal_list) < window_size:
        return signal_list[-1]
    window = list(signal_list)[-window_size:]
    median = np.median(window)
    last_val = signal_list[-1]
    if median - last_val > threshold:
        return int(median)
    return last_val

# ===================== Window features =====================
WINDOW_SEC = 5.0
WINDOW_SAMPLES = int(WINDOW_SEC * FS)
WINDOW_UPDATE_SEC = 5.0

EPS = 1e-6

# --------- NEW: mean of per-breath (max-min) inside the 5s window ----------
def mean_breath_pp(x: np.ndarray) -> float:
    """
    Compute A as: mean over breaths in this 5s window of (max-min) per breath.
    Breath segmentation: consecutive trough-to-trough segments.
    Fallback: if not enough troughs, use global peak-to-peak.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return np.nan

    # trough detection on -x
    # distance: avoid multiple troughs in same breath; 0.3s works well at FS=20Hz (6 samples)
    min_distance = max(1, int(0.3 * FS))

    # prominence tied to your threshold; keep it modest for robustness
    prom = max(5.0, MIN_BREATH_DELTA * 0.5)

    trough_idx, _ = find_peaks(-x, distance=min_distance, prominence=prom)

    # Need at least 2 troughs to define one trough-to-trough segment (= one breath)
    if trough_idx is None or len(trough_idx) < 2:
        return float(np.max(x) - np.min(x))

    # Ensure sorted
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

    # CHANGED: use mean per-breath peak-to-peak inside the 5s window
    A_r = mean_breath_pp(x_r)
    A_l = mean_breath_pp(x_l)

    if not np.isfinite(A_r) or not np.isfinite(A_l) or A_l < EPS:
        return np.nan, np.nan, np.nan, np.nan

    D = A_r / (A_l + EPS)
    z = float(np.log(D + EPS))
    return A_r, A_l, float(D), z

# ===================== Classification thresholds =====================
D_LOW  = 0.75
D_HIGH = 1.50
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
        return "Left-dominant"
    elif z > z_high:
        return "Right-dominant"
    else:
        return "Balanced"

# ===================== True label schedule =====================
MODE_INTERVAL_S = 60.0

def true_label_from_time(t_rel_s: float) -> str:
    mode = int((t_rel_s // MODE_INTERVAL_S) % 3)
    if mode == 0:
        return "Balanced"
    elif mode == 1:
        return "Left-dominant"
    else:
        return "Right-dominant"

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

def parse_latest_int(lines):
    for s in reversed(lines):
        s = s.strip()
        if s.isdigit():
            return int(s)
    return None

# ===================== Plot (single window, 2 lines) =====================
plt.ion()
fig, ax = plt.subplots()
ax.set_title("2Tx2Rx Filtered Signals — Right lung vs Left lung")
ax.set_xlabel("Samples")
ax.set_ylabel("ADC Value")
ax.set_xlim(0, BUFFER_LEN)
ax.set_ylim(2500, 4000)

line_right, = ax.plot(range(BUFFER_LEN), [0]*BUFFER_LEN, label="Rx_rightlung", linewidth=1.8)
line_left,  = ax.plot(range(BUFFER_LEN), [0]*BUFFER_LEN, label="Rx_leftlung", linewidth=1.8)
ax.legend(loc="upper right")

info_text = ax.text(0.5, 1.07, "", transform=ax.transAxes,
                    ha="center", fontsize=11, color="black")

# ===================== Buffers =====================
raw_right = collections.deque([0]*BUFFER_LEN, maxlen=BUFFER_LEN)
raw_left  = collections.deque([0]*BUFFER_LEN, maxlen=BUFFER_LEN)

pre_right = collections.deque([0]*BUFFER_LEN, maxlen=BUFFER_LEN)
pre_left  = collections.deque([0]*BUFFER_LEN, maxlen=BUFFER_LEN)

filt_right = collections.deque([0]*BUFFER_LEN, maxlen=BUFFER_LEN)
filt_left  = collections.deque([0]*BUFFER_LEN, maxlen=BUFFER_LEN)

win_right = collections.deque(maxlen=WINDOW_SAMPLES)
win_left  = collections.deque(maxlen=WINDOW_SAMPLES)

z_hist = collections.deque(maxlen=Z_HISTORY_MAX)
gmm = None
z_low_gmm = None
z_high_gmm = None
last_gmm_refit_window = -999

label_hist = collections.deque(maxlen=3)

# ===================== Breathing rate display (dominant channel, ~5s avg) =====================
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

# ===================== Breathing detector states (kept) =====================
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

# ===================== Main =====================
sock_right = None
sock_left  = None
buf_right = {"buf": ""}
buf_left  = {"buf": ""}

def ensure_connected():
    global sock_right, sock_left, buf_right, buf_left
    while True:
        try:
            if sock_right is None:
                sock_right = connect_socket(ESP32_RIGHTLUNG_IP, PORT, SOCKET_RECV_TIMEOUT)
                buf_right = {"buf": ""}
            if sock_left is None:
                sock_left = connect_socket(ESP32_LEFTLUNG_IP, PORT, SOCKET_RECV_TIMEOUT)
                buf_left = {"buf": ""}
            return
        except Exception as e:
            print(f"[Connect] failed: {e}. retrying...")
            safe_close(sock_right); safe_close(sock_left)
            sock_right = None; sock_left = None
            time.sleep(RECONNECT_BACKOFF_S)

try:
    print("[Info] Connecting sockets...")
    ensure_connected()
    print("[Info] Connected.")
    print("[Info] GMM enable after ~90s now (MIN_WINDOWS_FOR_GMM=18 with 5s windows). Refit every ~15s (GMM_REFIT_EVERY=3).")
    print(f"[Info] Window features use {WINDOW_SEC:.0f}s; display/log update every {WINDOW_UPDATE_SEC:.0f}s.")
    print(f"[Info] Mode interval: {MODE_INTERVAL_S:.0f}s; one full cycle: {3*MODE_INTERVAL_S:.0f}s.")
    print(f"[Info] Log rotation: every {LOG_ROTATE_SEC/60:.0f} minutes.")

    sample_counter = 0
    start_time = time.time()

    total_windows = 0
    last_window_emit_time = -1e9

    last_v_r = None
    last_v_l = None

    target_dt = 1.0 / FS
    next_tick = time.time()

    while True:
        now = time.time()

        maybe_rotate_log_file(now)

        # ---- select ----
        try:
            rlist, _, _ = select.select([sock_right, sock_left], [], [], SELECT_TIMEOUT)
        except Exception as e:
            print(f"[Select] error: {e}. reconnecting...")
            safe_close(sock_right); safe_close(sock_left)
            sock_right = None; sock_left = None
            ensure_connected()
            plt.pause(0.001)
            continue

        # ---- read readable sockets ----
        try:
            if sock_right in rlist:
                lines_r = recv_lines(sock_right, buf_right)
                vr = parse_latest_int(lines_r)
                if vr is not None:
                    last_v_r = vr

            if sock_left in rlist:
                lines_l = recv_lines(sock_left, buf_left)
                vl = parse_latest_int(lines_l)
                if vl is not None:
                    last_v_l = vl

        except Exception as e:
            print(f"[Socket] error: {e}. reconnecting...")
            safe_close(sock_right); safe_close(sock_left)
            sock_right = None; sock_left = None
            ensure_connected()
            plt.pause(0.001)
            continue

        if last_v_r is None or last_v_l is None:
            plt.pause(0.001)
            continue

        v_r = last_v_r
        v_l = last_v_l

        raw_right.append(v_r)
        raw_left.append(v_l)

        pre_vr = suppress_trough(raw_right)
        pre_vl = suppress_trough(raw_left)
        pre_right.append(pre_vr)
        pre_left.append(pre_vl)

        filtered_array_r, zi_rightlung = lfilter(b, a, [pre_vr], zi=zi_rightlung)
        filtered_array_l, zi_leftlung  = lfilter(b, a, [pre_vl], zi=zi_leftlung)
        fr = int(filtered_array_r[-1])
        fl = int(filtered_array_l[-1])

        filt_right.append(fr)
        filt_left.append(fl)

        win_right.append(fr)
        win_left.append(fl)

        state_r, min_r, max_r, tmin_r = update_breathing_detector(
            fr, now, state_r, min_r, max_r, tmin_r, breath_times_r
        )
        state_l, min_l, max_l, tmin_l = update_breathing_detector(
            fl, now, state_l, min_l, max_l, tmin_l, breath_times_l
        )

        # ===================== Window update =====================
        if len(win_right) == WINDOW_SAMPLES and len(win_left) == WINDOW_SAMPLES:
            if (now - last_window_emit_time) >= WINDOW_UPDATE_SEC:
                last_window_emit_time = now
                total_windows += 1

                A_r, A_l, D, z = compute_A_and_D(win_right, win_left)

                label_fixed = label_from_z(z, Z_LOW_FIXED, Z_HIGH_FIXED)

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
                            if (means[1] - means[0] >= MIN_CENTER_GAP) and (means[2] - means[1] >= MIN_CENTER_GAP):
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
                    z_low_final = float(ALPHA_FIXED * Z_LOW_FIXED  + (1 - ALPHA_FIXED) * z_low_gmm)
                    z_high_final = float(ALPHA_FIXED * Z_HIGH_FIXED + (1 - ALPHA_FIXED) * z_high_gmm)

                    if gmm is not None and np.isfinite(z):
                        try:
                            p = gmm.predict_proba(np.array([[z]], dtype=float))[0]
                            conf_str = f"{float(np.max(p)):.2f}"
                        except:
                            conf_str = "--"

                label_final = label_from_z(z, z_low_final, z_high_final)
                label_hist.append(label_final)
                label_smooth = majority_vote(list(label_hist))

                t_rel = now - start_time
                true_lab = true_label_from_time(t_rel)

                zlg = z_low_gmm if (z_low_gmm is not None) else ""
                zhg = z_high_gmm if (z_high_gmm is not None) else ""

                log_writer.writerow([
                    f"{now:.6f}", f"{t_rel:.3f}", total_windows,
                    f"{A_r:.6f}" if np.isfinite(A_r) else "",
                    f"{A_l:.6f}" if np.isfinite(A_l) else "",
                    f"{D:.6f}"   if np.isfinite(D) else "",
                    f"{z:.6f}"   if np.isfinite(z) else "",
                    f"{Z_LOW_FIXED:.6f}", f"{Z_HIGH_FIXED:.6f}",
                    zlg, zhg,
                    f"{z_low_final:.6f}", f"{z_high_final:.6f}",
                    1 if gmm_ready else 0,
                    conf_str,
                    true_lab,
                    label_final,
                    label_smooth
                ])

                update_confusion(true_lab, label_smooth)

                if (total_windows % 6) == 0:
                    try:
                        log_fp.flush()
                    except:
                        pass

                # -------- breathing rate display (dominant channel by A) --------
                freq_r = avg_freq_from_times(breath_times_r, now, BR_AVG_SEC)
                freq_l = avg_freq_from_times(breath_times_l, now, BR_AVG_SEC)

                if np.isfinite(A_r) and np.isfinite(A_l):
                    use_right = (A_r >= A_l)
                else:
                    use_right = True

                br_hz = freq_r if use_right else freq_l
                br_str = f"{br_hz:.2f} Hz" if np.isfinite(br_hz) else "--"

                Ar_str = f"{A_r:.1f}" if np.isfinite(A_r) else "--"
                Al_str = f"{A_l:.1f}" if np.isfinite(A_l) else "--"
                D_str  = f"{D:.3f}" if np.isfinite(D) else "--"

                info_text.set_text(
                    f"Breathing rate={br_str}   A_leftlung={Al_str}   A_rightlung={Ar_str}   D={D_str}   Class={label_smooth}"
                )

        # ===================== Plot update =====================
        sample_counter += 1
        if sample_counter >= UPDATE_INTERVAL:
            line_right.set_ydata(filt_right)
            line_left.set_ydata(filt_left)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            sample_counter = 0

        plt.pause(0.001)

        next_tick += target_dt
        sleep_s = next_tick - time.time()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_tick = time.time()

except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    safe_close(sock_right)
    safe_close(sock_left)

    try:
        if log_fp is not None:
            log_fp.flush()
            log_fp.close()
            print(f"[Log] Closed: {current_log_path}")
    except:
        pass

    save_confusion_matrix()