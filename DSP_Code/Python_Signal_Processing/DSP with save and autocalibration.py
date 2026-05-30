import matplotlib.pyplot as plt
import socket
import collections
import time
import numpy as np
from scipy.signal import butter, lfilter
import datetime
import os

# --- Configuration ---
ESP32_IP = '10.67.202.110'  # Replace with your ESP32's IP
# ESP32_IP = '192.168.116.186'
PORT = 3333                     # Match the port used in your ESP32 code
BUFFER_LEN = 200
UPDATE_INTERVAL = 5
MIN_BREATH_DELTA = 20
MIN_BREATH_DURATION = 0.5
MAX_BREATH_DURATION = 10.0
FS = 20.0
CUTOFF = 1.5
FILTER_ORDER = 2

# --- Butterworth filter setup ---
def butter_lowpass(cutoff, fs, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='low', analog=False)

b, a = butter_lowpass(CUTOFF, FS, FILTER_ORDER)
zi = np.zeros(max(len(a), len(b)) - 1)

# --- Trough Suppression Function ---
def suppress_trough(signal_list, window_size=5, threshold=300):
    if len(signal_list) < window_size:
        return signal_list[-1]
    window = list(signal_list)[-window_size:]
    median = np.median(window)
    last_val = signal_list[-1]
    if median - last_val > threshold:
        return int(median)
    return last_val

# =========================
# Auto-calibration functions
# =========================

# Physiology priors (soft bounds / priors, can be tuned later)
# NOTE: Updated infant_2y RR range to 20–30 bpm (no overlap).
AGE_CLASSES = [
    {
        "name": "newborn",
        "rr_bpm_range": (40, 60),
        "vt_ml_per_kg_range": (4, 6),
        "mass_kg": 2
    },
    {
        "name": "infant_1y",
        "rr_bpm_range": (20, 40),
        "vt_ml_per_kg_range": (5, 7),
        "mass_kg": 9.5
    },
    {
        "name": "infant_2y",
        "rr_bpm_range": (10, 20),
        "vt_ml_per_kg_range": (6, 8),
        "mass_kg": 12.5
    }
]

def infer_age_class(rr_bpm: float) -> dict:
    # Prefer interval membership (no overlap now).
    for cls in AGE_CLASSES:
        lo, hi = cls["rr_bpm_range"]
        if lo <= rr_bpm <= hi:
            return cls

    # Fallback: closest to midpoint
    best = None
    best_dist = float("inf")
    for cls in AGE_CLASSES:
        lo, hi = cls["rr_bpm_range"]
        mid = 0.5 * (lo + hi)
        dist = abs(rr_bpm - mid)
        if dist < best_dist:
            best_dist = dist
            best = cls
    return best

# --- Simulator validation mode ---
SIMULATOR_VALIDATION_MODE = True
PERIOD_REF_S = 1.20  # kept (not used in pump-based Vref below, but harmless)
EPS = 1e-6

def get_volume_bounds_ml(rr_bpm: float, alpha: float = 0.0):
    """
    Construct Vmin(fb), Vmax(fb) in mL (== cm^3).
    Base bounds: VT (mL/kg) * mass (kg).
    Optional weak intra-class RR modulation via alpha (keep small; alpha=0 disables).
    """
    cls = infer_age_class(rr_bpm)
    vt_lo, vt_hi = cls["vt_ml_per_kg_range"]
    m = cls["mass_kg"]

    Vmin0 = vt_lo * m
    Vmax0 = vt_hi * m

    rr_lo, rr_hi = cls["rr_bpm_range"]
    rr_ref = 0.5 * (rr_lo + rr_hi)

    scale = 1.0 + alpha * ((rr_bpm - rr_ref) / max(rr_ref, EPS))

    Vmin = max(0.0, Vmin0 * scale)
    Vmax = max(Vmin + EPS, Vmax0 * scale)
    return Vmin, Vmax, rr_ref

def is_stable_segment(breaths, min_breaths: int = 5,
                      cv_dur_max: float = 0.20, cv_amp_max: float = 0.35) -> bool:
    """
    Stable segment = low variability in duration and amplitude.
    Uses coefficient of variation (std/mean) on duration and amplitude features.
    """
    if len(breaths) < min_breaths:
        return False

    durs = np.array([b["dur_s"] for b in breaths], dtype=float)
    amps = np.array([b["amp"] for b in breaths], dtype=float)

    mean_d = max(durs.mean(), EPS)
    mean_a = max(amps.mean(), EPS)

    cv_d = durs.std(ddof=1) / mean_d if len(durs) > 1 else 0.0
    cv_a = amps.std(ddof=1) / mean_a if len(amps) > 1 else 0.0

    return (cv_d <= cv_dur_max) and (cv_a <= cv_amp_max)

def update_k_infant(breaths, rr_bpm: float, alpha: float = 0.0):
    """
    Closed-form estimate + clipping.

    SIMULATOR VALIDATION MODE:
      Use pump reference VT:
        V_pump = FLOW_CM3_PER_S * (period/2)
      where period = 60/rr_bpm.
    """
    if len(breaths) == 0:
        return None

    Vmin, Vmax, _ = get_volume_bounds_ml(rr_bpm, alpha=alpha)
    V_ref = 0.5 * (Vmin + Vmax)

    if SIMULATOR_VALIDATION_MODE:
        period = 60.0 / max(rr_bpm, EPS)

        # Set this to your pump flow in cm^3/s:
        # 3.5 L/min -> 3500/60 = 58.33 cm^3/s
        #FLOW_CM3_PER_S = 20 # for newborn
        FLOW_CM3_PER_S = 68.0
        V_pump = FLOW_CM3_PER_S * (period / 2.0)   # cm^3 == mL

        Vref = V_pump
    else:
        Vref = V_ref

    amps = np.array([b["amp"] for b in breaths], dtype=float)
    a_bar = max(amps.mean(), EPS)

    k_hat = Vref / a_bar

    # Conservative feasible interval for k across breaths:
    lower = np.max(Vmin / np.maximum(amps, EPS))
    upper = np.min(Vmax / np.maximum(amps, EPS))

    if lower > upper:
        return float(np.clip(k_hat, 0.1 * k_hat, 10.0 * k_hat))

    return float(np.clip(k_hat, lower, upper))

def estimate_volume_autocal(amp: float, dur_s: float,
                            breaths, k_state: dict,
                            alpha: float = 0.0):
    """
    NOTE: rr_bpm is derived from the *display-corrected* frequency:
      freq = (1/duration) * 0.6
      rr_bpm = freq * 60
    """
    # Keep your ADC delay correction consistent with displayed frequency
    freq_hz = (1.0 / max(dur_s, EPS)) * 0.6
    rr_bpm = freq_hz * 60.0

    breaths.append({"amp": float(amp), "dur_s": float(dur_s), "rr_bpm": float(rr_bpm), "t": time.time()})

    now = time.time()
    k_new = None
    if is_stable_segment(breaths) and (now - k_state["last_update_t"] > k_state["min_update_interval_s"]):
        k_new = update_k_infant(breaths, rr_bpm, alpha=alpha)
        if k_new is not None:
            first_time = (k_state["k_infant"] is None)
            k_state["k_infant"] = k_new
            k_state["last_update_t"] = now
            if first_time and (not k_state["calibrated"]):
                k_state["calibrated"] = True
                cls = infer_age_class(rr_bpm)
                mode_str = "SIMULATOR_VALIDATION_MODE" if SIMULATOR_VALIDATION_MODE else "PHYSIOLOGY_MODE"
                print(
                    f"[AutoCal] Calibration completed. "
                    f"k_infant = {k_new:.6f}, "
                    f"determined class = {cls['name']}, "
                    f"mode = {mode_str}"
                )

    k = k_state.get("k_infant", None)
    volume_ml = (k * float(amp)) if (k is not None) else None
    return volume_ml, rr_bpm, k_new

# --- Socket Setup ---
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((ESP32_IP, PORT))
sock_file = sock.makefile('r')  # Buffered reader

# --- Buffers ---
raw_data = collections.deque([0]*BUFFER_LEN, maxlen=BUFFER_LEN)
pre_filtered_data = collections.deque([0]*BUFFER_LEN, maxlen=BUFFER_LEN)
filtered_data = collections.deque([0]*BUFFER_LEN, maxlen=BUFFER_LEN)

# === Added for 5-minute capture ===
N_SAVE_SAMPLES = int(5 * 60 * FS)            # 5 minutes @ 20 Hz = 6000 samples
capture_buffer = collections.deque(maxlen=N_SAVE_SAMPLES)  # 存最近5分钟的滤波后信号
timestamp_buffer = collections.deque(maxlen=N_SAVE_SAMPLES) # 对应时间戳（秒）
last_autosave_time = time.time()
SAVE_DIR = "breathing_captures"
os.makedirs(SAVE_DIR, exist_ok=True)

def save_five_minute_segment(sig_arr, ts_arr, fs, prefix="breathing"):
    if len(sig_arr) == 0:
        return None, None
    # 对齐时间，让起点从0开始
    t0 = ts_arr[0]
    ts_rel = np.array(ts_arr) - t0
    sig_arr = np.array(sig_arr, dtype=float)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{prefix}_{stamp}_fs{int(fs)}Hz_5min"
    npy_path = os.path.join(SAVE_DIR, base + ".npy")
    csv_path = os.path.join(SAVE_DIR, base + ".csv")

    # 保存为 NPY（仅信号）和 CSV（含时间戳和信号）
    np.save(npy_path, sig_arr)
    csv_data = np.column_stack([ts_rel, sig_arr])
    np.savetxt(csv_path, csv_data, delimiter=",", header="time_s,signal", comments="")

    return npy_path, csv_path
# === end added ===

# --- Plot Setup ---
plt.ion()
fig1, ax1 = plt.subplots()
line1, = ax1.plot(range(BUFFER_LEN), list(raw_data))
ax1.set_ylim(2500, 4000)
ax1.set_xlim(0, BUFFER_LEN)
ax1.set_title("Original Breathing Signal (ADC)")
ax1.set_xlabel("Samples")
ax1.set_ylabel("ADC Value")

fig2, ax2 = plt.subplots()
line2, = ax2.plot(range(BUFFER_LEN), list(filtered_data))
ax2.set_ylim(2500, 4000)
ax2.set_xlim(0, BUFFER_LEN)
ax2.set_title("Filtered Breathing Signal")
ax2.set_xlabel("Samples")
ax2.set_ylabel("ADC Value")
info_text = ax2.text(0.5, 1.08, "", transform=ax2.transAxes, ha='center', fontsize=12, color='blue')

# --- Breathing Detection ---
state = "WAIT_MIN"
min_val = 4096
max_val = 0
t_min = 0
t_max = 0
sample_counter = 0

# --- Auto-calibration state ---
BREATH_HISTORY_LEN = 20
breath_hist = collections.deque(maxlen=BREATH_HISTORY_LEN)
k_state = {
    "k_infant": None,
    "last_update_t": 0.0,
    "min_update_interval_s": 15.0,
    "calibrated": False
}
ALPHA_RR_MOD = 0.0  # keep 0.0 by default; set 0.1–0.3 only after validation

try:
    # --- Main Loop ---
    while True:
        try:
            val_str = sock_file.readline().strip()
            if not val_str.isdigit():
                continue
            val = int(val_str)
            raw_data.append(val)

            # Trough suppression
            pre_val = suppress_trough(raw_data)
            pre_filtered_data.append(pre_val)

            # Low-pass filter
            filtered_array, zi = lfilter(b, a, [pre_val], zi=zi)
            filtered_val = int(filtered_array[-1])
            filtered_data.append(filtered_val)

            # === Added for 5-minute capture (append to buffers) ===
            now = time.time()
            capture_buffer.append(filtered_val)
            timestamp_buffer.append(now)
            # 满 5 分钟就自动保存一次；之后每过 5 分钟再次保存最近 5 分钟
            if len(capture_buffer) == N_SAVE_SAMPLES and (now - last_autosave_time) >= 300:
                npy_path, csv_path = save_five_minute_segment(list(capture_buffer), list(timestamp_buffer), FS)
                if csv_path:
                    print(f"[AutoSave] Saved 5-minute segment:\n  {npy_path}\n  {csv_path}")
                last_autosave_time = now
            # === end added ===

            # Breathing detection
            if state == "WAIT_MIN":
                if filtered_val < min_val:
                    min_val = filtered_val
                    t_min = now
                elif filtered_val - min_val > MIN_BREATH_DELTA:
                    max_val = filtered_val
                    t_max = now
                    state = "WAIT_MAX"
            elif state == "WAIT_MAX":
                if filtered_val > max_val:
                    max_val = filtered_val
                    t_max = now
                elif max_val - filtered_val > MIN_BREATH_DELTA:
                    duration = now - t_min
                    if MIN_BREATH_DURATION <= duration <= MAX_BREATH_DURATION:
                        # Keep your delay correction exactly as requested
                        freq = 1.0 / duration * 0.6

                        # ===== Auto-calibrated volume estimation (replaces original linear scaling) =====
                        amp = (max_val - min_val)  # a_hat_i feature
                        volume_ml, rr_bpm, _ = estimate_volume_autocal(
                            amp=amp,
                            dur_s=duration,
                            breaths=breath_hist,
                            k_state=k_state,
                            alpha=ALPHA_RR_MOD
                        )

                        if volume_ml is None:
                            info_text.set_text(f"Breathing Rate: {freq:.3f} Hz   Volume: calibrating...")
                        else:
                            # 1 mL == 1 cm^3
                            info_text.set_text(f"Breathing Rate: {freq:.3f} Hz   Volume: {volume_ml:.1f} cm³")
                        # ============================================================================

                    state = "WAIT_MIN"
                    min_val = 4096
                    max_val = 0
                    t_min = now

            if now - t_min > MAX_BREATH_DURATION:
                state = "WAIT_MIN"
                min_val = 4096
                max_val = 0
                t_min = now

            # Plot update
            sample_counter += 1
            if sample_counter >= UPDATE_INTERVAL:
                line1.set_ydata(raw_data)
                line2.set_ydata(filtered_data)
                fig1.canvas.draw()
                fig1.canvas.flush_events()
                fig2.canvas.draw()
                fig2.canvas.flush_events()
                sample_counter = 0

        except KeyboardInterrupt:
            print("Stopped by user.")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

finally:
    # === Added for 5-minute capture: exit-time save ===
    if len(capture_buffer) > 0:
        npy_path, csv_path = save_five_minute_segment(list(capture_buffer), list(timestamp_buffer), FS)
        if csv_path:
            print(f"[ExitSave] Saved latest segment:\n  {npy_path}\n  {csv_path}")
    try:
        sock_file.close()
    except:
        pass
    try:
        sock.close()
    except:
        pass
