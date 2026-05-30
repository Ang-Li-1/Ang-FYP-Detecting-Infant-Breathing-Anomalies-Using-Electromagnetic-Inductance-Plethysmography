#include <Arduino.h>
#include <math.h>

// =====================================================
// Pin definition
// =====================================================
const int VMAG_PIN = 36;   // ESP32 A0 / GPIO36
const int VPHS_PIN = 39;   // ESP32 A1 / GPIO39

// =====================================================
// Circuit parameters
// =====================================================
const float Rs = 47.0;     // sense resistor in ohms

// =====================================================
// AD8302 nominal parameters
// =====================================================
const float VMAG_CENTER = 0.9;   // 0 dB nominal output
const float VMAG_SLOPE  = 0.03;  // 30 mV/dB

const float VPHS_SLOPE  = 0.01;  // 10 mV/degree

// =====================================================
// Updated calibration from pure resistor measurements
// Supply voltage: 4.8 V
// Calibration resistors: 22, 47, 98, 214, 327 ohm
// =====================================================

// Gain calibration:
// Gain_cal = GAIN_M * Gain_raw + GAIN_C
const float GAIN_M = 0.956056;
const float GAIN_C = -0.984993;

// Phase zero reference
// From the new pure resistor calibration data
const float VPHS_ZERO = 1.895006;

// Phase error calibration:
// Phase_error = PHASE_ERR_M * Gain_raw + PHASE_ERR_C
// Phase_cal = Phase_raw - Phase_error
const float PHASE_ERR_M = 0.551849;
const float PHASE_ERR_C = -5.153245;

// Optional ADC voltage correction.
// If ESP32 ADC voltage is different from multimeter reading,
// tune these scale factors.
float VMAG_SCALE = 1.0;
float VPHS_SCALE = 1.0;

// =====================================================
// Averaging settings
// =====================================================
const int NUM_SAMPLES = 100;
const int SAMPLE_DELAY_MS = 2;

// =====================================================
// Helper function: read averaged voltage
// =====================================================
float readVoltageAverage(int pin) {
  uint32_t sum_mV = 0;

  for (int i = 0; i < NUM_SAMPLES; i++) {
    sum_mV += analogReadMilliVolts(pin);
    delay(SAMPLE_DELAY_MS);
  }

  float avg_mV = sum_mV / (float)NUM_SAMPLES;
  return avg_mV / 1000.0;
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  analogReadResolution(12);

  // AD8302 output is around 0–1.8 V.
  // ADC_6db is suitable for this range.
  analogSetPinAttenuation(VMAG_PIN, ADC_6db);
  analogSetPinAttenuation(VPHS_PIN, ADC_6db);

  Serial.println();
  Serial.println("AD8302-based calibrated AC resistance meter started");
  Serial.println("Supply = 4.8 V");
  Serial.println("Rs = 47 ohm");

  Serial.println();
  Serial.println("Calibration parameters:");
  Serial.print("GAIN_M = ");
  Serial.println(GAIN_M, 6);
  Serial.print("GAIN_C = ");
  Serial.println(GAIN_C, 6);
  Serial.print("VPHS_ZERO = ");
  Serial.print(VPHS_ZERO, 6);
  Serial.println(" V");
  Serial.print("PHASE_ERR_M = ");
  Serial.println(PHASE_ERR_M, 6);
  Serial.print("PHASE_ERR_C = ");
  Serial.println(PHASE_ERR_C, 6);

  Serial.println();
  Serial.println("VMAG(V), VPHS(V), Gain_raw(dB), Gain_cal(dB), Ratio, Phase_raw(deg), Phase_cal(deg), R_total(ohm), X_total(ohm), R_ac(ohm), X_coil(ohm)");
}

void loop() {
  // =====================================================
  // Read AD8302 outputs
  // =====================================================
  float VMAG = readVoltageAverage(VMAG_PIN) * VMAG_SCALE;
  float VPHS = readVoltageAverage(VPHS_PIN) * VPHS_SCALE;

  // =====================================================
  // Raw gain from AD8302
  // =====================================================
  float gain_dB_raw = (VMAG - VMAG_CENTER) / VMAG_SLOPE;

  // Calibrated gain
  float gain_dB_cal = GAIN_M * gain_dB_raw + GAIN_C;

  // |Va / Vb|
  float ratio = pow(10.0, gain_dB_cal / 20.0);

  // =====================================================
  // Raw phase from AD8302
  // =====================================================
  // Pure resistor reference is phase = 0 degree
  float phase_deg_raw = (VPHS_ZERO - VPHS) / VPHS_SLOPE;

  // Phase error estimated from pure resistor calibration
  float phase_error = PHASE_ERR_M * gain_dB_raw + PHASE_ERR_C;

  // Calibrated phase
  float phase_deg_cal = phase_deg_raw - phase_error;

  float phase_rad = phase_deg_cal * PI / 180.0;

  // =====================================================
  // Impedance calculation
  // =====================================================
  // Z_total = Rs * (Va/Vb) * exp(j*phi)
  float Z_total_mag = Rs * ratio;

  float R_total = Z_total_mag * cos(phase_rad);
  float X_total = Z_total_mag * sin(phase_rad);

  // Z_coil = Z_total - Rs
  float R_ac = R_total - Rs;
  float X_coil = X_total;

  // =====================================================
  // Print results
  // =====================================================
  Serial.print(VMAG, 4);
  Serial.print(", ");

  Serial.print(VPHS, 4);
  Serial.print(", ");

  Serial.print(gain_dB_raw, 3);
  Serial.print(", ");

  Serial.print(gain_dB_cal, 3);
  Serial.print(", ");

  Serial.print(ratio, 4);
  Serial.print(", ");

  Serial.print(phase_deg_raw, 3);
  Serial.print(", ");

  Serial.print(phase_deg_cal, 3);
  Serial.print(", ");

  Serial.print(R_total, 3);
  Serial.print(", ");

  Serial.print(X_total, 3);
  Serial.print(", ");

  Serial.print(R_ac, 3);
  Serial.print(", ");

  Serial.println(X_coil, 3);

  delay(500);
}