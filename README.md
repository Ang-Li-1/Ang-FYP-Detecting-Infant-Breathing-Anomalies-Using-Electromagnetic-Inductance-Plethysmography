# Ang FYP: Detecting Infant Breathing Anomalies Using Electromagnetic Inductance Plethysmography
Code and hardware design files for a knitted electromagnetic inductance plethysmography (EIP) system for wearable infant respiratory monitoring, including PCB designs, programmable air-pump control, embedded DSP, and machine-learning classification.

## Abstract

This repository contains the hardware design files and software implementation for a Final Year Project on knitted Electromagnetic Inductance Plethysmography (EIP) for wearable infant respiratory monitoring. The system uses deformable knitted Tx--Rx coils to capture breathing-induced electromagnetic coupling changes, allowing breathing waveforms, respiratory parameters, asymmetric breathing patterns, and spatial breathing information to be extracted.

The repository is organised into three main sections: PCB design files, programmable air-pump control code, and DSP / machine-learning code used for signal acquisition, processing, synchronisation, and respiratory classification.

## Repository Structure

```text
.
├── PCB_Design/
│   ├── Main_EIP_Readout_PCB/
│   └── Tunable_Bandpass_Filter_PCB/
├── Air_Pump_Code/
│   ├── Healthy_Breathing_Patterns/
│   └── Irregular_Breathing_Patterns/
├── DSP_Code/
│   ├── ESP32_Embedded_Code/
│   ├── Tx_MUX_Synchronisation/
│   └── Machine_Learning_Classification/
└── README.md
```

## 1. PCB Design

This section contains the PCB design files for the EIP readout and frequency-selective front-end circuits.

### 1.1 Main EIP Readout PCB

The main EIP readout PCB includes the mixed-signal readout electronics for acquiring the induced EMF envelope from the knitted Rx coil. It contains:

- analogue RF amplifier stage,
- peak detector for induced EMF envelope extraction,
- embedded ESP32 for ADC sampling, data transmission, and basic signal processing.

### 1.2 Tunable Bandpass Filter PCB

The tunable bandpass filter PCB provides a frequency-selective front-end whose centre frequency can be tuned over approximately **1--5 MHz** using ESP32 I/O control signals. The tuning strategy includes:

- coarse tuning using switched capacitor / inductor branches,
- fine tuning using varactor-based frequency adjustment.

## 2. Air Pump Code

This section contains the control code for generating repeatable breathing patterns using a programmable air-pump setup.

### 2.1 Healthy Breathing Patterns

The healthy breathing scripts generate regular breathing cycles. The delivered breathing volume is defined as:

```math
V = Q \times t_{\mathrm{inflation}}
```

where:

- `V` is the delivered breathing volume,
- `Q` is the air-flow rate,
- `t_inflation` is the inflation time.

### 2.2 Irregular Breathing Patterns

The irregular breathing scripts generate abnormal or non-uniform respiratory patterns for experimental validation, including:

- pneumonia-like breathing,
- bronchiolitis-like breathing,
- apnea or pause-containing breathing patterns,
- upper-airway-obstruction-like patterns.

## 3. DSP and Machine Learning Code

This section contains embedded and offline signal-processing code for EIP waveform extraction, respiratory parameter estimation, synchronisation, and classification.

### 3.1 ESP32 Embedded DSP Code

The ESP32 embedded code supports:

- ADC sampling,
- real-time breathing waveform acquisition,
- breathing rate estimation,
- basic breathing volume / amplitude feature extraction,
- wireless or serial data transmission.

### 3.2 Tx-Side MUX Synchronisation Code

The Tx-side MUX synchronisation code coordinates the readout electronics with time-division Tx excitation. This ensures that ADC sampling occurs during the Tx-on window and avoids sampling during the Tx-off interval.

### 3.3 Machine Learning Classification Code

The machine-learning scripts include model training and evaluation code for:

- **ResNet--LSTM** classification of respiratory waveform patterns,
- **Gaussian Mixture Model (GMM)** classification for two-channel asymmetric breathing detection,
- **Random Forest** classification for four-quadrant stitch-encoded spatial breathing classification.

## Notes

This repository is intended for research and development purposes. The system was developed and evaluated using infant dolls, phantoms, and laboratory test setups. It is not intended for clinical deployment without further safety testing, biocompatibility assessment, ethics approval, and clinical validation.
