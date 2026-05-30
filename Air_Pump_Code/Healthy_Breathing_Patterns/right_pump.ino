// ================= Right Air Pump =================
// Schedule (4-min loop, 60s each):
// 0-60s     Balanced        -> ACTIVE
// 60-120s   Left-dominant   -> STOP
// 120-180s  Right-dominant  -> ACTIVE
// 180-240s  Apnea           -> STOP
//
// When STOP -> ACTIVE, do residual volume first.

const int PIN_INFLATE = 3;    // Solenoid valve
const int PIN_VALVE   = 4;    // Inflation pump
const int PIN_DEFLATE = 5;    // Deflation pump

const int INFLATE_SPEED = 255;
const int VALVE_SPEED   = 255;
const int DEFLATE_SPEED = 255;

// Breathing timing
const unsigned long INHALE_MS = 1300;
const unsigned long EXHALE_MS = 1250;

// Residual volume pulse when STOP -> ACTIVE
const unsigned long RESIDUAL_MS = 1000;

// Mode timing
const unsigned long MODE_MS  = 60000UL;   // 1 minute per mode
const unsigned long CYCLE_MS = 240000UL;  // 4 minutes total

enum Phase { PH_STOP, PH_RESIDUAL, PH_INHALE, PH_EXHALE };

unsigned long t0 = 0;
unsigned long phaseStart = 0;
Phase phase = PH_STOP;

bool lastActive = false;

void allStop() {
  analogWrite(PIN_INFLATE, 0);
  analogWrite(PIN_VALVE, 0);
  analogWrite(PIN_DEFLATE, 0);
}

void doResidual() {
  analogWrite(PIN_INFLATE, INFLATE_SPEED);
  analogWrite(PIN_VALVE, VALVE_SPEED);
  analogWrite(PIN_DEFLATE, 0);
}

void doInhale() {
  analogWrite(PIN_INFLATE, INFLATE_SPEED);
  analogWrite(PIN_VALVE, VALVE_SPEED);
  analogWrite(PIN_DEFLATE, 0);
}

void doExhale() {
  analogWrite(PIN_INFLATE, 0);
  analogWrite(PIN_VALVE, 0);
  analogWrite(PIN_DEFLATE, DEFLATE_SPEED);
}

bool isActiveRight(unsigned long tInCycle) {
  // Right ACTIVE during Balanced + Right-dominant
  // STOP during Left-dominant + Apnea
  if (tInCycle < 60000UL)  return true;   // Balanced
  if (tInCycle < 120000UL) return false;  // Left-dominant
  if (tInCycle < 180000UL) return true;   // Right-dominant
  return false;                           // Apnea
}

const char* modeName(int modeIdx) {
  switch (modeIdx) {
    case 0: return "Balanced";
    case 1: return "Left-dominant";
    case 2: return "Right-dominant";
    case 3: return "Apnea";
    default: return "Unknown";
  }
}

void setup() {
  pinMode(PIN_INFLATE, OUTPUT);
  pinMode(PIN_VALVE, OUTPUT);
  pinMode(PIN_DEFLATE, OUTPUT);

  Serial.begin(9600);
  Serial.println("RIGHT pump started");

  t0 = millis();
  phaseStart = millis();
  phase = PH_STOP;
  allStop();
}

void loop() {
  unsigned long now = millis();
  unsigned long tInCycle = (now - t0) % CYCLE_MS;

  int modeIdx = (int)((tInCycle / MODE_MS) % 4); // 0,1,2,3
  bool active = isActiveRight(tInCycle);

  // Detect STOP -> ACTIVE transition: trigger residual
  if (active && !lastActive) {
    phase = PH_RESIDUAL;
    phaseStart = now;
    Serial.print("[RIGHT] STOP->ACTIVE, residual. mode=");
    Serial.println(modeName(modeIdx));
  }

  // Detect ACTIVE -> STOP transition: stop immediately
  if (!active && lastActive) {
    phase = PH_STOP;
    phaseStart = now;
    Serial.print("[RIGHT] ACTIVE->STOP. mode=");
    Serial.println(modeName(modeIdx));
    allStop();
  }

  lastActive = active;

  if (!active) {
    allStop();
    phase = PH_STOP;
    return;
  }

  switch (phase) {
    case PH_RESIDUAL:
      doResidual();
      if (now - phaseStart >= RESIDUAL_MS) {
        phase = PH_INHALE;
        phaseStart = now;
      }
      break;

    case PH_INHALE:
      doInhale();
      if (now - phaseStart >= INHALE_MS) {
        phase = PH_EXHALE;
        phaseStart = now;
      }
      break;

    case PH_EXHALE:
      doExhale();
      if (now - phaseStart >= EXHALE_MS) {
        phase = PH_INHALE;
        phaseStart = now;
      }
      break;

    default:
      phase = PH_INHALE;
      phaseStart = now;
      break;
  }
}