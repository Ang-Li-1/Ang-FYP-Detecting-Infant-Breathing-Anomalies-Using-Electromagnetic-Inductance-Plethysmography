#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>

const char* ssid = "OPPO Find X8";
const char* password = "18653510219";

WiFiServer server(3333);
WiFiClient client;

const int sensorPin = 34;

WiFiUDP udp;
const uint16_t UDP_PORT = 5005;

// ================= Channel setting =================
// Rx1 samples when MUX level = 1
const int MY_LEVEL = 1;

// ================= Timing =================
// MUX: 20 Hz full cycle
// Tx1 on 25 ms, Tx2 on 25 ms
//
// Because UDP already introduces delay, do not wait too long.
// Sample soon after receiving the correct MUX-level packet.
const uint32_t SAMPLE_DELAY_US = 6000;     // 6 ms after receiving correct UDP packet

// If scheduled sample becomes too old, cancel it.
// This prevents sampling too close to the next MUX edge.
const uint32_t MAX_SAMPLE_AGE_US = 16000;  // packet receive time -> sample must be <16 ms

// Same Rx should sample once every 50 ms.
// Use 40 ms to reject duplicate/repeated packets but allow normal 50 ms sampling.
const uint32_t MIN_GAP_US = 40000;

// ================= State =================
uint32_t lastSampleUs = 0;

bool hasSeq = false;
uint32_t lastSeq = 0;

// Latest MUX state according to received UDP packets
int latestLevel = -1;
uint32_t latestPacketRxUs = 0;

// Scheduled sampling task
bool samplePending = false;
uint32_t scheduledSampleUs = 0;
uint32_t scheduledFromPacketUs = 0;
uint32_t scheduledSeq = 0;

bool parseMsg(const char* s, uint32_t &seq, uint32_t &t_us, int &level) {
  char *endp;

  seq = strtoul(s, &endp, 10);
  if (*endp != ',') return false;

  t_us = strtoul(endp + 1, &endp, 10);
  if (*endp != ',') return false;

  level = (int)strtol(endp + 1, &endp, 10);
  return true;
}

void handleUdpPacket() {
  char buf[96];

  int n = udp.read(buf, sizeof(buf) - 1);
  if (n <= 0) return;
  buf[n] = '\0';

  uint32_t seq, t_mux_us;
  int level;

  if (!parseMsg(buf, seq, t_mux_us, level)) return;

  // Reject stale / out-of-order packets
  if (hasSeq && (int32_t)(seq - lastSeq) <= 0) {
    return;
  }

  hasSeq = true;
  lastSeq = seq;

  uint32_t nowUs = micros();

  latestLevel = level;
  latestPacketRxUs = nowUs;

  // If opposite level arrives before scheduled sample, cancel the pending sample.
  // This is the key protection against sampling after the MUX has already switched off.
  if (samplePending && level != MY_LEVEL) {
    samplePending = false;
    return;
  }

  // Only schedule sample when this packet indicates our ON state
  if (level != MY_LEVEL) {
    return;
  }

  // Prevent duplicate sampling
  if ((uint32_t)(nowUs - lastSampleUs) < MIN_GAP_US) {
    return;
  }

  // Schedule a sample shortly after receiving the correct level
  samplePending = true;
  scheduledFromPacketUs = nowUs;
  scheduledSampleUs = nowUs + SAMPLE_DELAY_US;
  scheduledSeq = seq;
}

void tryDoScheduledSample() {
  if (!samplePending) return;

  uint32_t nowUs = micros();

  // Not time yet
  if ((int32_t)(nowUs - scheduledSampleUs) < 0) {
    return;
  }

  samplePending = false;

  // If too much time has passed since the packet was received, skip.
  // This avoids sampling close to the next 25 ms MUX boundary.
  uint32_t ageUs = nowUs - scheduledFromPacketUs;
  if (ageUs > MAX_SAMPLE_AGE_US) {
    return;
  }

  // If the latest received UDP level is no longer our ON state, skip.
  if (latestLevel != MY_LEVEL) {
    return;
  }

  // Final duplicate guard
  if ((uint32_t)(nowUs - lastSampleUs) < MIN_GAP_US) {
    return;
  }

  int adcValue = analogRead(sensorPin);
  uint32_t tLocal = micros();

  lastSampleUs = tLocal;

  if (client && client.connected()) {
    client.print((unsigned long)tLocal);
    client.print(",");
    client.println(adcValue);
  }
}

void setup() {
  Serial.begin(115200);

  analogReadResolution(12);

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);   // reduce Wi-Fi latency/jitter

  WiFi.begin(ssid, password);

  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nConnected to WiFi");
  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP());

  server.begin();
  udp.begin(UDP_PORT);

  Serial.println("Rx1 readout started");
}

void loop() {
  // Accept TCP client
  if (!client || !client.connected()) {
    WiFiClient newClient = server.available();
    if (newClient) {
      client = newClient;
      client.setNoDelay(true);
    }
  }

  // Read all available UDP packets first.
  // This prevents old packets from sitting in the buffer.
  int packetSize = udp.parsePacket();
  while (packetSize > 0) {
    handleUdpPacket();
    packetSize = udp.parsePacket();
  }

  // Then perform scheduled sample if still valid
  tryDoScheduledSample();
}