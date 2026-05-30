#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include "DFRobot_BNO055.h"

// ================= Wi-Fi =================
const char* ssid = "OPPO Find X8";
const char* password = "18653510219";

// ================= CTRL waveform =================
const int MUX_CTRL_GPIO = 25;
const uint32_t HALF_PERIOD_MS = 25;   // 25ms + 25ms = 50ms => 20Hz

// ================= UDP sync =================
WiFiUDP udpSync;
WiFiUDP udpImu;

const uint16_t UDP_SYNC_PORT = 5005;  // MUX sync
const uint16_t UDP_IMU_PORT  = 5006;  // IMU stream

IPAddress broadcastIP(255, 255, 255, 255);

uint32_t seq = 0;

// ================= BNO055 =================
typedef DFRobot_BNO055_IIC BNO;
BNO bno(&Wire, 0x28);

// ================= Timing =================
uint32_t lastMuxToggleMs = 0;
uint32_t lastImuReadMs = 0;
uint32_t lastWiFiCheckMs = 0;

int muxLevel = LOW;

const uint32_t IMU_PERIOD_MS = 40;    // 25 Hz
const uint32_t WIFI_CHECK_MS = 1000;

// ================= Status printing =================
void printLastOperateStatus(BNO::eStatus_t status)
{
  switch (status) {
    case BNO::eStatusOK:
      Serial.println("everything ok");
      break;
    case BNO::eStatusErr:
      Serial.println("unknown error");
      break;
    case BNO::eStatusErrDeviceNotDetect:
      Serial.println("device not detected");
      break;
    case BNO::eStatusErrDeviceReadyTimeOut:
      Serial.println("device ready timeout");
      break;
    case BNO::eStatusErrDeviceStatus:
      Serial.println("device internal status error");
      break;
    default:
      Serial.println("unknown status");
      break;
  }
}

// ================= Wi-Fi reconnect =================
void connectWiFi()
{
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);   // important: reduce Wi-Fi latency/jitter

  WiFi.begin(ssid, password);

  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected.");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());
}

void checkWiFiReconnect()
{
  uint32_t now = millis();

  if (now - lastWiFiCheckMs < WIFI_CHECK_MS) {
    return;
  }

  lastWiFiCheckMs = now;

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] disconnected, reconnecting...");

    WiFi.disconnect();
    delay(100);

    WiFi.begin(ssid, password);

    uint32_t tStart = millis();

    while (WiFi.status() != WL_CONNECTED && millis() - tStart < 5000) {
      delay(200);
      Serial.print(".");
    }

    if (WiFi.status() == WL_CONNECTED) {
      Serial.println("\n[WiFi] reconnected.");
      Serial.print("IP: ");
      Serial.println(WiFi.localIP());
    } else {
      Serial.println("\n[WiFi] reconnect failed, will retry.");
    }
  }
}

// ================= Send MUX sync =================
void sendSync(int level)
{
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }

  char msg[64];
  uint32_t t = micros();

  int n = snprintf(
    msg,
    sizeof(msg),
    "%lu,%lu,%d",
    (unsigned long)seq++,
    (unsigned long)t,
    level
  );

  udpSync.beginPacket(broadcastIP, UDP_SYNC_PORT);
  udpSync.write((const uint8_t*)msg, n);
  udpSync.endPacket();
}

// ================= Send IMU via UDP =================
void sendImu(float pitch, float roll, float yaw)
{
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }

  char msg[96];
  uint32_t t = micros();

  int n = snprintf(
    msg,
    sizeof(msg),
    "%lu,%.3f,%.3f,%.3f",
    (unsigned long)t,
    pitch,
    roll,
    yaw
  );

  udpImu.beginPacket(broadcastIP, UDP_IMU_PORT);
  udpImu.write((const uint8_t*)msg, n);
  udpImu.endPacket();
}

void setup()
{
  Serial.begin(115200);

  // ---------- MUX control ----------
  pinMode(MUX_CTRL_GPIO, OUTPUT);
  digitalWrite(MUX_CTRL_GPIO, LOW);
  muxLevel = LOW;

  // ---------- I2C for BNO055 ----------
  Wire.begin(21, 22);   // SDA = IO21, SCL = IO22
  Wire.setClock(100000);

  delay(500);

  bno.reset();

  while (bno.begin() != BNO::eStatusOK) {
    Serial.println("BNO055 begin failed");
    printLastOperateStatus(bno.lastOperateStatus);
    delay(1000);
  }

  Serial.println("BNO055 begin success");

  // ---------- Wi-Fi ----------
  connectWiFi();

  udpSync.begin(UDP_SYNC_PORT);
  udpImu.begin(UDP_IMU_PORT);

  Serial.println("MUX + IMU UDP streamer started");
  Serial.println("MUX UDP: seq,t_us,level on port 5005");
  Serial.println("IMU UDP: t_us,pitch,roll,yaw on port 5006");
}

void loop()
{
  checkWiFiReconnect();

  uint32_t nowMs = millis();

  // ---------- 20 Hz MUX CTRL waveform ----------
  if (nowMs - lastMuxToggleMs >= HALF_PERIOD_MS) {
    lastMuxToggleMs = nowMs;

    muxLevel = !muxLevel;
    digitalWrite(MUX_CTRL_GPIO, muxLevel);

    sendSync(muxLevel);
  }

  // ---------- BNO055 readout at 25 Hz ----------
  if (nowMs - lastImuReadMs >= IMU_PERIOD_MS) {
    lastImuReadMs = nowMs;

    BNO::sEulAnalog_t eul = bno.getEul();

    float pitch = eul.pitch;
    float roll  = eul.roll;
    float yaw   = eul.head;

    // Send IMU to Python by UDP instead of USB Serial
    sendImu(pitch, roll, yaw);

    // Optional debug only.
    // Do not rely on Serial for Python IMU input.
    Serial.print(pitch, 3);
    Serial.print(",");
    Serial.print(roll, 3);
    Serial.print(",");
    Serial.println(yaw, 3);
  }
}