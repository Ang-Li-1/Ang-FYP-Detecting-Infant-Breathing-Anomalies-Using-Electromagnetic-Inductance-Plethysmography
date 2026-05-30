#include <WiFi.h>

// Wi-Fi Credentials
const char* ssid = "OPPO Find X8";
const char* password = "18653510219";

// TCP Server on port 3333
WiFiServer server(3333);

const int sensorPin = 34;
const int samplingRate = 20;  // Hz
const int intervalMicros = 1000000 / samplingRate;

WiFiClient client;
unsigned long lastSampleTime = 0;

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);  // 0–4095

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected to WiFi");
  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP());

  // Start server
  server.begin();
}

void loop() {
  // Accept new client if not already connected
  if (!client || !client.connected()) {
    client = server.available();
  }

  // If client is connected, send data at fixed interval
  if (client && client.connected()) {
    unsigned long now = micros();
    if (now - lastSampleTime >= intervalMicros) {
      lastSampleTime = now;
      int adcValue = analogRead(sensorPin);
      client.println(adcValue);  // Send clean ADC value line
    }
  }
}
