#include <Adafruit_PWMServoDriver.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <math.h>
#include <string.h>

// Network mode:
// - Leave WIFI_SSID empty to make the ESP32-C3 create its own access point.
// - Fill WIFI_SSID/WIFI_PASSWORD to join your home/lab WiFi instead.
const char *WIFI_SSID = "";
const char *WIFI_PASSWORD = "";
const char *AP_SSID = "crankbot-esp32";
const char *AP_PASSWORD = "crankbot123";

const uint16_t UDP_PORT = 4210;
const uint32_t COMMAND_TIMEOUT_MS = 750;
const uint32_t SERVO_UPDATE_PERIOD_MS = 20;  // 50 Hz, matching hobby servos.

// Change these pins if your ESP32-C3 board exposes I2C elsewhere.
const int I2C_SDA_PIN = 8;
const int I2C_SCL_PIN = 9;
const uint8_t PCA9685_ADDRESS = 0x40;

Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(PCA9685_ADDRESS);
WiFiUDP udp;

struct ServoConfig {
  const char *name;
  uint8_t channel;
  uint16_t minUs;
  uint16_t centerUs;
  uint16_t maxUs;
  float minRpm;
  float maxRpm;
  bool invert;
};

// Joint order expected by the Python client:
// 0 front_left_shoulder
// 1 front_left_elbow
// 2 front_right_shoulder
// 3 front_right_elbow
// 4 back_left_shoulder
// 5 back_left_elbow
// 6 back_right_shoulder
// 7 back_right_elbow
//
// Initial pulse limits come from firmware/robo_dog_sketch.ino.
// Tune centerUs per joint after mounting the servos. Setpoint 0.0 should be
// the mechanically safe neutral/standing reference used by the PC client.
ServoConfig servos[8] = {
    {"front_left_shoulder", 8, 500, 1500, 2750, 2.0, 50.0, false},
    {"front_left_elbow", 9, 500, 1500, 2750, 2.0, 80.0, false},
    {"front_right_shoulder", 12, 500, 1500, 2500, 2.0, 50.0, false},
    {"front_right_elbow", 13, 500, 1500, 2750, 2.0, 80.0, false},
    {"back_left_shoulder", 10, 500, 1500, 2500, 2.0, 50.0, false},
    {"back_left_elbow", 11, 500, 1500, 2750, 2.0, 80.0, false},
    {"back_right_shoulder", 14, 500, 1500, 2500, 2.0, 50.0, false},
    {"back_right_elbow", 15, 500, 1500, 2750, 2.0, 80.0, false},
};

float currentUs[8];
float targetUs[8];
float lastSetpoint[8];
uint32_t lastCommandMs = 0;
uint32_t lastServoUpdateMs = 0;

float clampFloat(float value, float low, float high) {
  if (value < low) {
    return low;
  }
  if (value > high) {
    return high;
  }
  return value;
}

float setpointToPulseUs(const ServoConfig &servo, float setpoint) {
  float value = clampFloat(setpoint, -1.0, 1.0);
  if (servo.invert) {
    value = -value;
  }

  if (value >= 0.0) {
    return servo.centerUs + value * (servo.maxUs - servo.centerUs);
  }
  return servo.centerUs + value * (servo.centerUs - servo.minUs);
}

void setTargetsFromSetpoints(float setpoints[8]) {
  for (int i = 0; i < 8; ++i) {
    lastSetpoint[i] = clampFloat(setpoints[i], -1.0, 1.0);
    targetUs[i] = setpointToPulseUs(servos[i], lastSetpoint[i]);
  }
  lastCommandMs = millis();
}

void holdCurrentPosition() {
  for (int i = 0; i < 8; ++i) {
    targetUs[i] = currentUs[i];
  }
}

void writeAllServosNow() {
  for (int i = 0; i < 8; ++i) {
    pca.writeMicroseconds(servos[i].channel, (uint16_t)(currentUs[i] + 0.5));
  }
}

void updateServos() {
  const uint32_t now = millis();
  if (now - lastServoUpdateMs < SERVO_UPDATE_PERIOD_MS) {
    return;
  }

  const float dt = (now - lastServoUpdateMs) / 1000.0;
  lastServoUpdateMs = now;

  if (now - lastCommandMs > COMMAND_TIMEOUT_MS) {
    holdCurrentPosition();
  }

  for (int i = 0; i < 8; ++i) {
    const ServoConfig &servo = servos[i];
    const float delta = targetUs[i] - currentUs[i];
    const float absDelta = fabs(delta);
    if (absDelta < 0.5) {
      currentUs[i] = targetUs[i];
    } else {
      const float pulseRangeUs = servo.maxUs - servo.minUs;
      float maxStepUs = servo.maxRpm * dt * pulseRangeUs / 30.0;
      float minStepUs = servo.minRpm * dt * pulseRangeUs / 30.0;
      maxStepUs = max(maxStepUs, 1.0f);
      minStepUs = min(minStepUs, maxStepUs);

      float stepUs = min(absDelta, maxStepUs);
      if (absDelta > minStepUs) {
        stepUs = max(stepUs, minStepUs);
      }
      currentUs[i] += (delta > 0.0 ? stepUs : -stepUs);
    }
    pca.writeMicroseconds(servo.channel, (uint16_t)(currentUs[i] + 0.5));
  }
}

bool parseSetCommand(char *packet, float values[8]) {
  char *cursor = packet;
  while (*cursor == ' ' || *cursor == '\t') {
    ++cursor;
  }

  if (strncmp(cursor, "SET", 3) == 0) {
    cursor += 3;
  }

  for (int i = 0; i < 8; ++i) {
    while (*cursor == ' ' || *cursor == '\t' || *cursor == ',') {
      ++cursor;
    }
    if (*cursor == '\0' || *cursor == '\r' || *cursor == '\n') {
      return false;
    }

    char *endPtr = nullptr;
    values[i] = strtof(cursor, &endPtr);
    if (endPtr == cursor) {
      return false;
    }
    cursor = endPtr;
  }
  return true;
}

void sendStatus(IPAddress remoteIp, uint16_t remotePort, const char *prefix) {
  udp.beginPacket(remoteIp, remotePort);
  udp.print(prefix);
  udp.print(" ip=");
  udp.print(WiFi.getMode() == WIFI_AP ? WiFi.softAPIP() : WiFi.localIP());
  udp.print(" setpoints=");
  for (int i = 0; i < 8; ++i) {
    if (i > 0) {
      udp.print(",");
    }
    udp.print(lastSetpoint[i], 3);
  }
  udp.print(" pulses=");
  for (int i = 0; i < 8; ++i) {
    if (i > 0) {
      udp.print(",");
    }
    udp.print((int)(currentUs[i] + 0.5));
  }
  udp.endPacket();
}

void handleUdp() {
  char packet[256];
  const int packetSize = udp.parsePacket();
  if (packetSize <= 0) {
    return;
  }

  const int len = udp.read(packet, min(packetSize, (int)sizeof(packet) - 1));
  packet[len] = '\0';

  IPAddress remoteIp = udp.remoteIP();
  uint16_t remotePort = udp.remotePort();

  if (strncmp(packet, "PING", 4) == 0 || strncmp(packet, "STATUS", 6) == 0) {
    sendStatus(remoteIp, remotePort, "OK");
    return;
  }

  if (strncmp(packet, "STOP", 4) == 0) {
    holdCurrentPosition();
    lastCommandMs = millis();
    sendStatus(remoteIp, remotePort, "OK");
    return;
  }

  if (strncmp(packet, "CENTER", 6) == 0) {
    float zeros[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    setTargetsFromSetpoints(zeros);
    sendStatus(remoteIp, remotePort, "OK");
    return;
  }

  float values[8];
  if (parseSetCommand(packet, values)) {
    setTargetsFromSetpoints(values);
    sendStatus(remoteIp, remotePort, "OK");
  } else {
    udp.beginPacket(remoteIp, remotePort);
    udp.print("ERR expected: SET v0 v1 v2 v3 v4 v5 v6 v7");
    udp.endPacket();
  }
}

void setupWiFi() {
  if (strlen(WIFI_SSID) > 0) {
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.print("Joining WiFi");
    while (WiFi.status() != WL_CONNECTED) {
      delay(500);
      Serial.print(".");
    }
    Serial.println();
    Serial.print("WiFi IP: ");
    Serial.println(WiFi.localIP());
  } else {
    WiFi.mode(WIFI_AP);
    WiFi.softAP(AP_SSID, AP_PASSWORD);
    Serial.print("Access point: ");
    Serial.println(AP_SSID);
    Serial.print("AP IP: ");
    Serial.println(WiFi.softAPIP());
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);

  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  pca.begin();
  pca.setPWMFreq(50);
  delay(10);

  for (int i = 0; i < 8; ++i) {
    currentUs[i] = servos[i].centerUs;
    targetUs[i] = servos[i].centerUs;
    lastSetpoint[i] = 0.0;
  }
  writeAllServosNow();

  setupWiFi();
  udp.begin(UDP_PORT);
  lastCommandMs = millis();
  lastServoUpdateMs = millis();

  Serial.print("UDP listening on port ");
  Serial.println(UDP_PORT);
}

void loop() {
  handleUdp();
  updateServos();
}
