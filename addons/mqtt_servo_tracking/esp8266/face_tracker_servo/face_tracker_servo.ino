#define USE_US_TIMER
#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>

// Wi-Fi settings
const char* WIFI_SSID = "MW40V_11E5";
const char* WIFI_PASSWORD = "raphael159";

// MQTT settings
const char* MQTT_SERVER = "157.173.101.159";
const uint16_t MQTT_PORT = 1883;
const char* MQTT_TOPIC = "vision/faithlock/movement";

// Servo configuration
const uint8_t SERVO_PIN = 14; // D5
const int SERVO_MIN_ANGLE = 0;
const int SERVO_MAX_ANGLE = 180;
const int SERVO_CENTER_ANGLE = 90;
const int TRACK_STEP = 4;             // degrees moved per LEFT/RIGHT command (higher = faster tracking)
const int SEARCH_STEP = 1;            // 1-degree steps = smooth sweep (no jerky jumps)
const unsigned long TRACK_INTERVAL_MS = 50;
const unsigned long SEARCH_INTERVAL_MS = 15; // 1 deg / 15 ms = ~65 deg/s sweep (180 in ~2.8s)
const unsigned long COMMAND_TIMEOUT_MS = 1500;
const unsigned long IDLE_DETACH_MS = 700;    // detach PWM after idle to stop trembling

const bool REVERSE_SERVO = true;

enum MovementCommand {
  CMD_IDLE,
  CMD_LEFT,
  CMD_RIGHT,
  CMD_CENTER,
  CMD_SEARCH
};

WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);
Servo panServo;

char mqttClientId[28] = {0};

MovementCommand currentCommand = CMD_IDLE;
int servoAngle = SERVO_CENTER_ANGLE;
int sweepDirection = 1;
unsigned long lastMoveAt = 0;
unsigned long lastReconnectAttempt = 0;
unsigned long reconnectBackoffMs = 3000;
unsigned long lastCommandAt = 0;
bool wifiReady = false;

// --- Core Logic ---

void setServoAngle(int angle) {
  angle = constrain(angle, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE);
  bool reattached = false;
  if (!panServo.attached()) {
    panServo.attach(SERVO_PIN);
    reattached = true;
  }
  if (angle == servoAngle && !reattached) {
    return; // no redundant writes — avoids needless PWM refresh twitch
  }
  servoAngle = angle;
  panServo.write(servoAngle);
  lastMoveAt = millis();
}

void applyTrackingStep(int logicalDirection) {
  unsigned long now = millis();
  if (now - lastMoveAt < TRACK_INTERVAL_MS) return;
  lastMoveAt = now;
  int direction = REVERSE_SERVO ? -logicalDirection : logicalDirection;
  setServoAngle(servoAngle + (direction * TRACK_STEP));
}

void pulseTrackCommand(MovementCommand cmd) {
  if (cmd == CMD_LEFT) applyTrackingStep(-1);
  else if (cmd == CMD_RIGHT) applyTrackingStep(1);
}

MovementCommand parseCommand(String message) {
  message.trim();
  message.toUpperCase();

  if (message.startsWith("CMD_")) {
    message = message.substring(4);
  }

  if (message == "MOVED_LEFT") return CMD_LEFT;
  if (message == "MOVED_RIGHT") return CMD_RIGHT;
  if (message == "CENTERED") return CMD_CENTER;
  if (message == "STOPPED") return CMD_IDLE;
  if (message == "SCAN") return CMD_SEARCH;
  // OUT_OF_FRAME and other non-action states: hold current pan angle.
  if (message == "OUT_OF_FRAME") return CMD_IDLE;

  if (message == "LEFT") return CMD_LEFT;
  if (message == "RIGHT") return CMD_RIGHT;
  if (message == "CENTER") return CMD_CENTER;
  if (message == "SEARCH") return CMD_SEARCH;
  if (message == "IDLE" || message == "STOP") return CMD_IDLE;
  return CMD_IDLE;
}

const char* mqttStateName(int state) {
  switch (state) {
    case -4: return "TIMEOUT";
    case -3: return "CONNECTION_LOST";
    case -2: return "CONNECT_FAILED";
    case -1: return "DISCONNECTED";
    case 1: return "BAD_PROTOCOL";
    case 2: return "BAD_CLIENT_ID";
    case 3: return "UNAVAILABLE";
    case 4: return "BAD_CREDENTIALS";
    case 5: return "UNAUTHORIZED";
    default: return "UNKNOWN";
  }
}

// --- Inputs (MQTT & Serial) ---

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  MovementCommand cmd = parseCommand(message);
  lastCommandAt = millis();

  // Ignore non-action occlusion messages (servo holds position).
  if (message == "OUT_OF_FRAME") {
    return;
  }

  Serial.print("[MQTT] Received: ");
  Serial.println(message);

  if (cmd == CMD_LEFT || cmd == CMD_RIGHT) {
    pulseTrackCommand(cmd);
    currentCommand = CMD_IDLE;
    return;
  }

  currentCommand = cmd;
}

void handleSerial() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    MovementCommand newCmd = parseCommand(input);

    if (newCmd != CMD_IDLE || input.indexOf("IDLE") >= 0) {
      currentCommand = newCmd;
      lastCommandAt = millis();
      Serial.print("[SERIAL] Executing: ");
      Serial.println(input);
    }
  }
}

// --- Networking ---

bool connectWiFi() {
  if (WiFi.status() == WL_CONNECTED && WiFi.localIP()[0] != 0) {
    wifiReady = true;
    return true;
  }

  wifiReady = false;
  Serial.print("[WiFi] Connecting");
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.persistent(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 15000) {
    delay(500);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED && WiFi.localIP()[0] != 0) {
    wifiReady = true;
    Serial.println();
    Serial.print("[WiFi] Connected, IP=");
    Serial.println(WiFi.localIP());
    delay(800);
    return true;
  }

  Serial.println();
  Serial.println("[WiFi] Failed");
  return false;
}

bool connectMqtt() {
  if (mqttClient.connected()) return true;
  if (!wifiReady) return false;

  unsigned long now = millis();
  if (now - lastReconnectAttempt < reconnectBackoffMs) return false;
  lastReconnectAttempt = now;

  Serial.print("[MQTT] Connecting to ");
  Serial.print(MQTT_SERVER);
  Serial.print(":");
  Serial.print(MQTT_PORT);
  Serial.print(" as ");
  Serial.print(mqttClientId);
  Serial.print("... ");

  if (!mqttClient.connect(mqttClientId)) {
    int state = mqttClient.state();
    Serial.print("Failed, rc=");
    Serial.print(state);
    Serial.print(" (");
    Serial.print(mqttStateName(state));
    Serial.println(")");
    reconnectBackoffMs = min(reconnectBackoffMs + 2000UL, 30000UL);
    return false;
  }

  reconnectBackoffMs = 3000;
  Serial.println("Connected");
  mqttClient.subscribe(MQTT_TOPIC);
  return true;
}

// --- Servo Handling ---

void handleServo() {
  unsigned long now = millis();

  // Keep sweeping during SCAN until a new command arrives or tracking resumes.
  if (currentCommand != CMD_SEARCH && (now - lastCommandAt) > COMMAND_TIMEOUT_MS) {
    currentCommand = CMD_IDLE;
  }

  if (currentCommand == CMD_CENTER) {
    currentCommand = CMD_IDLE;
    return;
  }

  if (currentCommand == CMD_SEARCH) {
    if (now - lastMoveAt < SEARCH_INTERVAL_MS) return;
    lastMoveAt = now;

    setServoAngle(servoAngle + (sweepDirection * SEARCH_STEP));
    if (servoAngle >= SERVO_MAX_ANGLE) sweepDirection = -1;
    if (servoAngle <= SERVO_MIN_ANGLE) sweepDirection = 1;
    return;
  }

  // Idle: cut the PWM signal after a short delay so the servo stops trembling.
  if (currentCommand == CMD_IDLE && panServo.attached() && (now - lastMoveAt) > IDLE_DETACH_MS) {
    panServo.detach();
  }
}

// --- Main ---

void setup() {
  Serial.begin(115200);
  delay(10);
  Serial.println("\n[SYS] FaithLock Face-Servo Initializing...");

  snprintf(mqttClientId, sizeof(mqttClientId), "faithlock-%06X", ESP.getChipId());

  panServo.attach(SERVO_PIN);
  setServoAngle(SERVO_CENTER_ANGLE);

  mqttClient.setServer(MQTT_SERVER, MQTT_PORT);
  mqttClient.setCallback(mqttCallback);
  mqttClient.setBufferSize(512);
  mqttClient.setSocketTimeout(15);
  mqttClient.setKeepAlive(45);

  connectWiFi();
  lastCommandAt = millis();
}

void loop() {
  if (!wifiReady || WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }
  if (wifiReady && !mqttClient.connected()) {
    connectMqtt();
  }

  if (mqttClient.connected()) {
    mqttClient.loop();
  }

  handleSerial();
  handleServo();
  yield();
}
