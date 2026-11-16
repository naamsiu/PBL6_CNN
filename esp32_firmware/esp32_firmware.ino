// PBL6 - ESP32 audio alerts selected by Raspberry Pi over UART2.
// Sketch folder: esp32_firmware/esp32_firmware.ino
// Required WAV byte-list files: esp32_firmware/src/audio/*.txt
// Each .txt must contain comma-separated WAV bytes: 0x52, 0x49, ...
// Libraries: ESP8266Audio (AudioGeneratorWAV, AudioOutputI2S, AudioFileSourcePROGMEM).

#include <Arduino.h>
#include <cstring>
#include "AudioGeneratorWAV.h"
#include "AudioOutputI2S.h"
#include "AudioFileSourcePROGMEM.h"

#define MQ3_PIN   34
#define I2S_BCLK  26
#define I2S_LRC   25
#define I2S_DOUT  27
#define RXD2      16
#define TXD2      17

// Arduino compiles from a build folder: keep audio inside sketch/src/audio.
const uint8_t audioDrowsy[] PROGMEM = {
#include "src/audio/drowsy.txt"
};
const uint8_t audioYawning[] PROGMEM = {
#include "src/audio/yawning.txt"
};
const uint8_t audioHeadNod[] PROGMEM = {
#include "src/audio/head_nod.txt"
};
const uint8_t audioLookAway[] PROGMEM = {
#include "src/audio/look_away.txt"
};
const uint8_t audioLookDown[] PROGMEM = {
#include "src/audio/look_down.txt"
};
const uint8_t audioNoFace[] PROGMEM = {
#include "src/audio/no_face.txt"
};
// Optional: alcohol is only played when the Pi sends ALERT:ALCOHOL.
// Do NOT enable alcohol detection based on uncalibrated raw MQ3 ADC.
#if __has_include("src/audio/alcohol.txt")
#define HAS_ALCOHOL_AUDIO 1
const uint8_t audioAlcohol[] PROGMEM = {
#include "src/audio/alcohol.txt"
};
#else
#define HAS_ALCOHOL_AUDIO 0
#endif

struct Clip {
  const char* command;
  const uint8_t* bytes;
  size_t length;
  bool repeat;
};

const Clip clips[] = {
  {"ALERT:DROWSY", audioDrowsy, sizeof(audioDrowsy), true},
  {"ALERT:YAWNING", audioYawning, sizeof(audioYawning), false},
  {"ALERT:HEAD_NOD", audioHeadNod, sizeof(audioHeadNod), false},
  {"ALERT:LOOK_AWAY", audioLookAway, sizeof(audioLookAway), false},
  {"ALERT:LOOK_DOWN", audioLookDown, sizeof(audioLookDown), false},
  {"ALERT:NO_FACE", audioNoFace, sizeof(audioNoFace), false},
#if HAS_ALCOHOL_AUDIO
  {"ALERT:ALCOHOL", audioAlcohol, sizeof(audioAlcohol), true},
#endif
};

AudioGeneratorWAV* wav = nullptr;
AudioFileSourcePROGMEM* file = nullptr;
AudioOutputI2S* out = nullptr;
const Clip* activeClip = nullptr;
bool isPlaying = false;
bool hasPlayedThisEvent = false;
unsigned long lastPlayEnd = 0;
unsigned long lastSensorRead = 0;
unsigned long lastAlertHeartbeat = 0;

constexpr unsigned long REPLAY_INTERVAL_MS = 3000;
constexpr unsigned long ALERT_TIMEOUT_MS = 4000;
char uartLine[64] = {};
size_t uartLength = 0;

void stopPlayback()
{
  if (wav != nullptr) {
    wav->stop();
    delete wav;
    wav = nullptr;
  }
  if (file != nullptr) {
    delete file;
    file = nullptr;
  }
  isPlaying = false;
  lastPlayEnd = millis();
}

void stopAlert()
{
  activeClip = nullptr;
  hasPlayedThisEvent = false;
  stopPlayback();
}

const Clip* findClip(const char* command)
{
  for (const auto& clip : clips) {
    if (std::strcmp(command, clip.command) == 0)
      return &clip;
  }
  return nullptr;
}

void playActiveClip()
{
  if (activeClip == nullptr || isPlaying) return;
  file = new AudioFileSourcePROGMEM(activeClip->bytes, activeClip->length);
  wav = new AudioGeneratorWAV();
  if (file == nullptr || wav == nullptr || !wav->begin(file, out)) {
    Serial.println("[AUDIO] Cannot start WAV");
    stopPlayback();
    hasPlayedThisEvent = true;  // avoid a rapid retry loop
    return;
  }
  isPlaying = true;
  hasPlayedThisEvent = true;
  Serial.printf("[AUDIO] Playing %s (%u bytes)\n", activeClip->command,
                static_cast<unsigned int>(activeClip->length));
}

void updateAudio()
{
  if (!isPlaying) return;
  if (wav != nullptr && wav->isRunning() && wav->loop()) return;
  stopPlayback();
  Serial.println("[AUDIO] Playback finished");
}

void handleCommand(const char* cmd)
{
  if (std::strcmp(cmd, "ALERT:STOP") == 0) {
    if (activeClip != nullptr) Serial.println("[UART2] ALERT:STOP");
    stopAlert();
    return;
  }
  const Clip* found = findClip(cmd);
  if (found == nullptr) {
    Serial.printf("[UART2] Unsupported: %s\n", cmd);
    return;
  }
  lastAlertHeartbeat = millis();
  if (found != activeClip) {
    stopAlert();
    activeClip = found;
    Serial.printf("[UART2] New alert: %s\n", cmd);
    playActiveClip();
  }
  // Same command is heartbeat only. Do not restart the voice every second.
}

void readPiCommands()
{
  int processed = 0;
  while (Serial2.available() > 0 && processed++ < 128) {
    const char c = static_cast<char>(Serial2.read());
    if (c == '\n') {
      uartLine[uartLength] = '\0';
      handleCommand(uartLine);
      uartLength = 0;
    } else if (c != '\r') {
      if (uartLength + 1 < sizeof(uartLine))
        uartLine[uartLength++] = c;
      else
        uartLength = 0;
    }
  }
}

void readMQ3()
{
  const int raw = analogRead(MQ3_PIN);
  Serial.printf("[MQ3] ADC: %d\n", raw);
  Serial2.printf("MQ3:%d\n", raw);
}

void setup()
{
  Serial.begin(115200);
  Serial2.begin(115200, SERIAL_8N1, RXD2, TXD2);
  analogReadResolution(12);
  analogSetPinAttenuation(MQ3_PIN, ADC_11db);
  audioLogger = &Serial;
  out = new AudioOutputI2S();
  out->SetPinout(I2S_BCLK, I2S_LRC, I2S_DOUT);
  out->SetGain(0.6);
  Serial.println("ESP32 MQ3 + multi-voice alert READY");
  Serial.println("Waiting for ALERT:* from Raspberry Pi");
}

void loop()
{
  readPiCommands();
  updateAudio();

  const unsigned long now = millis();
  if (activeClip != nullptr && now - lastAlertHeartbeat >= ALERT_TIMEOUT_MS) {
    Serial.println("[AUDIO] UART heartbeat timeout -> STOP");
    stopAlert();
  }

  if (activeClip != nullptr && !isPlaying) {
    if (!hasPlayedThisEvent)
      playActiveClip();
    else if (activeClip->repeat && now - lastPlayEnd >= REPLAY_INTERVAL_MS)
      playActiveClip();
  }

  if (now - lastSensorRead >= 1000) {
    lastSensorRead = now;
    readMQ3();
  }
}
