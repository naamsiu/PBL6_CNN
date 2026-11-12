
#include <Arduino.h>
#include "AudioGeneratorWAV.h"
#include "AudioOutputI2S.h"
#include "AudioFileSourcePROGMEM.h"

// ===== CAU HINH CHAN =====
#define MQ3_PIN   34

#define I2S_BCLK  26
#define I2S_LRC   25
#define I2S_DOUT  27

#define RXD2 16
#define TXD2 17

// ===== DU LIEU WAV HEX =====
const uint8_t xinchao_wav[] PROGMEM = {
#include "xinchao.txt"
};

// ===== DOI TUONG AUDIO =====
AudioGeneratorWAV *wav = nullptr;
AudioFileSourcePROGMEM *file = nullptr;
AudioOutputI2S *out = nullptr;

bool isPlaying = false;

unsigned long lastPlayEnd = 0;
unsigned long lastSensorRead = 0;

const unsigned long AUDIO_INTERVAL = 3000;

// ===== PHAT AM THANH =====
void playXinchao() {

  if (isPlaying) return;

  Serial.println("[LOA] Bat dau phat XIN CHAO");

  file = new AudioFileSourcePROGMEM(
    xinchao_wav,
    sizeof(xinchao_wav)
  );

  wav = new AudioGeneratorWAV();

  if (wav->begin(file, out)) {
    isPlaying = true;
  } else {
    Serial.println("[LOI] Khong phat duoc WAV");

    wav->stop();
    delete wav;
    delete file;

    wav = nullptr;
    file = nullptr;
    lastPlayEnd = millis();
  }
}

// ===== DUY TRI PHAT AUDIO =====
void updateAudio() {

  if (!isPlaying) return;

  if (wav->isRunning() && wav->loop()) {
    return;
  }

  wav->stop();

  delete wav;
  delete file;

  wav = nullptr;
  file = nullptr;

  isPlaying = false;
  lastPlayEnd = millis();

  Serial.println("[LOA] Da phat xong");
}

// ===== DOC CAM BIEN MQ3 =====
void readMQ3() {

  int raw = analogRead(MQ3_PIN);

  Serial.printf("[MQ3] ADC: %d\n", raw);

  Serial2.printf("MQ3:%d\n", raw);
}

// ===== SETUP =====
void setup() {

  Serial.begin(115200);

  Serial2.begin(
    115200,
    SERIAL_8N1,
    RXD2,
    TXD2
  );

  Serial.println("UART2 READY");
  Serial2.println("HELLO ESP32");

  analogReadResolution(12);
  analogSetPinAttenuation(MQ3_PIN, ADC_11db);

  audioLogger = &Serial;

  out = new AudioOutputI2S();

  out->SetPinout(
    I2S_BCLK,
    I2S_LRC,
    I2S_DOUT
  );

  out->SetGain(0.6);

  Serial.println("======================");
  Serial.println("ESP32 WAV AUDIO TEST");
  Serial.println("======================");

  Serial.printf(
    "WAV SIZE: %u bytes\n",
    (unsigned int)sizeof(xinchao_wav)
  );

  // Phat ngay khi khoi dong
  playXinchao();
}

// ===== LOOP =====
void loop() {

  // 1. Duy tri WAV
  updateAudio();

  // 2. Phat lai sau khi nghi 3 giay
  if (!isPlaying &&
      millis() - lastPlayEnd >= AUDIO_INTERVAL) {
    playXinchao();
  }

  // 3. Doc MQ3 moi giay
  if (millis() - lastSensorRead >= 1000) {
    lastSensorRead = millis();
    readMQ3();
  }
}
