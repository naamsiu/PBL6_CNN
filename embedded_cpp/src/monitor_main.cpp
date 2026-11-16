// PBL6 Driver Monitor - Raspberry Pi / SSD1306 128x64
// Features: CPU temperature, ESP32 MQ3 (UART), AI events (Unix socket).
// The AI value displayed is the LAST received state, not a live heartbeat.

#include "oled.hpp"
#include "system_info.hpp"
#include "uart.hpp"
#include "ai_event_server.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <utility>

namespace {

constexpr int WIDTH = 128;
constexpr int HEIGHT = 64;
constexpr auto LOOP_DELAY = std::chrono::milliseconds(15);
constexpr auto DISPLAY_MIN_INTERVAL = std::chrono::milliseconds(200);
constexpr auto SENSOR_INTERVAL = std::chrono::seconds(1);
constexpr auto ALERT_TX_INTERVAL = std::chrono::seconds(1);
constexpr auto AI_STALE_TIMEOUT = std::chrono::seconds(5);

volatile std::sig_atomic_t running = 1;

void stopProgram(int)
{
    running = 0;
}

// 5x7 glyphs: columns, LSB = top pixel. Covers 0-9, A-Z and punctuation.
const std::uint8_t* getFont(char raw)
{
    static const std::uint8_t digits[10][5] = {
        {0x3E,0x51,0x49,0x45,0x3E}, // 0
        {0x00,0x42,0x7F,0x40,0x00}, // 1
        {0x42,0x61,0x51,0x49,0x46}, // 2
        {0x21,0x41,0x45,0x4B,0x31}, // 3
        {0x18,0x14,0x12,0x7F,0x10}, // 4
        {0x27,0x45,0x45,0x45,0x39}, // 5
        {0x3C,0x4A,0x49,0x49,0x30}, // 6
        {0x01,0x71,0x09,0x05,0x03}, // 7
        {0x36,0x49,0x49,0x49,0x36}, // 8
        {0x06,0x49,0x49,0x29,0x1E}  // 9
    };
    static const std::uint8_t letters[26][5] = {
        {0x7E,0x11,0x11,0x11,0x7E}, // A
        {0x7F,0x49,0x49,0x49,0x36}, // B
        {0x3E,0x41,0x41,0x41,0x22}, // C
        {0x7F,0x41,0x41,0x22,0x1C}, // D
        {0x7F,0x49,0x49,0x49,0x41}, // E
        {0x7F,0x09,0x09,0x09,0x01}, // F
        {0x3E,0x41,0x49,0x49,0x7A}, // G
        {0x7F,0x08,0x08,0x08,0x7F}, // H
        {0x00,0x41,0x7F,0x41,0x00}, // I
        {0x20,0x40,0x41,0x3F,0x01}, // J
        {0x7F,0x08,0x14,0x22,0x41}, // K
        {0x7F,0x40,0x40,0x40,0x40}, // L
        {0x7F,0x02,0x0C,0x02,0x7F}, // M
        {0x7F,0x04,0x08,0x10,0x7F}, // N
        {0x3E,0x41,0x41,0x41,0x3E}, // O
        {0x7F,0x09,0x09,0x09,0x06}, // P
        {0x3E,0x41,0x51,0x21,0x5E}, // Q
        {0x7F,0x09,0x19,0x29,0x46}, // R
        {0x46,0x49,0x49,0x49,0x31}, // S
        {0x01,0x01,0x7F,0x01,0x01}, // T
        {0x3F,0x40,0x40,0x40,0x3F}, // U
        {0x1F,0x20,0x40,0x20,0x1F}, // V
        {0x3F,0x40,0x38,0x40,0x3F}, // W
        {0x63,0x14,0x08,0x14,0x63}, // X
        {0x07,0x08,0x70,0x08,0x07}, // Y
        {0x61,0x51,0x49,0x45,0x43}  // Z
    };
    static const std::uint8_t space[5] = {0,0,0,0,0};
    static const std::uint8_t dot[5]   = {0,0x60,0x60,0,0};
    static const std::uint8_t dash[5]  = {0x08,0x08,0x08,0x08,0x08};
    static const std::uint8_t colon[5] = {0,0x36,0x36,0,0};
    static const std::uint8_t slash[5] = {0x20,0x10,0x08,0x04,0x02};
    static const std::uint8_t excl[5]  = {0,0,0x5F,0,0};
    static const std::uint8_t under[5] = {0x40,0x40,0x40,0x40,0x40};

    char c = raw;
    if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    if (c >= '0' && c <= '9') return digits[c - '0'];
    if (c >= 'A' && c <= 'Z') return letters[c - 'A'];
    switch (c) {
        case '.': return dot;
        case '-': return dash;
        case ':': return colon;
        case '/': return slash;
        case '!': return excl;
        case '_': return under;
        default:  return space;
    }
}

void drawPixel(std::uint8_t* buffer, int x, int y)
{
    if (x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT) return;
    buffer[(y / 8) * WIDTH + x] |= static_cast<std::uint8_t>(1u << (y % 8));
}

void drawText(std::uint8_t* buffer, int x, int y, const char* text, int scale = 1)
{
    if (!text || scale < 1) return;
    for (; *text && x < WIDTH; ++text, x += 6 * scale) {
        const std::uint8_t* glyph = getFont(*text);
        for (int col = 0; col < 5; ++col) {
            for (int bit = 0; bit < 7; ++bit) {
                if (!(glyph[col] & (1u << bit))) continue;
                for (int dx = 0; dx < scale; ++dx) {
                    for (int dy = 0; dy < scale; ++dy) {
                        drawPixel(buffer, x + col * scale + dx, y + bit * scale + dy);
                    }
                }
            }
        }
    }
}

void drawLine(std::uint8_t* buffer, int y)
{
    for (int x = 0; x < WIDTH; ++x) drawPixel(buffer, x, y);
}

// Extract the uppercase "state" from the newline-delimited JSON protocol.
// Example: {"state": "DROWSY"}. No extra JSON dependency needed.
bool extractAiState(const std::string& message, std::string& state)
{
    if (message.size() > 512 || message.empty()) return false;
    const auto first = message.find_first_not_of(" \t\r\n");
    const auto last = message.find_last_not_of(" \t\r\n");
    if (first == std::string::npos || message[first] != '{' || message[last] != '}')
        return false;

    const auto key = message.find("\"state\"");
    if (key == std::string::npos) return false;
    auto pos = key + 7; // 7 characters in "state"
    pos = message.find_first_not_of(" \t", pos);
    if (pos == std::string::npos || message[pos] != ':') return false;
    pos = message.find_first_not_of(" \t", pos + 1);
    if (pos == std::string::npos || message[pos] != '"') return false;
    const auto end = message.find('"', pos + 1);
    if (end == std::string::npos) return false;

    std::string value = message.substr(pos + 1, end - pos - 1);
    if (value.empty() || value.size() > 32) return false;
    for (char c : value) {
        if (!((c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'))
            return false;
    }
    state = std::move(value);
    return true;
}

// Shorten long AI state names for the 128px screen.
std::string oledAiLabel(const std::string& state)
{
    if (state == "BLINK_OR_CLOSING") return "BLINK/CLOSING";
    if (state == "EYES_CLOSED") return "EYES CLOSED";
    if (state == "HEAD_NOD") return "HEAD NOD";
    if (state == "LOOK_AWAY") return "LOOK AWAY";
    if (state == "NO_FACE") return "NO FACE";
    if (state == "MOUTH_OPEN") return "MOUTH OPEN";
    if (state == "MONITORING") return "MONITORING";
    if (state == "") return "WAITING";

    std::string label = state;
    std::replace(label.begin(), label.end(), '_', ' ');
    if (label.size() > 16) label.resize(16); // 16 chars * 6 px = 96px
    return label;
}

bool isAlert(const std::string& state)
{
    return state == "DROWSY" || state == "YAWNING" ||
           state == "HEAD_NOD" || state == "LOOK_AWAY" ||
           state == "NO_FACE";
}

void drawDashboard(std::uint8_t* framebuffer, double cpuTemp,
                   bool cpuValid, int mq3Raw, const std::string& aiState,
                   bool aiReceived)
{
    std::memset(framebuffer, 0, WIDTH * HEIGHT / 8);

    drawText(framebuffer, 22, 1, "DRIVER MONITOR");
    drawLine(framebuffer, 11);

    char cpu[20];
    if (cpuValid) std::snprintf(cpu, sizeof(cpu), "%.1fC", cpuTemp);
    else std::snprintf(cpu, sizeof(cpu), "--.-C");
    drawText(framebuffer, 4, 16, "CPU TEMP");
    drawText(framebuffer, 86, 16, cpu);

    char mq[20];
    if (mq3Raw >= 0) std::snprintf(mq, sizeof(mq), "%d", mq3Raw);
    else std::snprintf(mq, sizeof(mq), "----");
    drawText(framebuffer, 4, 28, "MQ3 RAW");
    drawText(framebuffer, 92, 28, mq);

    drawText(framebuffer, 4, 40, "AI");
    const std::string label = oledAiLabel(aiState);
    drawText(framebuffer, 28, 40, label.c_str());

    drawLine(framebuffer, 51);
    const char* footer = !aiReceived ? "WAITING AI EVENT" :
                         (aiState == "AI_OFFLINE" ? "AI CONNECTION LOST" :
                         (isAlert(aiState) ? "! AI ALERT !" : "LAST AI STATE"));
    drawText(framebuffer, 4, 55, footer);
}

} // namespace

int main()
{
    std::signal(SIGINT, stopProgram);
    std::signal(SIGTERM, stopProgram);

    OledI2C oled;
    if (!oled.connect() || !oled.init()) {
        std::fprintf(stderr, "OLED connection or initialization failed\n");
        return 1;
    }

    UartDevice uart;
    if (!uart.connect()) {
        std::fprintf(stderr, "UART connection failed\n");
        return 1;
    }
    std::printf("UART connected successfully\n");

    // Independent AI IPC server. Stop /tmp/ai_socket_test before running.
    AiEventServer aiServer;
    const bool aiSocketReady = aiServer.start();
    std::printf("AI Socket Server %s\n", aiSocketReady ? "started" : "failed");
    std::printf("OLED CPU / MQ3 / AI monitor started\n");

    std::array<std::uint8_t, WIDTH * HEIGHT / 8> framebuffer{};
    std::string uartLine;
    int mq3Raw = -1;
    std::string aiState;
    bool aiReceived = false;
    double cpuTemp = 0.0;
    bool cpuValid = false;
    bool needsDraw = true;
    std::string currentAlertCommand = "ALERT:STOP";
    bool haveAiEvent = false;

    using Clock = std::chrono::steady_clock;
    auto nextSensorRead = Clock::now();
    auto lastDraw = Clock::now() - DISPLAY_MIN_INTERVAL;
    auto lastAiEventTime = Clock::now();
    auto nextAlertTx = Clock::now();

    while (running) {
        // 1. Read UART without waiting for a line; keep split lines buffered.
        char rxBuffer[128];
        const ssize_t n = uart.readAvailable(rxBuffer, sizeof(rxBuffer));
        if (n < 0) {
            if (running) std::perror("UART read failed");
            break;
        }
        for (ssize_t i = 0; i < n; ++i) {
            const char c = rxBuffer[i];
            if (c == '\n') {
                int value;
                char extra;
                if (std::sscanf(uartLine.c_str(), "MQ3:%d%c", &value, &extra) == 1 &&
                    value >= 0 && value <= 4095) {
                    if (value != mq3Raw) needsDraw = true;
                    mq3Raw = value;
                    std::printf("MQ3 RAW = %d\n", mq3Raw);
                }
                uartLine.clear();
            } else if (c != '\r') {
                if (uartLine.size() < 32) uartLine.push_back(c);
                else uartLine.clear();
            }
        }

        // 2. Drain several pending IPC messages each iteration.
        if (aiSocketReady) {
            for (int i = 0; i < 16; ++i) {
                std::string aiMessage;
                if (!aiServer.pollMessage(aiMessage)) break;
                std::string nextState;
                if (extractAiState(aiMessage, nextState)) {
                    if (!aiReceived || nextState != aiState) {
                        needsDraw = true;
                        nextAlertTx = Clock::now(); // react immediately to state changes
                    }
                    aiState = std::move(nextState);
                    aiReceived = true;
                    haveAiEvent = true;
                    lastAiEventTime = Clock::now();
                    std::printf("AI State = %s\n", aiState.c_str());
                } else {
                    std::printf("AI IPC ignored malformed message\n");
                }
            }
        }

        // 3. Expire stale AI state if Python stops sending heartbeats.
        // The Python integration should send the current state every <= 2 s.
        const auto now = Clock::now();
        if (haveAiEvent && now - lastAiEventTime > AI_STALE_TIMEOUT) {
            haveAiEvent = false;
            aiState = "AI_OFFLINE";
            nextAlertTx = now;
            aiReceived = true;
            needsDraw = true;
            std::printf("AI heartbeat lost: AI_OFFLINE\n");
        }

        // 4. Multi-voice alerts. Re-send active alert once per second as
        // heartbeat; STOP whenever there is no current actionable state.
        // LOOK_DOWN needs a distinct state/event from Python to activate.
        const char* wanted = "ALERT:STOP";
        if (haveAiEvent) {
            if      (aiState == "DROWSY")    wanted = "ALERT:DROWSY";
            else if (aiState == "YAWNING")   wanted = "ALERT:YAWNING";
            else if (aiState == "HEAD_NOD")  wanted = "ALERT:HEAD_NOD";
            else if (aiState == "LOOK_AWAY") wanted = "ALERT:LOOK_AWAY";
            else if (aiState == "LOOK_DOWN") wanted = "ALERT:LOOK_DOWN";
            else if (aiState == "NO_FACE")   wanted = "ALERT:NO_FACE";
        }
        const std::string desiredCommand(wanted);
        if ((desiredCommand != currentAlertCommand ||
             (desiredCommand != "ALERT:STOP" && now >= nextAlertTx)) &&
            now >= nextAlertTx) {
            if (uart.sendLine(wanted)) {
                if (desiredCommand != currentAlertCommand)
                    std::printf("ESP32 command: %s\n", wanted);
                currentAlertCommand = desiredCommand;
            }
            nextAlertTx = now + ALERT_TX_INTERVAL;
        }

        // 5. Read CPU temperature every 1 second (not every 15 ms).
        if (now >= nextSensorRead) {
            cpuValid = readCpuTemp(cpuTemp);
            if (cpuValid) std::printf("CPU Temp: %.1f C\n", cpuTemp);
            else std::printf("CPU temperature unavailable\n");
            nextSensorRead = now + SENSOR_INTERVAL;
            needsDraw = true;
        }

        // 6. Render changed state at most 5 times/second.
        if (needsDraw && now - lastDraw >= DISPLAY_MIN_INTERVAL) {
            drawDashboard(framebuffer.data(), cpuTemp, cpuValid,
                          mq3Raw, aiState, aiReceived);
            if (!oled.showFrame(framebuffer.data(), framebuffer.size())) {
                std::fprintf(stderr, "OLED display failed\n");
                break;
            }
            lastDraw = Clock::now();
            needsDraw = false;
        }
        std::this_thread::sleep_for(LOOP_DELAY);
    }

    if (currentAlertCommand != "ALERT:STOP") uart.sendLine("ALERT:STOP");
    std::printf("Program stopped\n");
    return 0; // RAII destructors close OLED, UART and the Unix socket.
}
