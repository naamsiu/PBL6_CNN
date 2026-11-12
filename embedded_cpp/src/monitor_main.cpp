
#include "oled.hpp"
#include "system_info.hpp"
#include "uart.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <chrono>
#include <csignal>
#include <string>

// Kich thuoc man hinh OLED
constexpr int WIDTH = 128;
constexpr int HEIGHT = 64;

// Bien kiem soat vong lap
volatile std::sig_atomic_t running = 1;

// Xu ly khi nhan Ctrl + C
void stopProgram(int)
{
    running = 0;
}

// Lay font 5x7 cua tung ky tu
const uint8_t* getFont(char c)
{
    static const uint8_t digits[10][5] = {
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

    static const uint8_t dot[5] =
        {0x00,0x60,0x60,0x00,0x00};

    static const uint8_t space[5] =
        {0,0,0,0,0};

    static const uint8_t dash[5] =
        {0x08,0x08,0x08,0x08,0x08};

    static const uint8_t C[5] =
        {0x3E,0x41,0x41,0x41,0x22};

    static const uint8_t P[5] =
        {0x7F,0x09,0x09,0x09,0x06};

    static const uint8_t U[5] =
        {0x3F,0x40,0x40,0x40,0x3F};

    static const uint8_t T[5] =
        {0x01,0x01,0x7F,0x01,0x01};

    static const uint8_t E[5] =
        {0x7F,0x49,0x49,0x49,0x41};

    static const uint8_t M[5] =
        {0x7F,0x02,0x0C,0x02,0x7F};
    static const uint8_t Q[5] =
    {0x3E, 0x41, 0x51, 0x21, 0x5E};

    if (c >= '0' && c <= '9')
        return digits[c - '0'];

    switch (c)
    {
        case 'C': return C;
        case 'P': return P;
        case 'U': return U;
        case 'T': return T;
        case 'E': return E;
        case 'M': return M;
        case 'Q': return Q;
        case '.': return dot;
        case '-': return dash;
        default:  return space;
    }
}

// Ve chu vao framebuffer
void drawText(uint8_t* buffer,
              int x, int y,
              const char* text,
              int scale)
{
    while (*text)
    {
        const uint8_t* glyph = getFont(*text);

        for (int col = 0; col < 5; col++)
        {
            for (int bit = 0; bit < 7; bit++)
            {
                if (!(glyph[col] & (1 << bit)))
                    continue;

                for (int dx = 0; dx < scale; dx++)
                {
                    for (int dy = 0; dy < scale; dy++)
                    {
                        int px = x + col * scale + dx;
                        int py = y + bit * scale + dy;

                        if (px >= 0 && px < WIDTH &&
                            py >= 0 && py < HEIGHT)
                        {
                            buffer[(py / 8) * WIDTH + px]
                                |= (1 << (py % 8));
                        }
                    }
                }
            }
        }

        x += 6 * scale;
        text++;
    }
}

// =============================
// MAIN PROGRAM
// =============================
int main()
{
    // Bat tin hieu Ctrl + C
    std::signal(SIGINT, stopProgram);

    // Tao doi tuong OLED
    OledI2C oled;

    // Ket noi thiet bi I2C
    if (!oled.connect())
    {
        std::printf("OLED connection failed\n");
        return 1;
    }

    // Khoi tao man hinh SSD1306
    if (!oled.init())
    {
        std::printf("OLED initialization failed\n");
        return 1;
    }
    // Tao doi tuong UART
UartDevice uart;

// Ket noi voi ESP32
if (!uart.connect()) {
    std::printf("UART connection failed\n");
    return 1;
}

std::printf("UART connected successfully\n");

    // Bo dem anh 128x64, moi pixel 1 bit
    uint8_t framebuffer[WIDTH * HEIGHT / 8] = {};

    std::printf("OLED CPU monitor started\n");

    std::string uartLine;
    int mq3Raw = -1;

    // Vong lap cap nhat nhiet do
    while (running)
    {

char rxBuffer[128];

ssize_t n = uart.readAvailable(
    rxBuffer,
    sizeof(rxBuffer)
);

if (n > 0) {

    for (ssize_t i = 0; i < n; i++) {

        char c = rxBuffer[i];

        // Ket thuc mot dong du lieu
        if (c == '\n') {

            int value;
            char extra;

            // Chi chap nhan dong MQ3:<so nguyen>
            if (std::sscanf(
                    uartLine.c_str(),
                    "MQ3:%d%c",
                    &value,
                    &extra
                ) == 1 &&
                value >= 0 &&
                value <= 4095) {

                mq3Raw = value;

                std::printf(
                    "MQ3 RAW = %d\n",
                    mq3Raw
                );
            }

            uartLine.clear();
        }
        else if (c != '\r') {

            // Gioi han chieu dai dong nhan
            if (uartLine.size() < 32) {
                uartLine += c;
            }
            else {
                uartLine.clear();
            }
        }
    }
}
else if (n < 0) {
    std::perror("UART read failed");
    break;
}

            char mqText[16];

if (mq3Raw >= 0) {
    std::snprintf(
        mqText,
        sizeof(mqText),
        "%d",
        mq3Raw
    );
}
else {
    std::snprintf(
        mqText,
        sizeof(mqText),
        "----"
    );
}




        double cpuTemp = 0.0;
        char tempText[20];

        // Doc nhiet do CPU tu sysfs
        bool valid = readCpuTemp(cpuTemp);

        // Chuyen nhiet do thanh chuoi
        if (valid)
        {
            std::snprintf(
                tempText,
                sizeof(tempText),
                "%.1fC",
                cpuTemp
            );
        }
        else
        {
            std::snprintf(
                tempText,
                sizeof(tempText),
                "--.-C"
            );
        }

        // Xoa framebuffer cu
        std::memset(
            framebuffer,
            0,
            sizeof(framebuffer)
        );

        // Ve tieu de
       
// Hien thi tieu de nhiet do CPU
drawText(framebuffer, 38, 3, "CPU TEMP", 1);

// Hien thi gia tri nhiet do
drawText(framebuffer, 28, 13, tempText, 2);

// Hien thi tieu de cam bien MQ3
drawText(framebuffer, 52, 35, "MQ3", 1);

// Hien thi gia tri ADC MQ3
drawText(framebuffer, 40, 45, mqText, 2);

        // Gui framebuffer len OLED
        if (!oled.showFrame(
                framebuffer,
                sizeof(framebuffer)))
        {
            std::printf("OLED display failed\n");
            break;
        }

        // In ket qua ra Terminal
        if (valid)
        {
            std::printf(
                "CPU Temp: %.1f C\n",
                cpuTemp
            );
        }
        else
        {
            std::printf(
                "CPU temperature unavailable\n"
            );
        }

        // Cap nhat moi 1 giay
        std::this_thread::sleep_for(
            std::chrono::seconds(1)
        );
    }
    

    std::printf("Program stopped\n");

    // OledI2C destructor tu dong dong I2C
    return 0;
}
    