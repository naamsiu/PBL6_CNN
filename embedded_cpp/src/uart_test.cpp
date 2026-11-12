
#include "uart.hpp"

#include <cstdio>

int main()
{
    // 1. Tao doi tuong UART
    UartDevice uart;

    // 2. Mo va cau hinh UART 115200 baud
    if (!uart.connect()) {
        std::printf("UART connection failed\n");
        return 1;
    }

    std::printf("UART receiver started\n");

    // 3. Bo dem nhan du lieu tu ESP32
    char buffer[128];

    // 4. Vong lap nhan du lieu
    while (true) {

        // Doc du lieu thong qua class UartDevice
        ssize_t n = uart.readBytes(
            buffer,
            sizeof(buffer) - 1
        );

        if (n > 0) {

            // Them ky tu ket thuc chuoi
            buffer[n] = '\0';

            // In du lieu nhan duoc
            std::printf("%s", buffer);
            std::fflush(stdout);
        }
        else if (n < 0) {
            std::perror("UART read failed");
            break;
        }
        else {
            std::printf("UART read returned 0\n");
            break;
        }
    }

    // Destructor cua UartDevice se dong UART
    return 0;
}
