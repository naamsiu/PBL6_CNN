
#pragma once

#include <cstddef>
#include <sys/types.h>

class UartDevice
{
public:
    UartDevice() = default;
    ~UartDevice();

    // Khong cho phep sao chep doi tuong
    UartDevice(const UartDevice&) = delete;
    UartDevice& operator=(const UartDevice&) = delete;

    // Mo va cau hinh UART
    bool connect(const char* device = "/dev/ttyAMA0");

    // Doc du lieu tu ESP32
    ssize_t readBytes(char* buffer, std::size_t size);
    ssize_t readAvailable(char* buffer, std::size_t size);

private:
    int fd_ = -1;
};
