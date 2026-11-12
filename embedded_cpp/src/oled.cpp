
#include "oled.hpp"

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>

bool OledI2C::connect()
{
    if (fd_ >= 0)
        return true;

    fd_ = open("/dev/i2c-1", O_RDWR);

    if (fd_ < 0) {
        perror("Cannot open I2C");
        return false;
    }

    if (ioctl(fd_, I2C_SLAVE, 0x3C) < 0) {
        perror("Cannot select OLED");
        close(fd_);
        fd_ = -1;
        return false;
    }

    return true;
}

bool OledI2C::sendCommand(uint8_t cmd)
{
    if (fd_ < 0)
        return false;

    uint8_t packet[2] = {0x00, cmd};

    return write(fd_, packet, 2) == 2;
}

bool OledI2C::sendData(
    const uint8_t* data,
    std::size_t len)
{
    if (fd_ < 0 || data == nullptr ||
        len == 0 || len > 16)
        return false;

    uint8_t packet[17] = {};
    packet[0] = 0x40;

    std::memcpy(packet + 1, data, len);

    return write(fd_, packet, len + 1)
           == static_cast<ssize_t>(len + 1);
}

bool OledI2C::init()
{
    if (fd_ < 0)
        return false;

    const uint8_t commands[] = {
        0xAE, 0xD5, 0x80,
        0xA8, 0x3F, 0xD3, 0x00,
        0x40, 0x8D, 0x14,
        0x20, 0x00, 0xA1, 0xC8,
        0xDA, 0x12, 0x81, 0x7F,
        0xD9, 0xF1, 0xDB, 0x40,
        0xA4, 0xA6, 0x2E, 0xAF
    };

    for (uint8_t cmd : commands) {
        if (!sendCommand(cmd))
            return false;
    }

    return true;
}

bool OledI2C::showFrame(
    const uint8_t* buffer,
    std::size_t len)
{
    if (buffer == nullptr || len != 1024)
        return false;

    const uint8_t address[] = {
        0x21, 0x00, 0x7F,
        0x22, 0x00, 0x07
    };

    for (uint8_t cmd : address) {
        if (!sendCommand(cmd))
            return false;
    }

    for (std::size_t i = 0; i < len; i += 16) {
        if (!sendData(buffer + i, 16))
            return false;
    }

    return true;
}

OledI2C::~OledI2C()
{
    if (fd_ >= 0)
        close(fd_);
}
