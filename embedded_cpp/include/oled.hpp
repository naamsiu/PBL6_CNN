
#pragma once

#include <cstddef>
#include <cstdint>

class OledI2C
{
public:
    OledI2C() = default;
    ~OledI2C();

    // Khong cho phep sao chep
    OledI2C(const OledI2C&) = delete;
    OledI2C& operator=(const OledI2C&) = delete;

    bool connect();
    bool init();

    bool showFrame(const uint8_t* buffer, std::size_t len);

private:
    int fd_ = -1;

    bool sendCommand(uint8_t cmd);
    bool sendData(const uint8_t* data, std::size_t len);
};
