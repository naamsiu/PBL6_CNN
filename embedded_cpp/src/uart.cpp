
#include "uart.hpp"

#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <poll.h>
#include <cerrno>

// Mo va cau hinh UART
bool UartDevice::connect(const char* device)
{
    if (fd_ >= 0)
        return true;

    // 1. Mo cong UART
    fd_ = open(device, O_RDWR | O_NOCTTY);

    if (fd_ < 0) {
        perror("Cannot open UART");
        return false;
    }

    // 2. Lay cau hinh hien tai
    termios tty{};

    if (tcgetattr(fd_, &tty) != 0) {
        perror("tcgetattr failed");
        close(fd_);
        fd_ = -1;
        return false;
    }

    // 3. Cau hinh UART
    cfmakeraw(&tty);

    cfsetispeed(&tty, B115200);
    cfsetospeed(&tty, B115200);

    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;

    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    tty.c_cflag |= CLOCAL | CREAD;

    tty.c_cc[VMIN] = 1;
    tty.c_cc[VTIME] = 0;

    // 4. Ap dung cau hinh
    if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
        perror("tcsetattr failed");
        close(fd_);
        fd_ = -1;
        return false;
    }

    return true;
}

// Tu dong dong UART khi doi tuong bi huy
UartDevice::~UartDevice()
{
    if (fd_ >= 0) {
        close(fd_);
    }
}

ssize_t UartDevice::readBytes(char* buffer, std::size_t size)
{
    if (fd_ < 0 || buffer == nullptr || size == 0)
        return -1;

    return read(fd_, buffer, size);
}

ssize_t UartDevice::readAvailable(char* buffer, std::size_t size)
{
    if (fd_ < 0 || buffer == nullptr || size == 0) {
        errno = EINVAL;
        return -1;
    }

    pollfd pfd{};
    pfd.fd = fd_;
    pfd.events = POLLIN;

    // Kiem tra ngay, khong cho du lieu
    int result = poll(&pfd, 1, 0);

    if (result < 0)
        return -1;

    if (result == 0)
        return 0;

    if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
        errno = EIO;
        return -1;
    }

    // Chi doc khi UART san sang
    return read(fd_, buffer, size);
}
