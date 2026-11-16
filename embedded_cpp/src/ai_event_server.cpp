#include "ai_event_server.hpp"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <cerrno>


bool AiEventServer::start(const char* path)
{
    if (server_fd_ != -1 || path == nullptr)
        return false;

    sockaddr_un addr{};

    if (std::strlen(path) >= sizeof(addr.sun_path))
        return false;

    server_fd_ = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0);

    if (server_fd_ < 0)
    {
        std::perror("socket");
        return false;
    }

    addr.sun_family = AF_UNIX;
    std::strcpy(addr.sun_path, path);

    unlink(path);

    if (bind(server_fd_,
             reinterpret_cast<sockaddr*>(&addr),
             sizeof(addr)) < 0)
    {
        std::perror("bind");
        close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    if (listen(server_fd_, 1) < 0)
    {
        std::perror("listen");
        close(server_fd_);
        server_fd_ = -1;
        unlink(path);
        return false;
    }

    socket_path_ = path;

    return true;
}

bool AiEventServer::pollMessage(std::string& message)
{
    if (server_fd_ < 0)
        return false;

    // 1. Kiểm tra thông điệp đã nhận trước đó
    auto pos = receive_buffer_.find('\n');

    if (pos != std::string::npos)
    {
        message = receive_buffer_.substr(0, pos);
        receive_buffer_.erase(0, pos + 1);
        return true;
    }

    // 2. Chấp nhận kết nối từ Python
    if (client_fd_ < 0)
    {
        client_fd_ = accept(server_fd_, nullptr, nullptr);

        if (client_fd_ < 0)
            return false;

        int flags = fcntl(client_fd_, F_GETFL, 0);

        if (flags < 0 ||
            fcntl(client_fd_, F_SETFL, flags | O_NONBLOCK) < 0)
        {
            close(client_fd_);
            client_fd_ = -1;
            return false;
        }
    }

    // 3. Nhận dữ liệu
    char buffer[512];

    ssize_t n = recv(client_fd_, buffer, sizeof(buffer), 0);

    if (n > 0)
    {
        receive_buffer_.append(buffer, static_cast<size_t>(n));

        if (receive_buffer_.size() > 4096)
        {
            receive_buffer_.clear();
            close(client_fd_);
            client_fd_ = -1;
            return false;
        }

        pos = receive_buffer_.find('\n');

        if (pos != std::string::npos)
        {
            message = receive_buffer_.substr(0, pos);
            receive_buffer_.erase(0, pos + 1);
            return true;
        }
    }
    else if (n == 0)
    {
        close(client_fd_);
        client_fd_ = -1;
        receive_buffer_.clear();
    }
    else if (errno != EAGAIN &&
             errno != EWOULDBLOCK &&
             errno != EINTR)
    {
        std::perror("recv");
        close(client_fd_);
        client_fd_ = -1;
        receive_buffer_.clear();
    }

    return false;
}

void AiEventServer::stop()
{
    // 1. Đóng kết nối với Python
    if (client_fd_ >= 0)
    {
        close(client_fd_);
        client_fd_ = -1;
    }

    // 2. Đóng Socket Server
    if (server_fd_ >= 0)
    {
        close(server_fd_);
        server_fd_ = -1;
    }

    // 3. Xóa đường dẫn Socket
    if (!socket_path_.empty())
    {
        unlink(socket_path_.c_str());
        socket_path_.clear();
    }

    // 4. Xóa dữ liệu còn trong bộ đệm
    receive_buffer_.clear();
}

AiEventServer::~AiEventServer()
{
    stop();
}