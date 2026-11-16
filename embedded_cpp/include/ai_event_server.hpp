#pragma once

#include <string>

class AiEventServer
{
public:
    AiEventServer() = default;
    ~AiEventServer();

    bool start(const char* path = "/tmp/pbl6_ai.sock");

    bool pollMessage(std::string& message);

    void stop();

private:
    int server_fd_ = -1;
    int client_fd_ = -1;

    std::string socket_path_;
    std::string receive_buffer_;
};