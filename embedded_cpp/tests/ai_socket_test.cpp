#include "ai_event_server.hpp"

#include <iostream>
#include <thread>
#include <chrono>

int main()
{
    AiEventServer server;

    if (!server.start())
    {
        std::cerr << "Cannot start AI Server\n";
        return 1;
    }

    std::cout << "AI Server started\n";
    std::cout << "Waiting for Python...\n";

    while (true)
    {
        std::string message;

        if (server.pollMessage(message))
        {
            std::cout << "Received: "
                      << message << std::endl;
        }

        std::this_thread::sleep_for(
            std::chrono::milliseconds(10)
        );
    }

    return 0;
}