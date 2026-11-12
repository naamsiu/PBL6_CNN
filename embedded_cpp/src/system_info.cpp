
#include "system_info.hpp"
#include <fstream>

bool readCpuTemp(double& temp)
{
    std::ifstream file(
        "/sys/class/thermal/thermal_zone0/temp"
    );

    long milliCelsius = 0;

    if (!(file >> milliCelsius))
    {
        return false;
    }

    temp = milliCelsius / 1000.0;

    return true;
}
