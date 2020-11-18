#include "Logger.h"
#include <cstring>
#include <unistd.h>

const char* FormatedNow()
{
  const unsigned BUFF_SIZE = 256;
  static char buf[BUFF_SIZE];
  memset(buf, '\0', sizeof(char) * BUFF_SIZE);

  time_t now;
  time(&now);
  strftime(buf, sizeof(char) * BUFF_SIZE, "%F %T %z", gmtime(&now));

  return buf;
}

Logger::Logger(const char* component)
    : _component(component)
{

}

std::ostream& Logger::log(std::ostream& os, const char *channel)
{
    os << FormatedNow() << " " << _component << " " << channel << " ";
    return os;
}

std::ostream& Logger::info(std::ostream& os)
{
    return log(os, "INFO");
}

std::ostream& Logger::warn(std::ostream& os)
{
    return log(os, "WARN");
}

std::ostream& Logger::error(std::ostream& os)
{
    return log(os, "ERROR");
}

std::ostream& Logger::debug(std::ostream& os)
{
    return log(os, "DEBUG");
}