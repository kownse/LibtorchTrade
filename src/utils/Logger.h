#ifndef _LOGGER_H_
#define _LOGGER_H_

#include <string>
#include <mutex>
#include <iostream>

const char* FormatedNow();

class Logger {
public:
    Logger(const char* component);

    std::ostream& info(std::ostream& = std::cout);
    std::ostream& warn(std::ostream& = std::cout);
    std::ostream& error(std::ostream& = std::cout);
    std::ostream& debug(std::ostream& = std::cout);

private:
    std::ostream& log(std::ostream& os, const char *channel);

    std::string _host;
    const std::string _component;
    std::mutex _mx;
};

#endif //_LOGGER_H_