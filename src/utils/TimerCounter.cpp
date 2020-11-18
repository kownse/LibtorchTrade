#include "TimerCounter.h"
#include <iostream>
#include <string>
#include <cstring>
#include <sstream>
#include "CombinationUtil.h"

#include <chrono>
using namespace std::chrono;

#include "../utils/Logger.h"
static Logger gLogger("TimerCounter");

TimerCounter::TimerCounter(unsigned threads, unsigned log)
    : _cum_duration_round(0),
    _round_start(duration_cast< milliseconds >(system_clock::now().time_since_epoch()).count()),
    _epoch_start(_round_start),
    _total_round(0),
    _cum_duration_epoch(0.0),
    _num_round(0),
    _threads(threads),
    _logThreshold(log)
{
    
}

void TimerCounter::setTotalRound(uint64_t total)
{
    _total_round = total;
    _cum_duration_round = 0.0;
    _num_round = 0;
    _round_start = duration_cast< milliseconds >(
            system_clock::now().time_since_epoch()
        ).count();
}

uint64_t TimerCounter::epochEnd(uint64_t start) {
  uint64_t now = duration_cast< milliseconds >(system_clock::now().time_since_epoch()).count();

  std::lock_guard<std::mutex> lk(_mx);
  float duration = (now - start) / _threads;
  _cum_duration_epoch = _cum_duration_epoch * 0.9 + duration * 0.1;

  return now - start;
}

static std::string formatTime(uint64_t sec_left)
{
  std::stringstream ss;
  uint64_t hours = sec_left / 60 / 60;
  sec_left -= hours * 60 * 60;

  uint64_t mins = sec_left / 60;
  sec_left -= mins * 60;

  ss << hours << ':' << mins << ':' << sec_left;
  return ss.str();
}

uint64_t TimerCounter::calcSecLeft() {
  _cum_duration_round = duration_cast< milliseconds >(
          system_clock::now().time_since_epoch()
      ).count() - _round_start;

  double ratio = (double)_num_round / _total_round;
  ratio = (1 - ratio) / ratio;
  uint64_t sec = _cum_duration_round * ratio / 1000;
  
  return sec;
}

void TimerCounter::roundEnd() {
  std::lock_guard<std::mutex> lk(_mx);
  if (++_num_round % _logThreshold == 0) {
    if (_linux_return)
      std::cout << "\r\e[K";

    std::cout << " ms_epoch=" << _cum_duration_epoch
              << " ms_round=" <<  _cum_duration_round / _num_round
              << " time_left=" << formatTime(calcSecLeft());

    if (_linux_return)
      std::cout << std::flush;
    else
      std::cout << std::endl;
  }
}

void TimerCounter::epochRoundEnd() {
  return roundEnd();
}

void TimerCounter::setCombinationCount(int num_cols, int start, int end) {
    setTotalRound(calcCombNum(num_cols, start, end));
}
