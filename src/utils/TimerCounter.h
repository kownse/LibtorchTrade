#ifndef _TIMER_COUNT_H_
#define _TIMER_COUNT_H_

#include <mutex>

class TimerCounter {
public:
  TimerCounter(unsigned threads, unsigned log);

  uint64_t epochEnd(uint64_t start);
  void roundEnd();
  void epochRoundEnd();
  
  void setTotalRound(uint64_t total);
  void setCombinationCount(int num_cols, int start, int end);
  void setLinuxReturn(bool ret) {_linux_return = ret;}

private:
  uint64_t calcSecLeft();

  uint64_t _cum_duration_round;
  uint64_t _round_start;
  uint64_t _epoch_start;
  uint64_t _total_round;

  double _cum_duration_epoch;
  
  unsigned _num_round;
  unsigned _threads;
  unsigned _logThreshold;

  std::mutex _mx;
  bool _linux_return = false;
};

#endif // _TIMER_COUNT_H_