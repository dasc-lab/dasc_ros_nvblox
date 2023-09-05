#pragma once
#ifndef SIMPLE_TIMER_HPP
#define SIMPLE_TIMER_HPP

#include <chrono>


namespace nvblox::timing {

class SimpleTimer
{

  public:

  typedef std::chrono::high_resolution_clock clock;
  typedef std::chrono::nanoseconds ns;


  SimpleTimer()
    : start_(clock::now())
  {}

  uint64_t elapsed_ns() {
    auto dt = std::chrono::duration_cast<ns>(clock::now() - start_);
    return dt.count();
  }

  private:

  const clock::time_point start_;

}; // class SimpleTimer


} // namespace nvblox::timing


#endif // SIMPLE_TIMER_HPP
