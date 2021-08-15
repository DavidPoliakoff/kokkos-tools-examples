/**
 * two_var
 *
 * Complexity: medium
 *
 * Tuning problem:
 *
 * This is a two-valued tuning problem, in which you need
 * both parameters to learn the answer. There are two
 * values between 0 and 11 (inclusive).
 *
 * The penalty function here is just the distance between
 * your answer and the provided value.
 *
 */
#include <Kokkos_Core.hpp>
#include <chrono>
#include <cmath> // cbrt
#include <cstdlib>
#include <iostream>
#include <random>
#include <tuple>
#include <unistd.h>
#include <vector>
auto make_value_candidates(int num_candidates) {
  std::vector<int64_t> candidates;
  for (int x = 0; x < num_candidates; ++x) {
    candidates.push_back(x);
  }
  int64_t *bad_candidate_impl =
      (int64_t *)malloc(sizeof(int64_t) * candidates.size());
  memcpy(bad_candidate_impl, candidates.data(),
         sizeof(int64_t) * candidates.size());
  return Kokkos::Tools::Experimental::make_candidate_set(candidates.size(),
                                                         bad_candidate_impl);
}
int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    constexpr const int num_iters = 50000;
    constexpr const int num_candidates = 4;
    srand(time(NULL));

    // IDs with which to refer to our tuning input/output types
    size_t x_value_id;
    size_t y_value_id;
    size_t x_answer_id;
    size_t y_answer_id;

    Kokkos::Tools::Experimental::VariableInfo x_value_info;
    x_value_info.type = Kokkos::Tools::Experimental::ValueType::
        kokkos_value_int64; // this value is int-like
    x_value_info.category = Kokkos::Tools::Experimental::StatisticalCategory::
        kokkos_value_ratio; // ratios can be formed of its values
    x_value_info.valueQuantity =
        Kokkos::Tools::Experimental::CandidateValueType::
            kokkos_value_set; // candidate values come from a set, not a range
    x_value_info.candidates = make_value_candidates(num_candidates);

    Kokkos::Tools::Experimental::VariableInfo y_value_info;
    y_value_info.type =
        Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;
    y_value_info.category =
        Kokkos::Tools::Experimental::StatisticalCategory::kokkos_value_ratio;
    y_value_info.valueQuantity =
        Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
    y_value_info.candidates = make_value_candidates(num_candidates);
    Kokkos::Tools::Experimental::VariableInfo x_answer_info;
    x_answer_info.type =
        Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;
    x_answer_info.category =
        Kokkos::Tools::Experimental::StatisticalCategory::kokkos_value_ratio;
    x_answer_info.valueQuantity =
        Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
    x_answer_info.candidates = make_value_candidates(num_candidates);

    Kokkos::Tools::Experimental::VariableInfo y_answer_info;
    y_answer_info.type =
        Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;
    y_answer_info.category = Kokkos::Tools::Experimental::StatisticalCategory::
        kokkos_value_categorical;
    y_answer_info.valueQuantity =
        Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
    y_answer_info.candidates = make_value_candidates(num_candidates);

    // declare the types to the tuning tool
    x_value_id = Kokkos::Tools::Experimental::declare_input_type(
        "tuning_playground.x_value", x_value_info);
    y_value_id = Kokkos::Tools::Experimental::declare_input_type(
        "tuning_playground.y_value", y_value_info);
    x_answer_id = Kokkos::Tools::Experimental::declare_output_type(
        "tuning_playground.x_answer", x_answer_info);
    y_answer_id = Kokkos::Tools::Experimental::declare_output_type(
        "tuning_playground.y_answer", y_answer_info);
    constexpr const int grid_size = num_candidates * num_candidates;
    for (int iter = 0; iter < num_iters; ++iter) {
      int64_t x = iter % num_candidates;
      int64_t y = (iter / num_candidates) % num_candidates;
      // fill a vector with the feature values we'd like to set
      std::vector<Kokkos::Tools::Experimental::VariableValue> feature_vector{
          Kokkos::Tools::Experimental::make_variable_value(x_value_id, x),
          Kokkos::Tools::Experimental::make_variable_value(y_value_id, y)};

      // fill a vector with the tuning values we'd like the tool to overwrite
      std::vector<Kokkos::Tools::Experimental::VariableValue> answer_vector{
          Kokkos::Tools::Experimental::make_variable_value(x_answer_id,
                                                           int64_t(0)),
          Kokkos::Tools::Experimental::make_variable_value(y_answer_id,
                                                           int64_t(0))};

      size_t context = Kokkos::Tools::Experimental::get_new_context_id();
      // tell the tool that there's a context it should measure values for
      Kokkos::Tools::Experimental::begin_context(context);

      // declare features to it
      Kokkos::Tools::Experimental::set_input_values(context, 2,
                                                    feature_vector.data());

      // ask it to overwrite our tuning values
      Kokkos::Tools::Experimental::request_output_values(context, 2,
                                                         answer_vector.data());

      // calculate a penalty
      auto penalty = std::abs(answer_vector[0].value.int_value - x) +
                     std::abs(answer_vector[1].value.int_value - y);
      // sleep for that time
      usleep(100 * penalty);

      // tell the tool the context is done, and to git learnin'
      Kokkos::Tools::Experimental::end_context(context);
    }
  }
  Kokkos::finalize();
}
