#include <Kokkos_Core.hpp>
#include <iostream>

constexpr const int data_size = 16000;
int repeats = 20000;
std::vector<int64_t> sizes {1, 16, 1024, 2048};
struct small_op {
  int num_elements;
  double result;
};
using view_type = Kokkos::View<small_op *, Kokkos::DefaultExecutionSpace>;
int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    size_t input_size_id;
    Kokkos::Tools::Experimental::VariableInfo info;
    info.type = Kokkos::Tools::Experimental::ValueType::
        kokkos_value_int64; // this value is int-like
    info.category = Kokkos::Tools::Experimental::StatisticalCategory::
        kokkos_value_ratio; // ratios can be formed of its values
    info.valueQuantity =
        Kokkos::Tools::Experimental::CandidateValueType::
            kokkos_value_set; // candidate values come from a set, not a range
    info.candidates = Kokkos::Tools::Experimental::make_candidate_set(sizes.size(), sizes.data());
    input_size_id = Kokkos::Tools::Experimental::declare_input_type("advanced_tuning.input_size", info);
    view_type small_ops("ops", data_size);
    using team_policy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
    using team_member = typename team_policy::member_type;
    for(const auto max_size: sizes) {
    size_t context = Kokkos::Tools::Experimental::get_new_context_id();
    auto size_value = Kokkos::Tools::Experimental::make_variable_value(input_size_id, max_size);
      Kokkos::Tools::Experimental::begin_context(context);
      Kokkos::Tools::Experimental::set_input_values(context, 1, &size_value);

    Kokkos::parallel_for(
        "init",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, data_size),
        KOKKOS_LAMBDA(int i) { small_ops[i].num_elements = max_size; });
    for (int x = 0; x < repeats; ++x) {
      Kokkos::parallel_for(
          "compute-adv-end", team_policy(data_size, Kokkos::AUTO, Kokkos::AUTO),
          KOKKOS_LAMBDA(const team_member &team) {
            int index = team.league_rank();
            Kokkos::parallel_reduce(
                Kokkos::ThreadVectorRange(team, small_ops(index).num_elements),
                [&](const int i, double &psum) {}, small_ops(index).result);
          });
    }
      Kokkos::Tools::Experimental::end_context(context);
    }
  }
  Kokkos::finalize();
}
