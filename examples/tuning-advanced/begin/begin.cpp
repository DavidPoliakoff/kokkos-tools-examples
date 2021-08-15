#include <Kokkos_Core.hpp>
#include <iostream>

constexpr const int data_size = 16000;
int repeats = 20000;
const std::vector<int> sizes {1, 16, 1024, 2048};
struct small_op {
  int num_elements;
  double result;
};
using view_type = Kokkos::View<small_op *, Kokkos::DefaultExecutionSpace>;
int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    view_type small_ops("ops", data_size);
    using team_policy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
    using team_member = typename team_policy::member_type;
    for(const auto max_size: sizes) {
    Kokkos::parallel_for(
        "init",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, data_size),
        KOKKOS_LAMBDA(int i) { small_ops[i].num_elements = max_size; });
    for (int x = 0; x < repeats; ++x) {
      Kokkos::parallel_for(
          "compute-advanced", team_policy(data_size, Kokkos::AUTO, Kokkos::AUTO),
          KOKKOS_LAMBDA(const team_member &team) {
            int index = team.league_rank();
            Kokkos::parallel_reduce(
                Kokkos::ThreadVectorRange(team, small_ops(index).num_elements),
                [&](const int i, double &psum) {}, small_ops(index).result);
          });
    }
    }
  }
  Kokkos::finalize();
}
