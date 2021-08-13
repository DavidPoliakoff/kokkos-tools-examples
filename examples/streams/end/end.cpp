#include <Kokkos_Core.hpp>
#include <iostream>

using view_type = Kokkos::View<float*, Kokkos::DefaultExecutionSpace>;
constexpr const int data_size = 16000;
int repeats = 200000;
int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    Kokkos::DefaultExecutionSpace root_space;
    auto streams = Kokkos::Experimental::partition_space(root_space, 1, 1);
    view_type temperature_field1("temperature_one", data_size);
    view_type temperature_field2("temperature_two", data_size);
    auto f1_mirror = Kokkos::create_mirror_view(temperature_field1);
    auto f2_mirror = Kokkos::create_mirror_view(temperature_field2);
    for (int x = 0; x < repeats; ++x) {
      Kokkos::parallel_for(
          "process_temp1", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(streams[0], 0, data_size),
          KOKKOS_LAMBDA(int i) { temperature_field1(i) -= 1.0f; });
      Kokkos::deep_copy(streams[0], f1_mirror, temperature_field1);
      Kokkos::parallel_for(
          "process_temp2", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(streams[1], 0, data_size),
          KOKKOS_LAMBDA(int i) { temperature_field2(i) -= 1.0f; });
      Kokkos::deep_copy(streams[1],f2_mirror, temperature_field2);
      
      /** could do an edit step here */ 

    }
  }
  Kokkos::finalize();
}
