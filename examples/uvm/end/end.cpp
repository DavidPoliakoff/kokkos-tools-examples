#include <Kokkos_Core.hpp>
#include <iostream>

using mem_space = Kokkos::CudaUVMSpace;
using view_type = Kokkos::View<float *, mem_space>;
constexpr const int data_size = 16000;
int repeats = 200;
constexpr const int output_interval = 1;
int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    view_type temperature("temperature", data_size);
    for (int x = 0; x < repeats; ++x) {
      Kokkos::parallel_for(
          "decrease_temp", Kokkos::RangePolicy<Kokkos::Cuda>(0, data_size),
          KOKKOS_LAMBDA(int i) { temperature(i) -= 1.0f; });
      Kokkos::Tools::pushRegion("edit_step");
      if ((x % output_interval) == 0) {
        double temperature_sum = 0.0;
        Kokkos::parallel_reduce(
            "edit", Kokkos::RangePolicy<Kokkos::Serial>(0, data_size),
            KOKKOS_LAMBDA(int i, double &contrib) {
              contrib += temperature(i);
            },
            Kokkos::Sum<double>(temperature_sum));
        std::cout << "Sum of temperatures on iteration " << x << ": "
                  << temperature_sum << std::endl;
      }
      Kokkos::Tools::popRegion();
    }
  }
  Kokkos::finalize();
}
