if(NOT (NOT WITH_PYTHON AND ON_INFER))
  cc_library(
    performance_benchmark_utils
    SRCS benchmark_utils.cc
    DEPS ${eager_deps}
         ${fluid_deps}
         ${generated_deps}
         eager_scale
         scale_node
         generated_op
         matmul_v2_op
         dygraph_function
         eager_prim_api)

  cc_test_old(
    test_egr_performance_benchmark_eager_cpu
    SRCS
    benchmark_eager_cpu.cc
    DEPS
    conditional_block_op
    performance_benchmark_utils
    ${eager_deps}
    ${fluid_deps})
  cc_test_old(
    test_egr_performance_benchmark_fluid_cpu
    SRCS
    benchmark_fluid_cpu.cc
    DEPS
    dygraph_function
    performance_benchmark_utils
    ${eager_deps}
    ${fluid_deps})

  cc_test_old(
    test_egr_performance_benchmark_eager_cuda
    SRCS
    benchmark_eager_cuda.cc
    DEPS
    conditional_block_op
    performance_benchmark_utils
    ${eager_deps}
    ${fluid_deps})
  cc_test_old(
    test_egr_performance_benchmark_fluid_cuda
    SRCS
    benchmark_fluid_cuda.cc
    DEPS
    conditional_block_op
    standalone_executor
    performance_benchmark_utils
    dygraph_function
    ${eager_deps}
    ${fluid_deps})
endif()
