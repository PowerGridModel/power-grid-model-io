add_library(dll_helpers INTERFACE)

target_include_directories(dll_helpers INTERFACE 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

install(TARGETS dll_helpers
  EXPORT experiment_loadTargets
  COMPONENT experiment_load
  FILE_SET shared_library_core_public_headers
)
