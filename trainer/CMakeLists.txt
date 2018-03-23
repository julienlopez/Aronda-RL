include_directories(".")
include_directories("../lib")

file(GLOB_RECURSE SOURCE_FILES ./*)
add_executable(trainer ${SOURCE_FILES})

target_link_libraries(trainer libAronda)
target_link_libraries(trainer "Cntk.Core-${AR_CNTK_VERSION}")
