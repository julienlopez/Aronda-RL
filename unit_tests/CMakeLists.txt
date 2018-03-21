include_directories(".")
include_directories("../lib")
include_directories(../3rd_party)

file(GLOB_RECURSE SOURCE_FILES ./*)
add_executable(unit_tests ${SOURCE_FILES})

target_link_libraries(unit_tests libAronda)


add_custom_command(TARGET unit_tests POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/unit_tests_data $<TARGET_FILE_DIR:unit_tests>)