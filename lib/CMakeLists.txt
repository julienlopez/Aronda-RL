include_directories(.)

file(GLOB_RECURSE SOURCE_FILES ./*)

add_library(libAronda ${SOURCE_FILES})
