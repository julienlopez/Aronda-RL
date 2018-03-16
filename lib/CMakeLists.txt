include_directories(.)
include_directories(../3rd_party)

file(GLOB_RECURSE SOURCE_FILES ./*)

add_library(libAronda ${SOURCE_FILES})
