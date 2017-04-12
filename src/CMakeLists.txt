# Include current dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})


# =======================================================================
# Add subdirectories
# =======================================================================

add_subdirectory(optimization)

add_subdirectory(loss)

add_subdirectory(mlnn)

add_subdirectory(tests)