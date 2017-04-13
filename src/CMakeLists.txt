# Include current dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Include current directory BEFORE the others - this will result in using LOCAL headers before the ones pointed by the CMAKE_INSTALL_PREFIX path!
# Include in here will result in enabling to use global headers paths e.g. <opengl/visualization/Window.hpp>
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

# =======================================================================
# Add subdirectories
# =======================================================================

add_subdirectory(optimization)

add_subdirectory(loss)

add_subdirectory(mlnn)

add_subdirectory(tests)

add_subdirectory(application)