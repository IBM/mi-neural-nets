# Copyright (C) tkornuta, IBM Corporation 2015-2019
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =======================================================================
set(BUILD_APP_MNIST_PATCH_RECONSTRUCTION ON CACHE BOOL "Build the application realizing MNIST patch autoencoder-based reconstruction")
if(${BUILD_APP_MNIST_PATCH_RECONSTRUCTION})
	# Create executable.
	add_executable(mnist_patch_autoencoder_reconstruction
		MNISTPatchReconstructionApplication.cpp
		console_application_main.cpp
		)
		
	# Link it with shared libraries.
	target_link_libraries(mnist_patch_autoencoder_reconstruction 
		logger
		configuration
		application
		data_utils
		importers
		opengl_visualization
		opengl_application
		${OPENGL_LIBRARIES}
		${GLUT_LIBRARY} 
		${Boost_LIBRARIES}  
		)
	if(OpenBLAS_FOUND)
		target_link_libraries(mnist_patch_autoencoder_reconstruction  ${OpenBLAS_LIB} )
	endif(OpenBLAS_FOUND)

	# install test to bin directory
	install(TARGETS mnist_patch_autoencoder_reconstruction RUNTIME DESTINATION bin)
	
endif(${BUILD_APP_MNIST_PATCH_RECONSTRUCTION})


# Include current dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# =======================================================================
set(BUILD_APP_MNIST_PATCH_SOFTMAX ON CACHE BOOL "Build the application realizing MNIST patch autoencoder-based softmax classifier, using the imported, previously trained auto-encoder")
if(${BUILD_APP_MNIST_PATCH_SOFTMAX})
	# Create executable.
	add_executable(mnist_patch_autoencoder_softmax
		MNISTPatchSoftmaxApplication.cpp
		console_application_main.cpp
		)
		
	# Link it with shared libraries.
	target_link_libraries(mnist_patch_autoencoder_softmax 
		logger
		configuration
		application
		data_utils
		importers
		encoders
		opengl_visualization
		opengl_application
		${OPENGL_LIBRARIES}
		${GLUT_LIBRARY} 
		${Boost_LIBRARIES}  
		)
	if(OpenBLAS_FOUND)
		target_link_libraries(mnist_patch_autoencoder_softmax  ${OpenBLAS_LIB} )
	endif(OpenBLAS_FOUND)

	# install test to bin directory
	install(TARGETS mnist_patch_autoencoder_softmax RUNTIME DESTINATION bin)
	
endif(${BUILD_APP_MNIST_PATCH_SOFTMAX})

