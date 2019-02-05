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

# Include current dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# =======================================================================
# Build and install - mlnn sample test.
# =======================================================================

set(BUILD_TEST_SAMPLE_TRAINING_MLNN ON CACHE BOOL "Build the application for testing of sample training of a multi-layer neural network")

if(${BUILD_TEST_SAMPLE_TRAINING_MLNN})
	# Create exeutable.
	ADD_EXECUTABLE(mlnn_sample_training_test mlnn_sample_training_test.cpp)
	# Link it with shared libraries.
	target_link_libraries(mlnn_sample_training_test 
	    logger
		configuration
		data_io
		${Boost_LIBRARIES}
		)
	if(OpenBLAS_FOUND)
		target_link_libraries(mlnn_sample_training_test  ${OpenBLAS_LIB} )
	endif(OpenBLAS_FOUND)

	# install test to bin directory
	install(TARGETS mlnn_sample_training_test RUNTIME DESTINATION bin)

endif(${BUILD_TEST_SAMPLE_TRAINING_MLNN})


# =======================================================================
# Build and install - mlnn batch training test.
# =======================================================================

set(BUILD_TEST_BATCH_TRAINING_MLNN ON CACHE BOOL "Build the application for testing batch training of a multi-layer neural network")

if(${BUILD_TEST_BATCH_TRAINING_MLNN})
	# Create executable.
	ADD_EXECUTABLE(mlnn_batch_training_test mlnn_batch_training_test.cpp)
	# Link it with shared libraries.
	target_link_libraries(mlnn_batch_training_test 
	    logger
		configuration
		data_io
		encoders
		${Boost_LIBRARIES}
		)
	if(OpenBLAS_FOUND)
		target_link_libraries(mlnn_batch_training_test  ${OpenBLAS_LIB} )
	endif(OpenBLAS_FOUND)

	# install test to bin directory
	install(TARGETS mlnn_batch_training_test RUNTIME DESTINATION bin)

endif(${BUILD_TEST_BATCH_TRAINING_MLNN})


# =======================================================================
# Build and install - MNIST recognition based on CNNs.
# =======================================================================

set(BUILD_MNIST_CNN_APP ON CACHE BOOL "Build the application for visualization of features of convolutional neural net trained on MNIST digits")

if(${BUILD_MNIST_CNN_APP})
        # Create exeutable.
        ADD_EXECUTABLE(mnist_convnet_features_visualization_test mnist_convnet_features_visualization_test.cpp)
        # Link it with shared libraries.
        target_link_libraries(mnist_convnet_features_visualization_test
			logger
			application
			configuration
			data_io
			encoders
			configuration
			opengl_visualization
			opengl_application
			${OPENGL_LIBRARIES}
			${GLUT_LIBRARY} 
			${Boost_LIBRARIES}  
	    )
        if(OpenBLAS_FOUND)
                target_link_libraries(mnist_convnet_features_visualization_test  ${OpenBLAS_LIB} )
        endif(OpenBLAS_FOUND)

        # install test to bin directory
        install(TARGETS mnist_convnet_features_visualization_test RUNTIME DESTINATION bin)

endif(${BUILD_MNIST_CNN_APP})


# =======================================================================
# Build and install -  MNIST recognition based on MLNN.
# =======================================================================

set(BUILD_MNIST_MLNN_APP ON CACHE BOOL "Build the application using a simple multi-Layer neural net for recognition of MNIST digits")

if(${BUILD_MNIST_MLNN_APP})
        # Create exeutable.
        ADD_EXECUTABLE(mnist_simple_mlnn_app mnist_simple_mlnn.cpp)
        # Link it with shared libraries.
        target_link_libraries(mnist_simple_mlnn_app
			logger
    		configuration
			data_io
			encoders
	        ${Boost_LIBRARIES}
	        )
        if(OpenBLAS_FOUND)
                target_link_libraries(mnist_simple_mlnn_app  ${OpenBLAS_LIB} )
        endif(OpenBLAS_FOUND)

        # install test to bin directory
        install(TARGETS mnist_simple_mlnn_app RUNTIME DESTINATION bin)

endif(${BUILD_MNIST_MLNN_APP})


# =======================================================================
# Build executables - mnist batch visualization test.
# =======================================================================

set(BUILD_TEST_MNIST_BATCH_VISUALIZATION ON CACHE BOOL "Build the MNIST batch visualization test application")

if(${BUILD_TEST_MNIST_BATCH_VISUALIZATION})
	# Create executable.
	add_executable(mnist_batch_visualization_test mnist_batch_visualization_test.cpp)
	# Link it with shared libraries.
	target_link_libraries(mnist_batch_visualization_test 
		logger
		application
		data_io
		encoders
		configuration
		opengl_visualization
		opengl_application
		${OPENGL_LIBRARIES}
		${GLUT_LIBRARY} 
		${Boost_LIBRARIES}  
		)
        if(OpenBLAS_FOUND)
                target_link_libraries(mnist_batch_visualization_test  ${OpenBLAS_LIB} )
        endif(OpenBLAS_FOUND)
	
	# install test to bin directory
	install(TARGETS mnist_batch_visualization_test RUNTIME DESTINATION bin)
	
endif(${BUILD_TEST_MNIST_BATCH_VISUALIZATION})

# =======================================================================
# Build executables - mnist mlnn features visualization test.
# =======================================================================

set(BUILD_TEST_MNIST_MLNN_FEATURES_VISUALIZATION ON CACHE BOOL "Build the program for visualization of features of mlnn layer trained on MNIST digits")

if(${BUILD_TEST_MNIST_MLNN_FEATURES_VISUALIZATION})
	# Create executable.
	add_executable(mnist_mlnn_features_visualization_test mnist_mlnn_features_visualization_test.cpp)
	# Link it with shared libraries.
	target_link_libraries(mnist_mlnn_features_visualization_test 
		logger
		application
		data_io
		encoders
		configuration
		opengl_visualization
		opengl_application
		${OPENGL_LIBRARIES}
		${GLUT_LIBRARY} 
		${Boost_LIBRARIES}  
		)
        if(OpenBLAS_FOUND)
                target_link_libraries(mnist_mlnn_features_visualization_test  ${OpenBLAS_LIB} )
        endif(OpenBLAS_FOUND)
	
	# install test to bin directory
	install(TARGETS mnist_mlnn_features_visualization_test RUNTIME DESTINATION bin)
	
endif(${BUILD_TEST_MNIST_MLNN_FEATURES_VISUALIZATION})


# =======================================================================
# Build executables - mnist hebbian features visualization test.
# =======================================================================

set(BUILD_TEST_MNIST_HEBBIAN_FEATURES_VISUALIZATION ON CACHE BOOL "Build the program for visualization of features of hebbian network trained on MNIST digits")

if(${BUILD_TEST_MNIST_HEBBIAN_FEATURES_VISUALIZATION})
	# Create executable.
	add_executable(mnist_hebbian_features_visualization_test mnist_hebbian_features_visualization_test.cpp)
	# Link it with shared libraries.
	target_link_libraries(mnist_hebbian_features_visualization_test 
		logger
		application
		data_io
		encoders
		configuration
		opengl_visualization
		opengl_application
		${OPENGL_LIBRARIES}
		${GLUT_LIBRARY} 
		${Boost_LIBRARIES}  
		)
        if(OpenBLAS_FOUND)
                target_link_libraries(mnist_hebbian_features_visualization_test  ${OpenBLAS_LIB} )
        endif(OpenBLAS_FOUND)
	
	# install test to bin directory
	install(TARGETS mnist_hebbian_features_visualization_test RUNTIME DESTINATION bin)
	
endif(${BUILD_TEST_MNIST_HEBBIAN_FEATURES_VISUALIZATION})

# =======================================================================
# Build and install - MNIST features learning through convolutional hebbian net
# =======================================================================

set(BUILD_MNIST_CONVHEBBIAN_APP ON CACHE BOOL "Build the application using Convolutional Hebbian learning for recognition of MNIST digits")

if(${BUILD_MNIST_CONVHEBBIAN_APP})
        # Create exeutable.
        ADD_EXECUTABLE(mnist_conv_hebbian mnist_conv_hebbian.cpp)
        # Link it with shared libraries.
        target_link_libraries(mnist_conv_hebbian
                    logger
                    application
                    data_io
                    encoders
                    configuration
                    opengl_visualization
                    opengl_application
					${OPENGL_LIBRARIES}
					${GLUT_LIBRARY} 
                    ${Boost_LIBRARIES}
	    )
        if(OpenBLAS_FOUND)
                target_link_libraries(mnist_conv_hebbian  ${OpenBLAS_LIB} )
        endif(OpenBLAS_FOUND)

        # install test to bin directory
        install(TARGETS mnist_conv_hebbian RUNTIME DESTINATION bin)

endif(${BUILD_MNIST_CONVHEBBIAN_APP})
