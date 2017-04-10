# Include current dir
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# =======================================================================
# Build and install - mlnn batch training test.
# =======================================================================

set(BUILD_TEST_BATCH_TRAINING_MLNN ON CACHE BOOL "Build the application for ttesting batch training of a multi-layer neural network")

if(${BUILD_TEST_BATCH_TRAINING_MLNN})
	# Create exeutable.
	ADD_EXECUTABLE(mlnn_batch_training_test mlnn_batch_training_test.cpp)
	# Link it with shared libraries.
	target_link_libraries(mlnn_batch_training_test 
	    logger
		configuration
		types
		data_io
		mlnn
		${Boost_LIBRARIES}
		)

	# install test to bin directory
	install(TARGETS mlnn_batch_training_test RUNTIME DESTINATION bin)

endif(${BUILD_TEST_BATCH_TRAINING_MLNN})

# =======================================================================
# Build and install - mlnn test.
# =======================================================================

set(BUILD_TEST_MLNN ON CACHE BOOL "Build the application for testing creation/serialization of multi-layer neural networks")

if(${BUILD_TEST_MLNN})
	# Create exeutable.
	ADD_EXECUTABLE(mlnn_test mlnn_test.cpp)
	# Link it with shared libraries.
	target_link_libraries(mlnn_test 
	    logger
		configuration
		types
		data_io
		mlnn
		${Boost_LIBRARIES}
		)

	# install test to bin directory
	install(TARGETS mlnn_test RUNTIME DESTINATION bin)

endif(${BUILD_TEST_MLNN})

# =======================================================================
# Build and install - MNIST recognition based on CNNs.
# =======================================================================

set(BUILD_MNIST_CNN_APP ON CACHE BOOL "Build the application using Convolutional Neural Network for recognition of MNIST digits")

if(${BUILD_MNIST_CNN_APP})
        # Create exeutable.
        ADD_EXECUTABLE(mnist_convnet mnist_convnet.cpp)
        # Link it with shared libraries.
        target_link_libraries(mnist_convnet
			logger
			configuration
	        types
			data_io
			encoders
			mlnn
		    ${Boost_LIBRARIES}
	    )
        if(OpenBLAS_FOUND)
                target_link_libraries(mnist_convnet  ${OpenBLAS_LIB} )
        endif(OpenBLAS_FOUND)

        # install test to bin directory
        install(TARGETS mnist_convnet RUNTIME DESTINATION bin)

endif(${BUILD_MNIST_CNN_APP})


# =======================================================================
# Build and install -  MNIST recognition based on MLP.
# =======================================================================

set(BUILD_MNIST_MLP_APP ON CACHE BOOL "Build the application using Multi-Layer Perceptron for recognition of MNIST digits")

if(${BUILD_MNIST_MLP_APP})
        # Create exeutable.
        ADD_EXECUTABLE(mnist_simple_mlp_app mnist_simple_mlp.cpp)
        # Link it with shared libraries.
        target_link_libraries(mnist_simple_mlp_app
			logger
    		configuration
	        types
			data_io
			encoders
			mlnn
	        ${Boost_LIBRARIES}
	        )
        if(OpenBLAS_FOUND)
                target_link_libraries(mnist_simple_mlp_app  ${OpenBLAS_LIB} )
        endif(OpenBLAS_FOUND)

        # install test to bin directory
        install(TARGETS mnist_simple_mlp_app RUNTIME DESTINATION bin)

endif(${BUILD_MNIST_MLP_APP})
