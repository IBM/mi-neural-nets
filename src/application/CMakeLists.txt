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
		types
		data_utils
		data_io
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
		types
		data_utils
		data_io
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

