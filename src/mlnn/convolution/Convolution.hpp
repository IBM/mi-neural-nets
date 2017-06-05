/*!
 * \file Convolution.hpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#ifndef SRC_MLNN_CONVOLUTION_HPP_
#define SRC_MLNN_CONVOLUTION_HPP_

#include <mlnn/layer/Layer.hpp>

#include<types/MatrixTypes.hpp>
#include<types/MatrixArray.hpp>

namespace mic {
namespace mlnn {
namespace convolution {

/*!
 * \brief Class representing a convolution layer, with "valid padding" and variable stride.
 * \author tkornuta
 * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT=float>
class Convolution : public mic::mlnn::Layer<eT> {
public:

	/*!
	 * Creates a convolutional layer.
	 * @param input_height_ Height of the input / rows (e.g. 28 for MNIST).
	 * @param input_width_ Width of the input / columns (e.g. 28 for MNIST).
	 * @param input_channels_ Number of channels of the input (e.g. 3 for RGB images).
	 * @param number_of_filters_ Number of filters = Length of the output vector.
	 * @param filter_size_ Size of filters (assuming square filters).
	 * @param stride_ Stride (assuming equal vertical and horizontal strides).
	 * @param name_ Name of the layer.
	 */
	Convolution(size_t input_height_, size_t input_width_, size_t input_channels_, size_t number_of_filters_, size_t filter_size_, size_t stride_, std::string name_ = "Convolution") :
		Layer<eT>::Layer(input_height_, input_width_, input_channels_,
				/* height, width: temporary output values to be set in constructor!*/
				1, 1, number_of_filters_,
				LayerTypes::Convolution, name_),
				filter_size(filter_size_),
				stride(stride_)
	{
		// Calculate number of receptive fields within a "single input channel".
		assert(input_height >= filter_size);
		int height_rest = (int)input_height-filter_size;
		output_height = 1;
		while(height_rest >= (int)stride){
			output_height++;
			height_rest -= stride;
		}//: while width
		// Filters must "exactly" fit!
		if (height_rest != 0) {
			LOG(LERROR) << " Filter width and stride does not fit image height";
			LOG(LINFO) << streamLayerParameters();
			exit(-1);
		}

		assert(input_width >= filter_size);
		int width_rest = (int)input_width-filter_size;
		output_width= 1;
		while(width_rest >= (int)stride){
			output_width++;
			width_rest -= stride;
		}//: while width
		// Filters must "exactly" fit!
		if (height_rest != 0) {
			LOG(LINFO) << "Filter height and stride does not fit image height";
			LOG(LNOTICE) << streamLayerParameters();
			exit(-1);
		}

		LOG(LDEBUG)<<streamLayerParameters();

		// Set output height and resize matrices!
		s["y"]->resize(Layer<eT>::outputSize(), batch_size); 	// outputs
		g["y"]->resize(Layer<eT>::outputSize(), batch_size); 	// gradients
		m["ys"]->resize(Layer<eT>::outputSize(), 1);			// sample
		m["yc"]->resize(output_width*output_height, 1);			// channel


		// Calculate "range" - for initialization.
		eT range_init = (eT) (input_width * input_height * input_depth) +
				(output_width * output_height * output_depth);
		eT range = sqrt(6.0 / range_init);

		// Create filters.
		for (size_t fi=0; fi< output_depth; fi++) {
			// A given filter (neuron layer) has in fact connection to all input channels.
			for (size_t ic=0; ic< input_depth; ic++) {
				// Create the weights matrix - a row vector.
				p.add ("W"+std::to_string(fi)+"x"+std::to_string(ic), 1, filter_size*filter_size);
				// Initialize weights of the W matrix.
				p["W"+std::to_string(fi)+"x"+std::to_string(ic)]->rand(-range, range);
				//std::cout<<"W.dims = " << p["W"+std::to_string(fi)+std::to_string(ic)]->rows() << "," << p["W"+std::to_string(fi)+std::to_string(ic)]->cols() << "\n";

				// Create the weights matrix for updates/gradients.
				g.add ("W"+std::to_string(fi)+"x"+std::to_string(ic), 1, filter_size*filter_size);
			}//: for input channels.

		}//: for filter

		// Create a single bias vector for all filters.
		p.add ("b", output_depth, 1);
		p["b"]->setZero();
		// Bias gradient.
		g.add ("b", output_depth, 1);

		// Allocate (temporary) memory for "input receptive fields".
		for (size_t ry=0; ry< output_height; ry++) {
			for (size_t rx=0; rx< output_width; rx++) {
				// Create receptive field matrix.
				m.add ("xrf"+std::to_string(ry)+"x"+std::to_string(rx), filter_size, filter_size);
			}//: for
		}//: for

		// Allocate (temporary) memory for "output channels" - matrices.
		for (size_t fi=0; fi< output_depth; fi++) {
			// Create output channel matrix.
			m.add ("yc"+std::to_string(fi), output_height, output_width);
		}//: for

/*		// Allocate memory for "gradient x channel".
		m.add ("xc", input_height, input_width);

		// Allocate memory for "gradient x sample".
		m.add ("xs", input_channels * input_height * input_width, 1);

		// Allocate memory for "gradient y sample".
		m.add ("ys", number_of_filters * output_height * output_width, 1);*/

		// Allocate (temporary) memory for "inverse input receptive fields" - used in backpropagation.
		for (size_t ry=0; ry< filter_size; ry++) {
			for (size_t rx=0; rx< filter_size; rx++) {
				// Create receptive field matrix.
				m.add ("ixrf"+std::to_string(ry)+"x"+std::to_string(rx), output_height, output_width);
			}//: for
		}//: for

		// Set gradient descent as default optimization function.
		Layer<eT>::template setOptimization<mic::neural_nets::optimization::GradientDescent<eT> > ();
	};

	/*!
	 * Virtual destructor - empty.
	 */
	virtual ~Convolution() {};

	/*!
	 * Stream layer parameters.
	 * @return Ostream object.
	 */
	virtual std::string streamLayerParameters() {
		std::ostringstream os_;
		// Display id/type.
		os_ << "  [" << Layer<eT>::type() << "]: " << Layer<eT>::layer_name << ": "
				<< Layer<eT>::inputSize() << "x" << batch_size << " -> " << Layer<eT>::outputSize() << "x" << batch_size << "\n";
		// Display dimensions.
		os_<<"    * input_height = " << input_height <<std::endl;
		os_<<"    * input_width = " << input_width <<std::endl;
		os_<<"    * input_channels = " << input_depth <<std::endl;
		os_<<"    * filter_size = " << filter_size <<std::endl;
		os_<<"    * stride = " << stride <<std::endl;
		os_<<"    * output_height = " << output_height <<std::endl;
		os_<<"    * output_width = " << output_width <<std::endl;
		os_<<"    * output_channels = " << output_depth;

		return os_.str();
	}

	/*!
	 * Performs forward pass through the filters. Can process batches.
	 */
	void forward(bool test = false) {
//		std::cout << "forward()\n";
		// Get input matrix.
		mic::types::MatrixPtr<eT> batch_x = s['x'];
		//std::cout<< "forward batch_x=\n" << (*batch) << std::endl;
		std::cout << "forward input x activation: min:" << (*batch_x).minCoeff() <<" max: " << (*batch_x).maxCoeff() << std::endl;

		// Get output pointer - so the results will be stored!
		mic::types::MatrixPtr<eT> batch_y = s['y'];
		batch_y->setZero();

		// Iterate through samples in the input batch.
//#pragma omp parallel for
		for (size_t ib=0; ib< batch_size; ib++) {

			// The "output sample" that will "concatenate" the output channels into one "column vector".
			mic::types::MatrixPtr<eT> y_sample = m["ys"];

			// 1. "Reset" output "channels".
			for (size_t fi=0; fi< output_depth; fi++) {
				// Get output channel matrix.
				mic::types::MatrixPtr<eT> y_channel = m["yc"+std::to_string(fi)];
				// Resize it to a matrix (!).
				y_channel->resize(output_height, output_width);
				// And "reset" i.e. set value of all cells to bias (so we can skip adding it later).
				y_channel->setValue((*p["b"])[fi]);
			}//: filters

			// 2. Get input sample from batch!
			mic::types::MatrixPtr<eT> sample = m["xs"];
			(*sample) = batch_x->col(ib);
			//std::cout<< "sample=\n" << (*sample) << std::endl;

			// 3. Iterate through input channels.
			for (size_t ic=0; ic< input_depth; ic++) {
				// 3.1. Get input channel from image.
				mic::types::MatrixPtr<eT> ichannel = m["xc"];
				// Copy block - resizes the input channel matrix.
				(*ichannel) = sample->block(ic*input_height*input_width, 0, input_height*input_width, 1);
				// Resize channel using the given dimensions.
				ichannel->resize(input_height, input_width);
				//std::cout<< "======  switching input channel = " << ic <<" ichannel=\n" << (*ichannel) << std::endl;

				// 3.2. Fill receptive fields from given input channel.
				// Iterate through receptive fields - vertical and horizontal
				// and copy data from given channel to array of "input receptive fields".
				// Image coordinates: ix, iy.
				// Receptive field "id" coordinates: rx, ry.
				for (size_t ry=0, iy = 0; ry< output_height; ry++, iy+=stride) {
					for (size_t rx=0, ix = 0; rx< output_width; rx++, ix+=stride) {
						//std::cout<<"ry =" << ry <<" rx =" << rx <<" iy =" << iy <<" ix =" << ix << std::endl;
						// Get receptive field matrix...
						mic::types::MatrixPtr<eT> field = m["xrf"+std::to_string(ry)+"x"+std::to_string(rx)];
						// Copy block from channel - resizes the field matrix.
						(*field) = ichannel->block(iy,ix,filter_size, filter_size);
						//std::cout<< "field=\n" << (*field) << std::endl;
						// Resize the field to a column vector.
						field->resize(filter_size*filter_size, 1);
					}//: for rx
				}//: for ry

				// 3.3. Convolve receptive fields with filters.

				// Iterate through filters and calculate the result of the convolution.
				for (size_t fi=0; fi< output_depth; fi++) {
					// Get output channel for a given filter.
					mic::types::MatrixPtr<eT> y_channel = m["yc"+std::to_string(fi)];
					// Resize to matrix.
					y_channel->resize(output_height, output_width);
					//std::cout << "====  switching to oc " << fi << " = \n" << (*y_channel)<<std::endl;
					// Get "part of a given neuron" responding to a given input channel.
					mic::types::MatrixPtr<eT> W = p["W"+std::to_string(fi)+"x"+std::to_string(ic)];
					// Not required - just in case. :]
					W->resize(1,filter_size*filter_size);
					// Iterate through receptive fields.
					for (size_t ry=0; ry< output_height; ry++) {
						for (size_t rx=0; rx< output_width; rx++) {
							// Get receptive field matrix of size (1, filter_size^2)...
							mic::types::MatrixPtr<eT> xrf = m["xrf"+std::to_string(ry)+"x"+std::to_string(rx)];
							// ... and result of "convolution" of that filter with the part of the input "below" the receptive field.
							/*std::cout<<"ic = " << ic  << " filter = " << fi << " ry =" << ry <<" rx =" << rx <<std::endl;
							std::cout<< "W=\n" << (*W) << std::endl;
							std::cout<< "xrf=\n" << (*xrf) << std::endl;
							std::cout<< " result = " << ((*W)*(*xrf)) << std::endl;*/
							(*y_channel)(ry, rx) += ((*W)*(*xrf))(0);
						}//: for rx
					}//: for ry
					//std::cout << "====  ic = " << ic << " filter= " << fi << " oc = \n" << (*y_channel)<<std::endl;

					// Resize output channel to a column vector.
					y_channel->resize(output_height*output_width, 1);
					// Concatenate.
					y_sample->block(fi*output_height*output_width, 0, output_height*output_width, 1)
							= (*y_channel);
				}//: for filters

			}//: for channels
			//std::cout <<"output before adding y =" << (*y) <<std::endl;
			// Set column in the output batch.
			batch_y->col(ib) = (*y_sample);

		}//: for batch

		//std::cout <<"output y =" << (*batch_y).transpose() <<std::endl;
		std::cout << "forward output y activation: min:" << (*batch_y).minCoeff() <<" max: " << (*batch_y).maxCoeff() << std::endl;
	}//: forward

	/*!
	 * Back-propagates the gradients through the layer.
	 */
	void backward() {
		mic::types::MatrixPtr<eT> batch_dy = g['y'];
		std::cout << "backward gradient dy: min:" << (*batch_dy).minCoeff() <<" max: " << (*batch_dy).maxCoeff() << std::endl;

		//std::cout<<"backpropagade_dy_to_dx!\n";
		// To dx.
		backpropagade_dy_to_dx();

		//std::cout<<"backpropagade_dy_to_dW!\n";
		// To dW.
		backpropagade_dy_to_dW();

		//std::cout<<"backpropagade_dy_to_db!\n";
		// To db.
		backpropagade_dy_to_db();

		mic::types::MatrixPtr<eT> batch_dx = g['x'];
		std::cout << "backward gradient dx: min:" << (*batch_dx).minCoeff() <<" max: " << (*batch_dx).maxCoeff() << std::endl;


		for (size_t fi=0; fi< output_depth; fi++) {
			// A given filter (neuron layer) has in fact connection to all input channels.
			for (size_t ic=0; ic< input_depth; ic++) {
				mic::types::MatrixPtr<eT> dw = g['W'+std::to_string(fi)+"x"+std::to_string(ic)];
				std::cout << "backward gradient dW"<< fi <<"x"<< ic <<": min:" << (*dw).minCoeff() <<" max: " << (*dw).maxCoeff() << std::endl;
			}//: for
		}//: for

		mic::types::MatrixPtr<eT> db = g['b'];
		std::cout << "backward gradient db: min:" << (*db).minCoeff() <<" max: " << (*db).maxCoeff() << std::endl;

		//std::cout<<"After Backward!\n";
	}//: backward


	/*!
	 * Back-propagates the gradients from dy to dx.
	 */
	void backpropagade_dy_to_dx() {
		// Get matrices.
		mic::types::MatrixPtr<eT> batch_dy = g['y'];
		mic::types::MatrixPtr<eT> batch_x = s['x'];
		//std::cout<< "backpropagate to dx batch_x=\n" << (*batch_x) << std::endl;
		// Get output pointers - so the results will be stored!
		mic::types::MatrixPtr<eT> batch_dx = g['x'];

		// Backpropagate gradient from dy to dx.


		// Calculate dx for a given batch of dy gradients.
		// Iterate through samples in the input batch.
//#pragma omp parallel for
		for (size_t ib=0; ib< batch_size; ib++) {

			// Get y gradient sample from batch.
			mic::types::MatrixPtr<eT> gys = m["ys"];
			(*gys) = batch_dy->col(ib);

			/*///////////////////////
			// Get gradient y channel.
			mic::types::MatrixPtr<eT> gyc = m["yc"];
			for (size_t fi=0; fi< number_of_filters; fi++) {
				(*gyc) = gys->block(fi*output_height*output_width, 0, output_height*output_width, 1);
				(*gyc).resize(output_height, output_width);
				std::cout<< "============  dy channel = " << fi << "(*gyc) = \n" << (*gyc) << std::endl;
			}// to remove*/


			// Get pointer to x gradient sample matrix.
			mic::types::MatrixPtr<eT> gxs = m["xs"];
			gxs->setZero();

			// Convolve reWs with sample channel by channel.
			for (size_t ic=0; ic< input_depth; ic++) {
				//std::cout<< "======  switching input channel = " << ic << std::endl;

				// Get pointer to x gradient channel "storage".
				mic::types::MatrixPtr<eT> gxc = m["xc"];
				// Clean it up!
				gxc->setZero();
				// Resize just in case.
				gxc->resize(input_height, input_width);


				// For each filter.
				for (size_t fi=0; fi< output_depth; fi++) {
					//std::cout<< "======  switching filter = " << fi << std::endl;

					// Get filter weight matrix.
					mic::types::MatrixPtr<eT> W = p["W"+std::to_string(fi)+"x"+std::to_string(ic)];
					W->resize(filter_size, filter_size);

					// Get gradient y channel.
					mic::types::MatrixPtr<eT> gyc = m["yc"];
					// Copy block - resizes the input channel matrix.
					(*gyc) = gys->block(fi*output_height*output_width, 0, output_height*output_width, 1);
					(*gyc).resize(output_height, output_width);
					//std::cout<< "fi = " << fi << "(*gyc) = \n" << (*gyc) << std::endl;

/*					// Get reW.
					mic::types::MatrixPtr<eT> reW = m["reW"+std::to_string(fi)+std::to_string(ic)];
					// Get pointer to "reverse receptive field".
					mic::types::MatrixPtr<eT> rerf = m["rerf"];*/

					// Iterate through "stride blocks" - their number is equal to output size.
					// Those are also coordinates of the outputs.
					for(size_t oy=0; oy < output_height; oy++) {
						for(size_t ox=0; ox < output_width; ox++) {
							// Get top upper coordinate of input block.
							size_t isby = oy * stride;
							size_t isbx = ox * stride;
							//std::cout<< "block coordinates: isby = " << isby << " isbx=" << isbx << std::endl;

							// Iterate through the elements and add them to gxc.
							for (size_t iy=0; iy<filter_size; iy++) {
								for (size_t ix=0; ix<filter_size; ix++) {

									eT conv = (*W)(iy,ix)*(*gyc)(oy,ox);//((*rerf)*(*gyc))(0);//1;
									size_t y = isby+iy;
									size_t x = isbx+ix;
									//std::cout<< "adding to (*gxc) = \n" << (*gxc) << "\n in: " << " y=" << y<< " x=" << x << std::endl;
									(*gxc)(y,x) += conv;

								}//: for ix
							}//: for iy


						}//: for stride blocks y
					}//: for stride blocks x
					//std::cout<< "filter = "<< fi << " resulting (*gxc) = \n" << (*gxc) << std::endl;
				}//: for filters
				//std::cout<< "result for input channel = "<< ic << " (*gxc) = \n" << (*gxc) << std::endl;
				// Resize gradient channel to a column vector.
				gxc->resize(input_height * input_width, 1);

				// Concatenate gradient.
				gxs->block(ic*input_height*input_width, 0, input_height*input_width, 1)
						= (*gxc);

				//std::cout<< "(*gs) = \n" << (*gxs) << std::endl;
			}//: for channels

			// Set column in the gradient batch.
			batch_dx->col(ib) = (*gxs);

		}//: batch

	}

	/*!
	 * Back-propagates the gradients from dy to dW.
	 */
	void backpropagade_dy_to_dW() {
		// Get weight delta matrix.
		//mic::types::MatrixPtr<eT> dW = g['W'];

		// Get matrices.
		mic::types::MatrixPtr<eT> batch_dy = g['y'];
		mic::types::MatrixPtr<eT> batch_x = s['x'];
		//std::cout<< "backpropagate to dW batch_x=\n" << (*batch_x) << std::endl;

		// Reset weight gradiends.
		for (size_t fi=0; fi< output_depth; fi++) {
			// A given filter (neuron layer) has in fact connection to all input channels.
			for (size_t ic=0; ic< input_depth; ic++) {
				g["W"+std::to_string(fi)+"x"+std::to_string(ic)]->setZero();
			}//: for ic
		}//: for fi


		// Iterate through samples in the input batch.
		for (size_t ib=0; ib< batch_size; ib++) {

			// Get y gradient sample from batch.
			mic::types::MatrixPtr<eT> gys = m["ys"];
			(*gys) = batch_dy->col(ib);

			// Get x sample from batch.
			mic::types::MatrixPtr<eT> xs = m["xs"];
			(*xs) = batch_x->col(ib);
			//std::cout<< "xs=\n" << (*xs) << std::endl;

			// Iterate through input channels.
			for (size_t ic=0; ic< input_depth; ic++) {
				// 3.1. Get input channel from image.
				mic::types::MatrixPtr<eT> x_channel = m["xc"];
				// Copy block - resizes the input channel matrix.
				(*x_channel) = xs->block(ic*input_height*input_width, 0, input_height*input_width, 1);
				// Resize channel using the given dimensions.
				x_channel->resize(input_height, input_width);
				//std::cout<< "======  switching input channel = " << ic << " x_channel=\n" << (*x_channel) << std::endl;

				// Fill "inverse input receptive fields" from given input channel.
				// Image coordinates: ix, iy.
				// Coordinates in the filter space: fx, fy.
				for (size_t fy=0; fy< filter_size; fy++) {
					for (size_t fx=0; fx< filter_size; fx++) {
						// Get inverse receptive field matrix.
						mic::types::MatrixPtr<eT> ixrf = m["ixrf"+std::to_string(fy)+"x"+std::to_string(fx)];
						ixrf->setZero();
						//std::cout << (*x_field).rows() << "x" << (*x_field).cols() <<std::endl;
						//std::cout<< "x_field=\n" << (*x_field) << std::endl;
						ixrf->resize(output_height, output_width);
						ixrf->setZero();
						// Iterate through the input channel using stride.
						for (size_t iy=0; iy< output_height; iy++) {
							for (size_t ix=0; ix< output_width; ix++) {
								/*std::cout<<"fy =" << fy <<" fx =" << fx <<" iy =" << iy <<" ix =" << ix << std::endl;
								std::cout<<"fy*iy*stride =" << fy*iy*stride <<" fx+ix*stride =" << fx+ix*stride <<std::endl;*/
								// Copy cell - one by one :]
								(*ixrf)(iy, ix) = (*x_channel)(fy+iy*stride,fx+ix*stride);

							}//: for ix
						}//: for iy
						//(*ixrf) = (*x_channel).block(fy, fx, output_height, output_width);
						// Resize the field to a column vector.
						ixrf->resize(1, output_height*output_width);
						//std::cout<< "x_field=\n" << (*x_field) << std::endl;
					}//: for rx
				}//: for ry

				// For each filter (= each output channel).
				for (size_t fi=0; fi< output_depth; fi++) {
					// Get output channel for a given filter.
					mic::types::MatrixPtr<eT> gyc = m["yc"+std::to_string(fi)];
					(*gyc) = gys->block(fi*output_height*output_width, 0, output_height*output_width, 1);
					//std::cout<< "gyc=\n" << (*gyc) << std::endl;

					// Get matrix of a given "part of a given neuron".
					mic::types::MatrixPtr<eT> dW = g["W"+std::to_string(fi)+"x"+std::to_string(ic)];
					// Not required - just in case. :]
					dW->resize(filter_size, filter_size);
					// Iterate through inverse receptive fields and CONVOLVE.
					for (size_t ry=0; ry< filter_size; ry++) {
						for (size_t rx=0; rx< filter_size; rx++) {
							//std::cout<<"filter = " << fi << " ry =" << ry <<" rx =" << rx <<std::endl;
							// Get inverse receptive field matrix of size (filter_size^2, 1)...
							mic::types::MatrixPtr<eT> ixrf = m["ixrf"+std::to_string(ry)+"x"+std::to_string(rx)];
							ixrf->resize(1, output_height*output_width);
							/*std::cout<< "x_field=\n" << (*ixrf) << std::endl;
							std::cout<< "gyc=\n" << (*gyc) << std::endl;
							std::cout<< " result = \n" << ((*ixrf)*(*gyc)) << std::endl;*/
							// ... and convolve it with dy channel.
							(*dW)(ry, rx) += ((*ixrf)*(*gyc))(0);
						}//: for rx
					}//: for ry
					//std::cout << "==== result: dW [" << fi << ic <<"] = " << (*dW)<<std::endl;
					//std::cout<<"fi = " << fi << std::endl<< (*dW) << std::endl;
					//dW->setValue(fi);

				}//: for filter

			}//: for input_channels
		}//: for batch
	}



	/*!
	 * Back-propagates the gradients from dy to dx.
	 */
	void backpropagade_dy_to_db() {
		// Get bias delta matrix (vector).
		mic::types::MatrixPtr<eT> db = g['b'];
		// Get dy matrix - input.
		mic::types::MatrixPtr<eT> batch_dy = g['y'];

		// Iterate through output channels i.e. filters.
		for (size_t fi=0; fi< output_depth; fi++) {
			// Sum block [output_channel x batch].
			eT channel_bach_sum = batch_dy->block(fi*output_height*output_width, 0, output_height*output_width, batch_size).sum();
			//std::cout<< "fi = " << fi << "channel_bach_sum = " << channel_bach_sum << std::endl;
			(*db)[fi] = channel_bach_sum;
		}//: for filters

		// And that's it! :)
	}


	/*!
	 * Resets all gradients. Warning: including gx and gy!
	 */
	void resetGrads()  {
		// Reset array matrix with gradients.
		g.setZero();
	}


	/*!
	 * Applies the gradient update, using the selected optimization method.
	 * @param alpha_ Learning rate - passed to the optimization functions of all layers.
	 * @param decay_ Weight decay rate (determining that the "unused/unupdated" weights will decay to 0) (DEFAULT=0.0 - no decay).
	 */
	void update(eT alpha_, eT decay_  = 0.0f) {
		std::cout<<"update\n";
		// Get keys of all parameters.
		std::map<std::string, size_t> keys = p.keys();

		for (auto& i: keys) {
/*			std::cout << "* " << i.first << "\t = < " << (*p[i.first]).minCoeff() << ",\t " << (*p[i.first]).maxCoeff() << ">" <<
					" * d" << i.first << "\t = <" << (*g[i.first]).minCoeff() << ",\t " << (*g[i.first]).maxCoeff() << ">" << std::endl;*/
			opt[i.first]->update(p[i.first], g[i.first], 1.0*alpha_, decay_);
			/*for (size_t j=0; j<(size_t)p[i.first]->size(); j++)
				(*p[i.first])[j] -= alpha_ * (*g[i.first])[j];*/
		}//: for

		/*std::string key = "W0x0";
		std::cout << "* " << key << "\t = < " << (*p[key])[0]  <<",\t " << (*p[key])[1];
		std::cout << "\t* d" << key << "\t = < " << (*g[key])[0]  <<",\t " << (*g[key])[1] << std::endl;*/

	}



	/*!
	 * Returns activations of weights.
	 */
	std::vector< std::shared_ptr <mic::types::Matrix<eT> > > & getWeightActivations(bool normalize_ = true) {

		// Allocate memory.
		lazyAllocateMatrixVector(w_activations, output_depth*input_depth, filter_size*filter_size, 1);

		// Iterate through filters and generate "activation image" for each one.
		for (size_t fi=0; fi< output_depth; fi++) {

			// Iterate through input channels.
			for (size_t ic=0; ic< input_depth; ic++) {
				// Get matrix of a given "part of a given neuron".
				mic::types::MatrixPtr<eT> W = p["W"+std::to_string(fi)+"x"+std::to_string(ic)];

				// Get row.
				mic::types::MatrixPtr<eT> row = w_activations[fi*input_depth + ic];
				// Copy data.
				(*row) = (*W);
				row->resize(filter_size, filter_size);

				std::cout << "weight W"<< fi <<"x"<< ic <<": min:" << (*W).minCoeff() <<" max: " << (*W).maxCoeff() << std::endl;
				// Normalize.
				if (normalize_ )
					normalizeMatrixForVisualization(row);
				//std::cout << "normalized W"<< fi <<"x"<< ic <<": min:" << (*row).minCoeff() <<" max: " << (*row).maxCoeff() << std::endl;
			}//: for channels
		}//: for filters

		// Return activations.
		return w_activations;
	}



	/*!
	 * Returns activations of weight gradients (dx).
	 */
	std::vector< std::shared_ptr <mic::types::Matrix<eT> > > & getWeightGradientActivations(bool normalize_ = true) {

		// Allocate memory.
		lazyAllocateMatrixVector(dw_activations, output_depth * input_depth, filter_size*filter_size, 1);

		// Iterate through filters and generate "activation image" for each one.
		for (size_t fi=0; fi< output_depth; fi++) {

			// Iterate through input channels.
			for (size_t ic=0; ic< input_depth; ic++) {

				// Get matrix of a given "part of a given neuron dW".
				mic::types::MatrixPtr<eT> W = g["W"+std::to_string(fi)+"x"+std::to_string(ic)];

				// Get row.
				mic::types::MatrixPtr<eT> row = dw_activations[fi*input_depth + ic];
				// Copy data.
				(*row) = (*W);

				// Normalize.
				if (normalize_ )
					normalizeMatrixForVisualization(row);
			}//: for channel
		}//: for filter

		// Return activations.
		return dw_activations;
	}


	/*!
	 * Returns activations of receptive fields.
	 * Limitation: displays receptive fields of the last sample from batch!
	 */
	std::vector< std::shared_ptr <mic::types::Matrix<eT> > > & getReceptiveFields(bool normalize_ = true) {

		// TODO: batch!
		//assert(batch_size==1);

		// Allocate memory.
		lazyAllocateMatrixVector(xrf_activations, output_height * output_width, filter_size, filter_size);

		// Receptive field "id" coordinates: rx, ry.
		for (size_t ry=0; ry< output_height; ry++) {
			for (size_t rx=0; rx< output_width; rx++) {
				// Get receptive field matrix...
				mic::types::MatrixPtr<eT> xrf = m["xrf"+std::to_string(ry)+"x"+std::to_string(rx)];

				// Get activation "row".
				mic::types::MatrixPtr<eT> row = xrf_activations[ry*output_width + rx];

				// Copy field.
				(*row) = *(xrf);
				row->resize(filter_size, filter_size);

				// Normalize.
				if (normalize_ )
					normalizeMatrixForVisualization(row);
			}//: for ry
		}//: for rx

		// Return activations.
		return xrf_activations;
	}

	/*!
	 * Returns activations of inverse receptive fields.
	 * Limitation: displays receptive fields of the last sample from batch!
	 */
	std::vector< std::shared_ptr <mic::types::Matrix<eT> > > & getInverseReceptiveFields(bool normalize_ = true) {

		// TODO: batch!
		//assert(batch_size==1);

		// Allocate memory.
		lazyAllocateMatrixVector(irf_activations, filter_size*filter_size, output_height, output_width);

		for (size_t fy=0; fy< filter_size; fy++) {
			for (size_t fx=0; fx< filter_size; fx++) {
				// Get inverse receptive field matrix.
				mic::types::MatrixPtr<eT> ixrf = m["ixrf"+std::to_string(fy)+"x"+std::to_string(fx)];

				// Get activation "row".
				mic::types::MatrixPtr<eT> row = irf_activations[fy*filter_size + fx];

				// Copy field.
				(*row) = *(ixrf);
				row->resize(output_height, output_width);

				// Normalize.
				if (normalize_ )
					normalizeMatrixForVisualization(row);
				}//: for rx
			}//: for ry

		// Return activations.
		return irf_activations;
	}


	// Unhide the overloaded methods inherited from the template class Layer fields via "using" statement.
	using Layer<eT>::forward;
	using Layer<eT>::backward;

protected:
	// Unhide the fields inherited from the template class Layer via "using" statement.
    using Layer<eT>::g;
    using Layer<eT>::s;
    using Layer<eT>::p;
    using Layer<eT>::m;
    using Layer<eT>::opt;

    // Uncover "sizes" for visualization.
    using Layer<eT>::input_height;
    using Layer<eT>::input_width;
    using Layer<eT>::input_depth;
	using Layer<eT>::output_height;
	using Layer<eT>::output_width;
	using Layer<eT>::output_depth;
    using Layer<eT>::batch_size;

	/// Size of filters (assuming square filters). Filter_size^2 = length of the output vector.
	size_t filter_size;

	/// Stride (assuming equal vertical and horizontal strides).
	 size_t stride;

	 // Uncover methods useful in visualization.
	 using Layer<eT>::lazyAllocateMatrixVector;
	 using Layer<eT>::normalizeMatrixForVisualization;

private:
	// Friend class - required for using boost serialization.
	template<typename tmp> friend class MultiLayerNeuralNetwork;

	/// Vector containing activations of weights/filters.
	std::vector< std::shared_ptr <mic::types::MatrixXf> > w_activations;

	/// Vector containing activations of gradients of weights (dW).
	std::vector< std::shared_ptr <mic::types::MatrixXf> > dw_activations;

	/// Vector containing receptive fields.
	std::vector< std::shared_ptr <mic::types::MatrixXf> > xrf_activations;

	/// Vector containing inverse receptive fields.
	std::vector< std::shared_ptr <mic::types::MatrixXf> > irf_activations;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	Convolution<eT>() : Layer<eT> () { }

};

} /* convolution */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_CONVOLUTION_HPP_ */
