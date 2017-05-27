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
 * \brief Class representing a convolution layer.
 * \author tkornuta/krocki
 * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT=float>
class Convolution : public mic::mlnn::Layer<eT> {
public:

	/*!
	 * Creates the a convolutional layer.
	 * @param input_height_ Height of the input / rows (e.g. 28 for MNIST).
	 * @param input_width_ Width of the input / cols (e.g. 28 for MNIST).
	 * @param input_channels_ Number of channels of the input (e.g. 3 for RGB images).
	 * @param number_of_filters_ Number of filters = Length of the output vector.
	 * @param filter_size_ Size of filters (assuming square filters).
	 * @param stride_ Stride (assuming equal vertical and horizontal strides).
	 * @param name_ Name of the layer.
	 */
	Convolution<eT>(size_t input_height_, size_t input_width_, size_t input_channels_, size_t number_of_filters_, size_t filter_size_, size_t stride_, std::string name_ = "Convolution") :
		Layer<eT>(input_width_ * input_height_ * input_channels_, number_of_filters_ * ((input_height_ - filter_size_)/stride_ + 1) * ((input_width_ - filter_size_)/stride_ + 1), 1, LayerTypes::Convolution, name_),
	    input_width(input_width_),
	    input_channels(input_channels_),
		input_height(input_height_),
		filter_size(filter_size_),
		stride(stride_),
		number_of_filters(number_of_filters_)
	{

/*		std::cout<<"input_height = " << input_height <<std::endl;
		std::cout<<"input_width = " << input_width <<std::endl;
		std::cout<<"number_of_filters = " << number_of_filters <<std::endl;
		std::cout<<"filter_size = " << filter_size <<std::endl;
		std::cout<<"stride = " << stride <<std::endl;*/

		// Calculate number of receptive fields within a "single input channel".
		number_of_receptive_fields_vertical = (input_height - filter_size)/stride + 1;
		number_of_receptive_fields_horizontal = (input_width - filter_size)/stride + 1;
/*		std::cout<<"number_of_receptive_fields_vertical = " << number_of_receptive_fields_vertical <<std::endl;
		std::cout<<"number_of_receptive_fields_horizontal = " << number_of_receptive_fields_horizontal <<std::endl;*/
		// Filters must "exactly" fit!
		assert((number_of_receptive_fields_vertical - 1) * stride + filter_size == input_height);
		assert((number_of_receptive_fields_horizontal - 1) * stride + filter_size == input_width);

		// Calculate "range" - for initialization.
		eT range_init = (eT) (input_width * input_height * input_channels) +
				(number_of_receptive_fields_horizontal * number_of_receptive_fields_vertical * number_of_filters);
		eT range = sqrt(6.0 / range_init);

		// Create filters.
		for (size_t fi=0; fi< number_of_filters; fi++) {
			// A given filter (neuron layer) has in fact connection to all input channels.
			for (size_t ic=0; ic< input_channels; ic++) {
				// Create the weights matrix - a row vector.
				p.add ("W"+std::to_string(fi)+std::to_string(ic), 1, filter_size*filter_size);
				// Initialize weights of the W matrix.
				p["W"+std::to_string(fi)+std::to_string(ic)]->rand(-range, range);
				//std::cout<<"W.dims = " << p["W"+std::to_string(fi)+std::to_string(ic)]->rows() << "," << p["W"+std::to_string(fi)+std::to_string(ic)]->cols() << "\n";

				// Create the weights matrix for updates/gradients.
				g.add ("W"+std::to_string(fi)+std::to_string(ic), 1, filter_size*filter_size);
			}//: for input channels.

		}//: for filter

		// Create a single bias vector for all filters.
		p.add ("b", number_of_filters, 1);
		p["b"]->setZero();
		// Bias gradient.
		g.add ("b", number_of_filters, 1);

		// Allocate (temporary) memory for "input receptive fields".
		for (size_t ry=0; ry< number_of_receptive_fields_vertical; ry++) {
			for (size_t rx=0; rx< number_of_receptive_fields_horizontal; rx++) {
				// Create receptive field matrix.
				m.add ("xrf"+std::to_string(ry)+std::to_string(rx), filter_size, filter_size);
			}//: for
		}//: for

		// Allocate (temporary) memory for "input sample" - column vector.
		m.add ("xs", input_channels*input_height*input_width, 1);

		// Allocate (temporary) memory for "input channel" - column vector.
		m.add ("xc", input_height*input_width, 1);

		// Allocate (temporary) memory for "output channels" - matrices.
		for (size_t fi=0; fi< number_of_filters; fi++) {
			// Create output channel matrix.
			m.add ("yc"+std::to_string(fi), number_of_receptive_fields_vertical, number_of_receptive_fields_horizontal);
		}//: for

		// Allocate (temporary) memory for "output sample" - a column vector of all channels of a given sample.
		m.add ("ys", number_of_filters*number_of_receptive_fields_vertical*number_of_receptive_fields_horizontal, 1);

		// Allocate memory for "full convolution" (180 degree rotated and expanded) filters.
		for (size_t fi=0; fi< number_of_filters; fi++) {
			// A given filter (neuron layer) has in fact connection to all input channels.
			for (size_t ic=0; ic< input_channels; ic++) {
				m.add ("reW"+std::to_string(fi)+std::to_string(ic), (filter_size+(1+stride)*(number_of_receptive_fields_vertical-1)), (filter_size+(1+stride)*(number_of_receptive_fields_horizontal-1)));
				m["reW"+std::to_string(fi)+std::to_string(ic)]->setZero();
			}//: for channels
		}//: for filters

		// Allocate memory for "reverse" receptive field.
		m.add ("rerf", number_of_receptive_fields_vertical, number_of_receptive_fields_horizontal);

		// Allocate memory for "gradient x channel".
		m.add ("gxc", input_height, input_width);

		// Allocate memory for "gradient x sample".
		m.add ("gxs", input_channels * input_height * input_width, 1);

		// Allocate memory for "gradient y sample".
		m.add ("gys", number_of_filters * number_of_receptive_fields_vertical * number_of_receptive_fields_horizontal, 1);

		// Allocate memory for "gradient y channel".
		m.add ("gyc", number_of_receptive_fields_vertical * number_of_receptive_fields_horizontal, 1);

		// Allocate (temporary) memory for "inverse input receptive fields" - used in backpropagation.
		for (size_t ry=0; ry< filter_size; ry++) {
			for (size_t rx=0; rx< filter_size; rx++) {
				// Create receptive field matrix.
				m.add ("ixrf"+std::to_string(ry)+std::to_string(rx), number_of_receptive_fields_vertical, number_of_receptive_fields_horizontal);
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
	 * Performs forward pass through the filters. Can process batches.
	 */
	void forward(bool test = false) {
		// Get input matrix.
		mic::types::MatrixPtr<eT> batch = s['x'];
		//std::cout<< "forward batch_x=\n" << (*batch) << std::endl;

		// Iterate through samples in the input batch.
		for (size_t ib=0; ib< batch_size; ib++) {

			// 1. "Reset" output "channels".
			for (size_t fi=0; fi< number_of_filters; fi++) {
				// Get output channel matrix.
				mic::types::MatrixPtr<eT> y_channel = m["yc"+std::to_string(fi)];
				// Resize it to a matrix (!).
				y_channel->resize(number_of_receptive_fields_vertical, number_of_receptive_fields_horizontal);
				// And "reset" i.e. set value of all cells to bias (so we can skip adding it later).
				y_channel->setValue((*p["b"])[fi]);
			}//: filters

			// 2. Get input sample from batch!
			mic::types::MatrixPtr<eT> sample = m["xs"];
			(*sample) = batch->col(ib);
			//std::cout<< "sample=\n" << (*sample) << std::endl;

			// 3. Iterate through input channels.
			for (size_t ic=0; ic< input_channels; ic++) {
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
				// Receptive fields coordinates: rx, ry.
				for (size_t ry=0, iy = 0; ry< number_of_receptive_fields_vertical; ry++, iy+=stride) {
					for (size_t rx=0, ix = 0; rx< number_of_receptive_fields_horizontal; rx++, ix+=stride) {
						//std::cout<<"ry =" << ry <<" rx =" << rx <<" iy =" << iy <<" ix =" << ix << std::endl;
						// Get receptive field matrix...
						mic::types::MatrixPtr<eT> field = m["xrf"+std::to_string(ry)+std::to_string(rx)];
						// Copy block from channel - resizes the field matrix.
						(*field) = ichannel->block(iy,ix,filter_size, filter_size);
						//std::cout<< "field=\n" << (*field) << std::endl;
						// Resize the field to a column vector.
						field->resize(filter_size*filter_size, 1);
					}//: for rx
				}//: for ry

				// 3.3. Convolve receptive fields with filters.
				// Iterate through filters and calculate the result of the convolution.
				for (size_t fi=0; fi< number_of_filters; fi++) {
					// Get output channel for a given filter.
					mic::types::MatrixPtr<eT> y_channel = m["yc"+std::to_string(fi)];
					//std::cout << "====  switching to oc = \n" << (*y_channel)<<std::endl;
					// Get "part of a given neuron" responding to a given input channel.
					mic::types::MatrixPtr<eT> W = p["W"+std::to_string(fi)+std::to_string(ic)];
					// Not required - just in case. :]
					W->resize(1,filter_size*filter_size);
					// Iterate through receptive fields.
					for (size_t ry=0; ry< number_of_receptive_fields_vertical; ry++) {
						for (size_t rx=0; rx< number_of_receptive_fields_horizontal; rx++) {
							// Get receptive field matrix of size (1, filter_size^2)...
							mic::types::MatrixPtr<eT> xrf = m["xrf"+std::to_string(ry)+std::to_string(rx)];
							// ... and result of "convolution" of that filter with the part of the input "below" the receptive field.
							/*std::cout<<"ic = " << ic  << "filter = " << fi << " ry =" << ry <<" rx =" << rx <<std::endl;
							std::cout<< "W=\n" << (*W) << std::endl;
							std::cout<< "xrf=\n" << (*xrf) << std::endl;
							std::cout<< " result = " << ((*W)*(*xrf)) << std::endl;*/
							(*y_channel)(ry, rx) += ((*W)*(*xrf))(0);
						}//: for rx
					}//: for ry
					//std::cout << "====  ic = " << ic << "filter= " << fi << " oc = \n" << (*y_channel)<<std::endl;
				}//: for filters
			}//: for channels
		}//: for batch

		// 3. Produce output.
		// Create output batch from samples one by one.

		// Get output pointer - so the results will be stored!
		mic::types::MatrixPtr<eT> y = s['y'];
		y->setZero();

		// Iterate through samples in the input batch.
		for (size_t ib=0; ib< batch_size; ib++) {
			// "Concatenate" output channels into one vector - "output sample".
			mic::types::MatrixPtr<eT> y_sample = m["ys"];
			y_sample->setZero();
			for (size_t fi=0; fi< number_of_filters; fi++) {
				// Get output channel for a given filter.
				mic::types::MatrixPtr<eT> y_channel = m["yc"+std::to_string(fi)];
				// Resize output channel to a column vector.
				y_channel->resize(number_of_receptive_fields_vertical*number_of_receptive_fields_horizontal, 1);
				//std::cout << "filter= " << fi << " oc = " << (*y_channel)<<std::endl;

				// Concatenate.
				y_sample->block(fi*number_of_receptive_fields_vertical*number_of_receptive_fields_horizontal, 0, number_of_receptive_fields_vertical*number_of_receptive_fields_horizontal, 1)
						= (*y_channel);
				//std::cout <<"os =" << (*y_sample)<<std::endl;
			}//: for filters
			/*std::cout<<"os.dims = " << y_sample->rows() << "," << y_sample->cols() << "\n";
			std::cout<<"y.dims = " << y->rows() << "," << y->cols() << "\n";*/
			// Set column in the output batch.
			y->col(ib) = (*y_sample);

		}//: for batch
	}//: forward

	/*!
	 * Back-propagates the gradients through the layer.
	 */
	void backward() {
		// To dx.
		backpropagade_dy_to_dx();

		// To dW.
		backpropagade_dy_to_dW();

		// To db.
		backpropagade_dy_to_db();

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

		// reW sizes.
		size_t rew_height = filter_size+(stride+1)*(number_of_receptive_fields_vertical-1);
		size_t rew_width = filter_size+(stride+1)*(number_of_receptive_fields_horizontal-1);

		// Backpropagate gradient from dy to dx.

		// 1. Fill the "rotated-expanded" filters.
		for (size_t ic=0; ic< input_channels; ic++) {
			//std::cout<<"Channel: " << ic <<std::endl;
			for (size_t fi=0; fi< number_of_filters; fi++) {
				mic::types::MatrixPtr<eT> reW = m["reW"+std::to_string(fi)+std::to_string(ic)];
				reW->setZero();
				// Get filter.
				mic::types::MatrixPtr<eT> W = p["W"+std::to_string(fi)+std::to_string(ic)];
				W->resize(filter_size, filter_size);
				// "Rotate" and insert filter values into right positions of "rotated-expanded" filter.
				for(size_t y=0; y < filter_size; y++)
					for(size_t x=0; x < filter_size; x++)
					(*reW)(stride*y + (number_of_receptive_fields_vertical-1), stride*x + (number_of_receptive_fields_horizontal -1)) = (*W)(filter_size-y-1,filter_size-x-1);
				//std::cout<<"reW=\n"<<(*reW)<<std::endl;
				// Resize filter matrix W back to a row vector.
				W->resize(1, filter_size*filter_size);
			}//: for filters
		}//: for channels

		// 2. Calculate dx for a given gradient batch.
		// Iterate through samples in the input batch.
		for (size_t ib=0; ib< batch_size; ib++) {

			// Get y gradient sample from batch.
			mic::types::MatrixPtr<eT> gys = m["gys"];
			(*gys) = batch_dy->col(ib);

			// Get pointer to x gradient sample matrix.
			mic::types::MatrixPtr<eT> gxs = m["gxs"];
			gxs->setZero();

			// Convolve reWs with sample channel by channel.
			for (size_t ic=0; ic< input_channels; ic++) {
				//std::cout<<"Input channel: " << ic <<std::endl;

				// Get pointer to x gradient channel "storage".
				mic::types::MatrixPtr<eT> gxc = m["gxc"];
				// Clean it up!
				gxc->setZero();
				// Resize just in case.
				gxc->resize(input_height, input_width);

				// For each filter.
				for (size_t fi=0; fi< number_of_filters; fi++) {

					// Get gradient y channel.
					mic::types::MatrixPtr<eT> gyc = m["gyc"];
					// Copy block - resizes the input channel matrix.
					(*gyc) = gys->block(fi*number_of_receptive_fields_vertical*number_of_receptive_fields_horizontal, 0, number_of_receptive_fields_vertical*number_of_receptive_fields_horizontal, 1);
					(*gyc).resize(number_of_receptive_fields_vertical*number_of_receptive_fields_horizontal,1);
					//std::cout<< "fi = " << fi << "(*gyc) = \n" << (*gyc) << std::endl;

					// Get reW.
					mic::types::MatrixPtr<eT> reW = m["reW"+std::to_string(fi)+std::to_string(ic)];
					// Get pointer to "reverse receptive field".
					mic::types::MatrixPtr<eT> rerf = m["rerf"];
					for (size_t y=0, rew_y= rew_height - number_of_receptive_fields_vertical; y< input_height; y++, rew_y--) {
						for (size_t x=0, rew_x= rew_width - number_of_receptive_fields_horizontal; x< input_width; x++, rew_x--) {
							// Get block under "reverse receptive field".
							(*rerf) = reW->block(rew_y, rew_x, number_of_receptive_fields_vertical, number_of_receptive_fields_horizontal);
							rerf->resize(1, number_of_receptive_fields_vertical * number_of_receptive_fields_horizontal);
							/*std::cout<< "filter = " << fi <<" y=" << y<< " x=" << x <<" rew_y=" << rew_y<< " rew_x=" << rew_x << std::endl;
							std::cout<< "     (*rerf) = \n" << (*rerf) << std::endl;
							std::cout<< "     (*gyc) = \n" << (*gyc) << std::endl;
							std::cout<< "     calculated value= " << ((*rerf)*(*gyc))(0) << std::endl;*/
							// Convolve.
							(*gxc)(y,x) += ((*rerf)*(*gyc))(0);
							//std::cout<< "     tmp (*gxc) = \n" << (*gxc) << std::endl;
						}//: x
					}//: y
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

		// Iterate through samples in the input batch.
		for (size_t ib=0; ib< batch_size; ib++) {

			// Get y gradient sample from batch.
			mic::types::MatrixPtr<eT> gys = m["gys"];
			(*gys) = batch_dy->col(ib);

			// Get x sample from batch.
			mic::types::MatrixPtr<eT> xs = m["xs"];
			(*xs) = batch_x->col(ib);
			//std::cout<< "xs=\n" << (*xs) << std::endl;

			// Iterate through input channels.
			for (size_t ic=0; ic< input_channels; ic++) {
				// 3.1. Get input channel from image.
				mic::types::MatrixPtr<eT> x_channel = m["xc"];
				// Copy block - resizes the input channel matrix.
				(*x_channel) = xs->block(ic*input_height*input_width, 0, input_height*input_width, 1);
				// Resize channel using the given dimensions.
				x_channel->resize(input_height, input_width);
				//std::cout<< "======  switching input channel = " << ic << " x_channel=\n" << (*x_channel) << std::endl;

				// Fill "inverse input receptive fields" from given input channel.
				// Image coordinates: ix, iy.
				// Receptive fields coordinates: rx, ry.
				for (size_t ry=0; ry< filter_size; ry++) {
					for (size_t rx=0; rx< filter_size; rx++) {
						// Get inverse receptive field matrix.
						mic::types::MatrixPtr<eT> x_field = m["ixrf"+std::to_string(ry)+std::to_string(rx)];
						//std::cout << (*x_field).rows() << "x" << (*x_field).cols() <<std::endl;
						//std::cout<< "x_field=\n" << (*x_field) << std::endl;
						x_field->resize(number_of_receptive_fields_vertical, number_of_receptive_fields_horizontal);
						// Iterate through the input channel using stride.
						for (size_t iy=0, fy=0; fy< number_of_receptive_fields_vertical; iy+=stride, fy++) {
							for (size_t ix=0, fx=0; fx< number_of_receptive_fields_horizontal; ix+=stride, fx++) {
								//std::cout<<"ry =" << ry <<" rx =" << rx <<" iy =" << iy <<" ix =" << ix << std::endl;
								// Copy cell - one by one :]
								(*x_field)(fy, fx) = (*x_channel)(ry+iy,rx+ix);

							}//: for ix
						}//: for iy
						// Resize the field to a column vector.
						x_field->resize(1, number_of_receptive_fields_vertical*number_of_receptive_fields_horizontal);
						//std::cout<< "x_field=\n" << (*x_field) << std::endl;
					}//: for rx
				}//: for ry

				// For each filter (= each output channel).
				for (size_t fi=0; fi< number_of_filters; fi++) {
					// Get output channel for a given filter.
					mic::types::MatrixPtr<eT> gyc = m["yc"+std::to_string(fi)];
					(*gyc) = gys->block(fi*number_of_receptive_fields_vertical*number_of_receptive_fields_horizontal, 0, number_of_receptive_fields_vertical*number_of_receptive_fields_horizontal, 1);
					//std::cout<< "gyc=\n" << (*gyc) << std::endl;

					// Get matrix of a given "part of a given neuron".
					mic::types::MatrixPtr<eT> dW = g["W"+std::to_string(fi)+std::to_string(ic)];
					// Not required - just in case. :]
					dW->resize(filter_size, filter_size);
					// Iterate through inverse receptive fields and CONVOLVE.
					for (size_t ry=0; ry< filter_size; ry++) {
						for (size_t rx=0; rx< filter_size; rx++) {
							//std::cout<<"filter = " << fi << " ry =" << ry <<" rx =" << rx <<std::endl;
							// Get inverse receptive field matrix of size (filter_size^2, 1)...
							mic::types::MatrixPtr<eT> ixrf = m["ixrf"+std::to_string(ry)+std::to_string(rx)];
							/*std::cout<< "x_field=\n" << (*ixrf) << std::endl;
							std::cout<< "gyc=\n" << (*gyc) << std::endl;
							std::cout<< " result = \n" << ((*ixrf)*(*gyc)) << std::endl;*/
							// ... and convolve it with dy channel.
							(*dW)(ry, rx) += ((*ixrf)*(*gyc))(0);
						}//: for rx
					}//: for ry
					//std::cout << "==== result: dW [" << fi << ic <<"] = " << (*dW)<<std::endl;

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
		for (size_t fi=0; fi< number_of_filters; fi++) {
			// Sum block [output_channel x batch].
			eT channel_bach_sum = batch_dy->block(fi*number_of_receptive_fields_vertical*number_of_receptive_fields_horizontal, 0, number_of_receptive_fields_vertical*number_of_receptive_fields_horizontal, batch_size).sum();
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
		// Get keys of all parameters.
		std::map<std::string, size_t> keys = p.keys();

		for (auto& i: keys) {
			opt[i.first]->update(p[i.first], g[i.first], alpha_, decay_);
		}//: for
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
//    using Layer<eT>::input_size;
//    using Layer<eT>::output_size;
    using Layer<eT>::batch_size;
    using Layer<eT>::opt;

    /// Width of the input (e.g. 28 for MNIST).
    size_t input_width;

    /// Number of channels of the input (e.g. 3 for RGB images).
    size_t input_channels;

    /// Height of the input (e.g. 28 for MNIST).
    size_t input_height;

    /// Size of filters (assuming square filters). Filter_size^2 = length of the output vector.
	size_t filter_size;

	/// Stride (assuming equal vertical and horizontal strides).
	 size_t stride;

	/// Number of filters = number of output channels.
	size_t number_of_filters;

	/// Number of receptive fields in a single channel - horizontal direction.
	size_t number_of_receptive_fields_horizontal;

	/// Number of receptive fields in a single channel - vertical direction.
	size_t number_of_receptive_fields_vertical;

private:
	// Friend class - required for using boost serialization.
	template<typename tmp> friend class MultiLayerNeuralNetwork;


	/*!
	 * Private constructor, used only during the serialization.
	 */
	Convolution<eT>() : Layer<eT> () { }

};

} /* convolution */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_CONVOLUTION_HPP_ */
