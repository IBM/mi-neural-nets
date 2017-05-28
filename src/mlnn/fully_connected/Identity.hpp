/*!
 * \file Identity.hpp
 * \brief 
 * \author tkornut
 * \date Mar 31, 2016
 */

#ifndef SRC_MLNN_IDENTITY_HPP_
#define SRC_MLNN_IDENTITY_HPP_

#include <mlnn/layer/Layer.hpp>

namespace mic {
namespace mlnn {
namespace fully_connected {


/*!
 * \brief Identity layer - passes inputs/gradients without any modifications.
 * Implemented for testing purposes.
 * \author krocki/tkornuta
 * \tparam eT Template parameter denoting precision of variables (float for calculations/double for testing).
 */
template <typename eT=float>
class Identity : public mic::mlnn::Layer <eT> {
public:

	Identity<eT>(size_t inputs, size_t batch_size = 1, std::string name_ = "Identity") :
		Layer<eT>(inputs, inputs, batch_size, LayerTypes::Identity, name_) {

	}

	virtual ~Identity() {};

	void forward(bool test = false) {
		(*s['y']) = (*s['x']);
	}

	void backward() {
		(*g['x']) = (*g['y']);
	}

	/*!
	 * Performs the update according to the calculated gradients and injected optimization method. Empty as this is a "const" layer.
	 * @param alpha_ Learning rate - passed to the optimization functions of all layers.
	 * @param decay_ Weight decay rate (determining that the "unused/unupdated" weights will decay to 0) (DEFAULT=0.0 - no decay).
	 */
	virtual void update(eT alpha_, eT decay_  = 0.0f) { };

	// Unhide the overloaded methods inherited from the template class Layer fields via "using" statement.
	using Layer<eT>::forward;
	using Layer<eT>::backward;

protected:
	// Unhide the fields inherited from the template class Layer via "using" statement.
    using Layer<eT>::g;
    using Layer<eT>::s;
    using Layer<eT>::p;
    using Layer<eT>::m;
    using Layer<eT>::input_size;
    using Layer<eT>::output_size;
    using Layer<eT>::batch_size;

private:
	// Friend class - required for using boost serialization.
	template<typename tmp> friend class MultiLayerNeuralNetwork;

	/*!
	 * Private constructor, used only during the serialization.
	 */
	Identity<eT>() : Layer<eT> () { }

};


} /* namespace fully_connected */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_IDENTITY_HPP_ */
