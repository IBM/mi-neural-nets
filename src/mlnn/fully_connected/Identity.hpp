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
 * \author krocki
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
		// y = x;
		(*s['y']) = (*s['x']);
	}

	void backward() {
		// dx = dy;
		(*g['x']) = (*g['y']);
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
    using Layer<eT>::input_size;
    using Layer<eT>::output_size;
    using Layer<eT>::batch_size;

private:
	/*!
	 * Private constructor, used only during the serialization.
	 */
	Identity<eT>() : Layer<eT> () { }

};


} /* namespace fully_connected */
} /* namespace mlnn */
} /* namespace mic */

#endif /* SRC_MLNN_IDENTITY_HPP_ */
