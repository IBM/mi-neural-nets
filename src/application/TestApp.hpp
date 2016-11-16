/*!
 * \file TestApp.hpp
 * \brief 
 * \author tkornut
 * \date Mar 14, 2016
 */

#ifndef SRC_APPLICATION_SIMPLE_TEST_HPP_
#define SRC_APPLICATION_SIMPLE_TEST_HPP_

//#include <vector>
//#include <types/MatrixTypes.hpp>

namespace mic {
namespace neural_nets {
namespace application {

/*!
 * \brief Class implementing a n-Armed Bandits problem solving the n armed bandits problem using simple Q-learning rule.
 * \author tkornuta
 */
class TestApp: public mic::application::OpenGLApplication {
public:
	/*!
	 * Default Constructor. Sets the application/node name, default values of variables, initializes classifier etc.
	 * @param node_name_ Name of the application/node (in configuration file).
	 */
	TestApp(std::string node_name_ = "application");

	/*!
	 * Destructor.
	 */
	virtual ~TestApp();

protected:
	/*!
	 * Initializes all variables that are property-dependent.
	 */
	virtual void initializePropertyDependentVariables();

	/*!
	 * Method initializes GLUT and OpenGL windows.
	 * @param argc Number of application parameters.
	 * @param argv Array of application parameters.
	 */
	virtual void initialize(int argc, char* argv[]);

	/*!
	 * Performs single step of computations.
	 */
	virtual bool performSingleStep();

private:

	/// Property: variable denoting epsilon in action selection (the probability "below" which a random action will be selected).
	mic::configuration::Property<double> epsilon;

};

} /* namespace application */
} /* namespace neural_nets */
} /* namespace mic */

#endif /* SRC_APPLICATION_SIMPLE_TEST_HPP_ */
