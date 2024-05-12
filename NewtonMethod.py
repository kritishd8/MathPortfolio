
#? ----------------------
#? Name: Kritish Dhakal
#? Uni ID: 2408573
#? ----------------------

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def plot_function(func, a, b):
    """
    This function plots the graph of the input function
    within the given interval [a, b).
    """
    x = np.linspace(a, b, 100)
    y = func(x)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Graph of the Function')
    plt.grid(True)
    plt.show()


def newton_method(func, grad, x0, tol=1e-6, max_iter=100):
    '''Approximate solution of f(x)=0 by Newton-Raphson's method.

        Parameters
        ----------
        func : function 
            Function value for which we are searching for a solution f(x)=0,
        grad: function
            Gradient value of function f(x)
        x0 : number
            Initial guess for a solution f(x)=0.
        tol : number
            Stopping criteria is abs(f(x)) < tol.
        max_iter : integer
            Maximum number of iterations of Newton's method.

        Returns
        -------
        xn : root
    '''
    # Main Loop starts here
    iter_count = 1
    while iter_count <= max_iter:
        # Calculate next approximation
        xn = x0 - func(x0) / grad(x0)
        
        # Check convergence
        if abs(func(xn)) < tol:
            return xn
        
        # Update x0 for next iteration
        x0 = xn
        
        iter_count += 1

    print("Warning! Exceeded the maximum number of iterations.")
    return xn


# Main Driver Function:
if __name__ == "__main__":
    # Define the 1st Function for which the root is to be found
    # func = lambda x: x**2 - x - 1
    # Define the gradient of the Function
    # grad = lambda x: 2*x - 1

    # Uncomment the next two lines to use the 2nd Function
    func = lambda x: x**3 - x**2 - 2*x + 1
    grad = lambda x: 3*x**2 - 2*x -2

    # Call plot_function to plot graph of the function
    plot_function(func, -2, 2)

    x0_1 = 1  # Initial guess for 1st root (change the value as required)
    # Call the Newton's method for 1st root
    our_root_1 = newton_method(func, grad, x0_1)

    # Call SciPy method (reference method) for 1st root
    sp_result_1 = sp.optimize.root(func, x0_1)
    sp_root_1 = sp_result_1.x.item()

    x0_2 = -1  # Initial guess for 2nd root (change the value as required)
    # Call the Newton's method for 2nd root
    our_root_2 = newton_method(func, grad, x0_2)

    # Call SciPy method (reference method) for 2nd root
    sp_result_2 = sp.optimize.root(func, x0_2)
    sp_root_2 = sp_result_2.x.item()
    
    x0_3 = -2  # Initial guess for 3rd root (change the value as required)
    # Call the Newton's method for 3rd root
    our_root_3 = newton_method(func, grad, x0_3)

    # Call SciPy method (reference method) for 2nd root
    sp_result_3 = sp.optimize.root(func, x0_3)
    sp_root_3 = sp_result_3.x.item()

    # Print the result
    print("1st root found by Newton's Method = {:0.8f}.".format(our_root_1))
    print("1st root found by SciPy = {:0.8f}".format(sp_root_1))

    print("2nd root found by Newton's Method = {:0.8f}.".format(our_root_2))
    print("2nd root found by SciPy = {:0.8f}".format(sp_root_2))
    
    print("3rd root found by Newton's Method = {:0.8f}.".format(our_root_3))
    print("3rd root found by SciPy = {:0.8f}".format(sp_root_3))
