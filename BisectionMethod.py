
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
    within the given interval [a,b).
    """
    x = np.linspace(a, b, 100)
    y = func(x)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Graph of the Function')
    plt.grid(True)
    plt.show()


def bisection_method(func, a, b, tol=1e-6, max_iter=100):
    """
    Bisection method to find the root of a function within a given interval.

    Parameters:
    - func: The function for which the root is to be found.
    - a, b: Interval [a, b] within which the root is searched for.
    - tol: Tolerance level for checking convergence of the method.
    - max_iter: Maximum number of iterations.

    Returns:
    - root: Approximation of the root.
    """

    # Check if the interval is valid (signs of f(a) and f(b) are different)
    if np.sign(func(a)) == np.sign(func(b)):
        raise ValueError("Signs of f(a) and f(b) must be different.")

    # Main loop starts here
    iter_count = 1
    while iter_count <= max_iter:
        # Calculate midpoint
        c = (a + b) / 2
        
        # Check if the root is found or within tolerance
        if np.abs(func(c)) < tol:
            return c
        
        # Update interval [a, b]
        if np.sign(func(c)) == np.sign(func(a)):
            a = c
        else:
            b = c
        
        iter_count += 1
    
    print("Warning! Exceeded the maximum number of iterations.")
    return (a + b) / 2

# Example usage:
if __name__ == "__main__":
    # Define the function for which the root is to be found
    # func = lambda x: x**2 - x - 1  # First Function (f1)
    
    # Uncomment the below line to use the Second Function
    func = lambda x: x**3 - x**2 - 2*x + 1  # Second Function (f2)

    # Call plot_function to plot graph of the function
    plot_function(func, -2, 0)

    # Set the interval [a, b] for the search
    a_1 = -1; b_1 = 1;  # For first root (change the values as required)
    a_2 = -2; b_2 = 2;  # For second root (change the values as required)
    a_3 = -3; b_3 = 3;  # For third root (change the values as required)

    # Call the bisection method
    our_root_1 = bisection_method(func, a_1, b_1)
    our_root_2 = bisection_method(func, a_2, b_2)
    our_root_3 = bisection_method(func, a_3, b_3)

    # Call SciPy method root, which we consider as a reference method.
    sp_result_1 = sp.optimize.root(func, (a_1 + b_1) / 2)
    sp_root_1 = sp_result_1.x.item()

    sp_result_2 = sp.optimize.root(func, (a_2 + b_2) / 2)
    sp_root_2 = sp_result_2.x.item()
    
    sp_result_3 = sp.optimize.root(func, (a_3 + b_3) / 2)
    sp_root_3 = sp_result_3.x.item()

    # Print the result
    print("1st root found by Bisection Method = {:0.8f}.".format(our_root_1))
    print("1st root found by SciPy = {:0.8f}".format(sp_root_1))

    print("2nd root found by Bisection Method = {:0.8f}.".format(our_root_2))
    print("2nd root found by SciPy = {:0.8f}".format(sp_root_2))
    
    print("3rd root found by Bisection Method = {:0.8f}.".format(our_root_3))
    print("3rd root found by SciPy = {:0.8f}".format(sp_root_3))

