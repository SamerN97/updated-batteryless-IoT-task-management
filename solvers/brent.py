import math


def brent(f, a, b, args, tol=1e-6, max_iter=100):
    """
    Finds a root of `f` in the interval [a, b] using [Brent's method](https://en.wikipedia.org/wiki/Brent%27s_method).

    Parameters:
        f (function): The function to find a root of.
        a (float): Left endpoint of the interval.
        b (float): Right endpoint of the interval.
        tol (float): Tolerance for convergence (default: 1e-6).
        max_iter (int): Maximum number of iterations (default: 100).

    Returns:
        float: The estimated root.
    """
    # Check if the interval brackets a root
    f_a = f(a, *args)
    f_b = f(b, *args)
    if f_a * f_b >= 0:
        raise ValueError("Root not bracketed. f(a) and f(b) must have opposite signs.")

    # Ensure |f(b)| < |f(a)|
    if abs(f_a) < abs(f_b):
        a, b = b, a

    c = a  # Previous iterate
    d = a  # Second previous iterate (for IQI)
    flag = True  # Force bisection if IQI/Secant fails

    for _ in range(max_iter):
        # Check convergence
        f_b = f(b, *args)
        if abs(f_b) < tol or abs(b - a) < tol:
            return b

        # Compute new estimate using IQI or Secant
        s = 0.0
        f_a = f(a, *args)
        f_c = f(c, *args)
        if f_a != f_c and f_b != f_c:
            # Inverse Quadratic Interpolation (IQI)
            s = (a * f_b * f_c) / ((f_a - f_b) * (f_a - f_c)) + \
                (b * f_a * f_c) / ((f_b - f_a) * (f_b - f_c)) + \
                (c * f_a * f_b) / ((f_c - f_a) * (f_c - f_b))
        else:
            # Secant Method (fallback)
            s = b - f_b * (b - a) / (f_b - f_a)

        # Check if the estimate is "good"
        if (s < min((3*a + b)/4, b) or s > max((3*a + b)/4, b)) or \
           (flag and abs(s - b) >= abs(b - c)/2) or \
           (not flag and abs(s - b) >= abs(c - d)/2):
            # Revert to bisection
            s = (a + b) / 2
            flag = True
        else:
            flag = False

        # Update variables
        d = c
        c = b
        f_s = f(s, *args)
        if f_a * f_s < 0:
            b = s
        else:
            a = s

        # Ensure |f(b)| < |f(a)|
        if abs(f_a) < abs(f_b):
            a, b = b, a

    raise ValueError(f"Brent's method did not converge in {max_iter} iterations.")