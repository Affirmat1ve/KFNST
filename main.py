import numpy as np
from scipy.integrate import quad
from scipy.special import gamma


# Define the function for the Riemann-Mellin integral
def f(t, s):
    return t ** (s - 1) * np.exp(-t)


# Define the Laplace inversion function
def laplace_inversion(s, a, b):
    integral, error = quad(f, a, b, args=(s,))
    return integral


def original(t, w):
    return np.exp(-(5 - 6j) * t) * np.cos(w * t) * t


def l_image(p, w):
    return (p ** 2 - w ** 2) / ((p ** 2 + w ** 2) ** 2)


def get_gamma_values():
    x_values = [0, 1, 5, 1.5]
    gamma_values = gamma(x_values)

    for x, g in zip(x_values, gamma_values):
        print(f"Gamma({x:.2f}) = {g:.4f}")
    return gamma_values


s = 3
a = 0  # Lower limit of integration
b = np.inf  # Upper limit of integration

# Compute the Laplace inversion
result = laplace_inversion(s, a, b)
print(f"Laplacian inversion result: {result}")

z = 5 - 6j
omega = 1
q, err = quad(original, 0, np.inf, args=(omega,))
ans = l_image(z, omega)
print(ans, q, err)
print(ans - q)
