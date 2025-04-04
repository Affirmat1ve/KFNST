import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import gamma, comb
import numpy as np

n=40
s=1


def pochhammer(a, k):
    if k == 0:
        return 1
    else:
        return np.prod([a + i for i in range(k)])


def p_n_s(x):
    total = 0 + 0j
    for k in range(n + 1):
        coeff = (-1) ** (n - k) * comb(n, k) * pochhammer(n + s - 1, k)
        total += coeff * x ** k
    return total

# Create data
x = np.linspace(-2, 5, 1000)
y = np.linspace(-2.5, 2.5, 1000)
X, Y = np.meshgrid(x, y)  # Create a grid of x and y values
Z = np.abs(p_n_s(X+1j*Y))  # Compute Z values based on X and Y


# Create the 3D plot
fig = plt.figure(figsize=(10, 6))  # Optional: set the figure size
ax = fig.add_subplot(111, projection='3d')  # Create a 3D axis
ax.plot_surface(X, Y, Z, cmap='viridis')  # Plot the surface

# Add labels and title
ax.set_title('3D Sine Wave Plot')  # Title of the plot
ax.set_xlabel('X axis')  # X-axis label
ax.set_ylabel('Y axis')  # Y-axis label
ax.set_zlabel('Z axis')  # Z-axis label

# Show the plot
plt.show()  # Display the plot
'''
[39.39564254527574598257458470760773357559+ 1.711988702554371729293600447764749315943j,
39.18832221927327956844416849336005708984 - 5.1378349168759365566027324205340707454471j,
38.77122694601690684811381267227462191010+8.569374558821712024311496098047057906976j]
'''