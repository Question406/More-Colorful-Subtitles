import colour
import numpy as np
a = np.array([18, -13, 20])
b = np.array([18, -13, 122])
delta_E = colour.delta_E(a, b, method="CIE 2000")
print(delta_E)