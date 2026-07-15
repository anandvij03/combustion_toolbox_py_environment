import numpy as np
from combustiontoolbox.core.EquationStateJCZ3.EquationStateJCZ3 import EquationStateJCZ3

# Define species and properties
species_names = ['N2', 'O2']
index_condensed = []
moles = np.array([0.5, 0.5])
V = 1.0
T = 300.0

# Initialize EOS and compute Jacobian
eos = EquationStateJCZ3(species_names, index_condensed)
J = eos.getDepartureJacobian(moles, V, T)

print("Jacobian:")
print(J)
print("\nSymmetric?", np.allclose(J, J.T))
