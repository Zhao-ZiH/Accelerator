from GaussianBunchField import Gaussian_Bunch
import matplotlib.pyplot as plt
import numpy as np

E_k = 0.4E9  # [MeV]
e_number = 2e4
sigma_x = 1e-4
sigma_y = 1e-4
sigma_z = 1e-7

x_val = 0
y_val = 0
E_z = []
E_z_p = []
Z_range = []
bunch = Gaussian_Bunch(Energy=E_k, sigma_x=sigma_x, sigma_y=sigma_y, sigma_z=sigma_z, Number_e=e_number)
for z_val in np.linspace(-5 * sigma_z, 5 * sigma_z, 100):
    bunch.set_Ez_local(x=x_val, y=y_val, z=z_val)
    bunch.set_Ez_derivative_z_local(x=x_val, y=y_val, z=z_val)
    E_z.append(bunch.E_z)
    E_z_p.append(bunch.E_z_derivative_z)
    Z_range.append(z_val)
fig, ax = plt.subplots(1)
ax.plot([i / sigma_z for i in Z_range], E_z)
ax_twin = ax.twinx()
ax_twin.tick_params(axis='y', colors='red')
ax_twin.plot([i / sigma_z for i in Z_range], E_z_p, c='red')
ax.set_xlabel(r'z/$\sigma_{z}$')
ax.set_ylabel('$E_{z}$')
ax_twin.set_ylabel('$E^{\'}_{z}$',color='red')
plt.show()
