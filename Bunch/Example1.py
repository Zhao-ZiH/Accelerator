from GaussianBunchField import Gaussian_Bunch
import matplotlib.pyplot as plt
import numpy as np

E_k=0.4E9 #[MeV]
e_number=2e4
sigma_x=1e-4
sigma_y=1e-4
sigma_z=1e-7

bunch=Gaussian_Bunch(Energy=E_k,sigma_x=sigma_x,sigma_y=sigma_y,sigma_z=sigma_z,Number_e=e_number)
x_val=0
z_val=0
E_y=[]
Y_range=[]
for y_val in np.linspace(0,10*sigma_y,300):
    bunch.set_Ey_local(x=x_val,y=y_val,z=z_val)
    E_y.append(bunch.E_y)
    Y_range.append(y_val)
fig,ax=plt.subplots(1)
ax.plot([i/sigma_y for i in Y_range],E_y)
ax.set_xlabel(r'y/$\sigma_{y}$')
ax.set_ylabel(r'$E_{y}$')
plt.show()