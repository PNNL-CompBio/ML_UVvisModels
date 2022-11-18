import os
import numpy as np
import matplotlib.pyplot as plt

data = open('val_full.csv', 'r')

x = np.arange(220, 401)

print(x, len(x))

for lines in data.readlines():
	line = lines.split(',')
	line1 = [float(i) for i in line[1:]]
	plt.plot(x, line1/np.max(line1))

plt.axvline(x=0, color="black", linestyle='-')
plt.axhline(y=0, color="black", linestyle='-')

plt.ylim([-0.1, 1.2])
plt.xlim([210, 410])

plt.xlabel('Wavelength (nm)', size=28)
plt.ylabel('Absorbance',  size=28)
plt.legend(loc='best', fontsize =25)

plt.legend()
plt.savefig('all_spectra.png')
plt.show()
	
