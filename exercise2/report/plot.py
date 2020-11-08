import matplotlib.pyplot as plt

x = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]
y1 = [280, 260, 246, 233, 213, 202, 132, 87]

plt.plot(x, y1)
plt.xlabel(r'$\sigma_1$')
plt.ylabel(r'Number of detected corners')
plt.show()