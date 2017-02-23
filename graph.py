import pickle
import matplotlib.pyplot as plt
import seaborn
import numpy as np
x = list(range(1, 101))
y = pickle.load(open('pelletsearned.p', 'rb'))[1]

plt.scatter(x,np.log(y))
plt.xlabel("Iteration Number")
plt.ylabel("Log of Pellets Eearned")
plt.show()
