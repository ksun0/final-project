import pickle
import matplotlib.pyplot as plt
import seaborn
import numpy as np

x = list(range(1, 101))
y = pickle.load(open('pelletsearned.p', 'rb'))[1]
# plt.figure(1)
plt.subplot(211)
plt.scatter(x,y)
plt.xlabel("Iteration Number")
plt.ylabel("Pellets Eearned")

# plt.figure(2)
plt.subplot(212)
plt.scatter(x,np.log(y))
plt.xlabel("Iteration Number")
plt.ylabel("Log of Pellets Eearned")
plt.show()
