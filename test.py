import numpy as np
import neural_network as nn
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
np.random.seed(0)
X, y = make_moons(200, noise=.2)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

plt.figure(figsize=(16,32))
hidden_layer_dimensions = [1,2,3,4]

for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5,2,i+1)
    plt.title('HiddenLayerSize%d' % nn_hdim)
    model = nn.build_model(X,y,nn_hdim)
    nn.plot_decision_boundary(lambda x: predict(model, x), X, y)
plt.show()
