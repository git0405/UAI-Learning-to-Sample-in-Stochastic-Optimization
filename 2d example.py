import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from matplotlib.lines import Line2D


n_points = 60
class_1 = np.random.normal(loc=[2, 2], scale=0.8, size=(n_points, 2))
class_2 = np.random.normal(loc=[5, 5], scale=0.8, size=(n_points, 2))

X = np.vstack((class_1, class_2))
y = np.hstack((np.zeros(n_points), np.ones(n_points)))


noise_ratio = 0.1

n_noisy_class_1 = int(noise_ratio * len(y))  # noise in class 1

class_0_indices = np.where(y == 0)[0]
class_1_indices = np.where(y == 1)[0]


noisy_indices_class_1 = np.random.choice(class_1_indices, size=n_noisy_class_1, replace=False)

noisy_indices = noisy_indices_class_1
y[noisy_indices] = 1 - y[noisy_indices]

class CustomSGD:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None

    def fit(self, X, y, alg_type='sgd'):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features + 1)
        X_bias = np.c_[X, np.ones(n_samples)]


        sampling_probs = np.ones(n_samples) / n_samples

        if alg_type == 'SVRG':
            for iteration in range(int(self.max_iter / n_samples)):
                snapshot_weights = self.weights.copy()
                z = np.dot(X_bias, snapshot_weights)
                p = 1 / (1 + np.exp(-z))
                full_gradient = np.dot((p - y), X_bias) / n_samples

                for j in range(n_samples):
                    i = np.random.randint(n_samples)
                    xi = X_bias[i]
                    yi = y[i]
                    pi_current = 1 / (1 + np.exp(-np.dot(xi, self.weights)))
                    pi_snapshot = 1 / (1 + np.exp(-np.dot(xi, snapshot_weights)))
                    grad_current = (pi_current - yi) * xi
                    grad_snapshot = (pi_snapshot - yi) * xi

                    self.weights -= self.learning_rate * (grad_current - grad_snapshot + full_gradient)

        elif alg_type == 'SAGA':
            gradient_memory = np.zeros((n_samples, n_features + 1))
            avg_gradient = np.zeros(n_features + 1)

            for iteration in range(int(self.max_iter / n_samples)):
                for i in range(n_samples):
                    xi = X_bias[i]
                    yi = y[i]

                    z_i = np.dot(xi, self.weights)
                    pi = 1 / (1 + np.exp(-z_i))
                    grad_i = (pi - yi) * xi

                    # Update rule using stored gradients
                    self.weights -= self.learning_rate * (grad_i - gradient_memory[i] + avg_gradient)

                    # Update gradient memory and average
                    avg_gradient += (grad_i - gradient_memory[i]) / n_samples
                    gradient_memory[i] = grad_i

        else:
            for iteration in range(int(self.max_iter / n_samples)):

                if iteration == 100:
                    self.learning_rate = self.learning_rate * 0.2

                loss_sum = 0

                for j in range(n_samples):
                    # Sample one example based on current sampling probabilities
                    sampled_index = np.random.choice(n_samples, p=sampling_probs)
                    x_sample = X_bias[sampled_index]
                    y_sample = y[sampled_index]

                    # Compute prediction and loss for the sampled point
                    linear_output = np.dot(x_sample, self.weights)
                    prediction = 1 / (1 + np.exp(-linear_output))  # Sigmoid
                    loss = -(y_sample * np.log(prediction + 1e-9) + (1 - y_sample) * np.log(1 - prediction + 1e-9))

                    # Gradient update based on sampled point
                    gradient = (prediction - y_sample) * x_sample
                    if alg_type == 'SGD-IS-grad':
                        self.weights -= self.learning_rate * gradient*1/(n_samples*sampling_probs[sampled_index])  # Adjust gradient by sampling probability
                    else:
                        self.weights -= self.learning_rate * gradient
                    loss_sum += loss

                    if alg_type == 'SGD-IS-grad':
                                x_sample = X_bias[:]
                                y_sample = y[:]

                                # Compute prediction and loss for the sampled point
                                linear_output = np.dot(x_sample, self.weights)
                                prediction = 1 / (1 + np.exp(-linear_output))  # Sigmoid

                                errors = prediction - y_sample
                                gradient = errors[:, np.newaxis] * x_sample# Exclude bias term
                                w_gradient =gradient [:,:-1]
                                gradient_l2_norm = np.linalg.norm(w_gradient, axis=1)
                                # Gradient update based on sampled point
                                all_grad = gradient_l2_norm
                                sampling_probs = all_grad / np.sum(all_grad)


                if alg_type == 'SGD-Q':
                    # Update sampling probabilities based on current loss
                    all_losses = -(
                            y * np.log(1 / (1 + np.exp(-np.dot(X_bias, self.weights)))) +
                            (1 - y) * np.log(1 - 1 / (1 + np.exp(-np.dot(X_bias, self.weights))))
                    )

                    q_loss = np.exp(-all_losses)
                    sampling_probs = q_loss / np.sum(q_loss)

        return sampling_probs

    def predict(self, X):
        X_bias = np.c_[X, np.ones(X.shape[0])]
        linear_output = np.dot(X_bias, self.weights)
        predictions = 1 / (1 + np.exp(-linear_output))
        return (predictions >= 0.5).astype(int)


alg_compare = 'SGD'


k=10000
#
custom_sgd_uni = CustomSGD(learning_rate=0.01, max_iter=k)
custom_sgd_uni.fit(X, y, alg_compare)


custom_sgd = CustomSGD(learning_rate=0.01, max_iter=k)
custom_sgd.fit(X, y, 'SGD-Q')

y_pred = custom_sgd.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


def plot_decision_boundary(clf, clf_uni,X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    contour1 = plt.contour(xx, yy, Z, colors='red', linestyles='solid')

    Z_uni = clf_uni.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_uni = Z_uni.reshape(xx.shape)

    contour2 = plt.contour(xx, yy, Z_uni, colors='k', linestyles='dashed')



    custom_legend = [
        Line2D([0], [0], linestyle='solid', color='r', label='SGD-Q boundary'),
        Line2D([0], [0], linestyle='dashed', color='k', label=alg_compare+' boundary'),
    ]

    pos_1 = np.where(y == 1)
    pos = np.where(y == 0)
    scatter1 = plt.scatter(X[pos, 0], X[pos, 1], marker='s', edgecolors='blue', s=100, facecolor='none',
                           label='Class 0')
    scatter2 = plt.scatter(X[pos_1, 0], X[pos_1, 1], marker='D', edgecolors='red', s=100, facecolor='none',
                           label='Class 1')

    plt.title('Decision Boundary', fontsize=20)
    plt.xlabel('Feature 1', fontsize=20)
    plt.ylabel('Feature 2', fontsize=20)
    plt.legend(handles=custom_legend + [scatter1, scatter2])
    plt.show()


plot_decision_boundary(custom_sgd,custom_sgd_uni, X, y)
