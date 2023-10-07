from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

def cross_validation(X, z, k=(5, 11), method='OLS'):
    """
    Arguments:
    X: design matrix
    z: data
    k: number of folds  [start, stop[
    method : 'OLS', 'Ridge', 'Lasso'

    Returns:
    """

    for i in range(k[0], k[1]):  # loop over number of folds
        if method == 'OLS':
            degree = 8
            MSE_train = np.zeros(degree)
            MSE_test = np.zeros(degree)
        else:
            lambdas = np.logspace(-4, 6, 50)
            MSE_train = np.zeros(len(lambdas))
            MSE_test = np.zeros(len(lambdas))

        # shuffle indexes
        indices = np.arange(len(z))
        np.random.shuffle(indices)
        indices = np.array_split(indices, i)  # split index into k folds, without raising an error when the number of indexes is not divisible by k
        # for each fold, use the other k-1 folds for training and test on the remaining fold

        for j in range(len(indices)):  # loop over each fold
            # split data into training and test
            X_test = X[indices[j]]
            z_test = z[indices[j]]

            X_train = np.delete(X, indices[j], axis=0)
            z_train = np.delete(z, indices[j])

            # scale data (we only center the data here)
            X_train_mean = np.mean(X_train, axis=0)
            X_train = X_train - X_train_mean
            X_test = X_test - X_train_mean

            z_train_mean = np.mean(z_train)
            z_train = z_train - z_train_mean
            z_test = z_test - z_train_mean

            if method == 'OLS':
                MSE_train_indiv = np.zeros(degree)
                MSE_test_indiv = np.zeros(degree)

                for d in range(1, degree + 1):
                    c = int((d + 2) * (d + 1) / 2)
                    X_tilde = X_train[:, 0:c - 1]
                    beta = np.linalg.pinv(X_tilde.T @ X_tilde) @ X_tilde.T @ z_train

                    ypredict = X_tilde @ beta + z_train_mean
                    ypredict_test = X_test[:, 0:c - 1] @ beta + z_train_mean

                    MSE_train_indiv[d - 1] = MSE(z_train, ypredict)
                    MSE_test_indiv[d - 1] = MSE(z_test, ypredict_test)
                MSE_train += MSE_train_indiv
                MSE_test += MSE_test_indiv

            elif method == 'Ridge':
                degree = 5  # choose the degree of the polynomial
                MSE_train_lmb = np.zeros(len(lambdas))
                MSE_test_lmb = np.zeros(len(lambdas))

                for s, lmb in enumerate(lambdas):
                    ridge = Ridge(alpha=lmb)
                    ridge.fit(X_train, z_train)
                    z_train_predict = ridge.predict(X_train)
                    z_test_predict = ridge.predict(X_test)
                    MSE_train_lmb[s] = MSE(z_train, z_train_predict)
                    MSE_test_lmb[s] = MSE(z_test, z_test_predict)   
                MSE_train += MSE_train_lmb
                MSE_test += MSE_test_lmb

            elif method == 'Lasso':
                degree = 5
                MSE_train_lmb = np.zeros(len(lambdas))
                MSE_test_lmb = np.zeros(len(lambdas))

                for s, lmb in enumerate(lambdas):
                    lasso = Lasso(alpha=lmb)
                    lasso.fit(X_train, z_train)
                    z_train_predict = lasso.predict(X_train)
                    z_test_predict = lasso.predict(X_test)
                    MSE_train_lmb[s] = MSE(z_train, z_train_predict)
                    MSE_test_lmb[s] = MSE(z_test, z_test_predict)
                MSE_train += MSE_train_lmb
                MSE_test += MSE_test_lmb
        # return mean MSE
        MSE_train_mean = MSE_train/i
        MSE_test_mean = MSE_test/i

        # plot MSE for each number of folds
        if method == 'OLS':
            degree_range = np.arange(1, degree + 1)
            plt.plot(degree_range, MSE_train_mean, label='MSE_train, number of folds = %i' % i)
            plt.plot(degree_range, MSE_test_mean, label='MSE_test, number of folds = %i' % i)

        else:
            plt.plot(lambdas, MSE_train_mean, label='MSE_train, number of folds = %i' % i)
            plt.plot(lambdas, MSE_test_mean, label='MSE_test, number of folds = %i' % i)

    plt.legend()
    plt.show()

cross_validation(X, z)
cross_validation (X, z, method='Ridge')
#cross_validation (X, z, method='Lasso')
