import numpy as np
import matplotlib.pyplot as plt

def predict_OLS(X_train, X_test, z_train):
    U, S, VT = np.linalg.svd(X_train,full_matrices=False)
    beta = (VT.T @ np.linalg.pinv(np.diag(S)) @ U.T) @ z_train
    z_train_predict = X_train@beta
    z_test_predict = X_test@beta
    return z_train_predict, z_test_predict

lmb = np.logspace(-4, 6, 50)
def ridge(X_train, X_test, lmb, degree):
    X_train_scaled = X_train - X_train_mean
    X_test_scaled = X_test - X_train_mean
    z_scaler = np.mean(z_train)   
    beta_list = []
    MSE_train = np.zeros((len(lmb),degree))
    MSE_test = np.zeros((len(lmb),degree))
    for j in range(len(lmb)):
        for i in range(1, degree+1):
                c = int((i+2)*(i+1)/2)
                X_tilde = X_train_scaled[:,0:c-1]
                beta = np.linalg.pinv(X_tilde.T @ X_tilde + lmb[j]*np.ones((len(X_tilde.T),len(X_tilde.T)))) @ X_tilde.T @ z_train
                beta_list.append(list(beta))
                
                ypredict = X_tilde @ beta + z_scaler
                ypredict_test = X_test_scaled[:,0:c-1] @ beta + z_scaler

                MSE_train[j, i-1] = MSE(z_train, ypredict)
                MSE_test[j, i-1] = MSE(z_test, ypredict_test)
    return MSE_train, MSE_test

def lasso(X_train, X_test, lmb, degree):
    MSE_train_Lasso = np.zeros((len(lmb),degree))
    MSE_test_Lasso = np.zeros((len(lmb),degree))
    for j in range(len(lmb)):
        for i in range(1, degree+1):
            c = int((i+2)*(i+1)/2)
            X_tilde = X_train_scaled[:,0:c-1]

            RegLasso = linear_model.Lasso(lmb[i])
            RegLasso.fit(X_tilde,z_train)

            ypredict_Lasso = RegLasso.predict(X_tilde) + z_scaler
            ypredict_test_Lasso = RegLasso.predict(X_test_scaled[:,0:c-1]) + z_scaler

            MSE_train_Lasso[j, i-1] = MSE(z_train, ypredict_Lasso)
            MSE_test_Lasso[j, i-1] = MSE(z_test, ypredict_test_Lasso)
    return MSE_train, MSE_test

def cross_validation (k, x, y, z, degree, method = 'OLS'):
    """
    k: number of folds
    z: data
    degree: polynomial degree
    n: number of data points
    method : 'OLS', 'Ridge', 'Lasso'
    """
    #create numpy arrays for MSE
    MSE_train = np.zeros(k)
    MSE_test = np.zeros(k)

    #shuffle indexes
    i = np.arange(len(z))
    np.random.shuffle(i)
    i = np.array_split(i,k) #split index into k folds, without raising an error when the number of indexes is not divisible by k
    #for each fold, use the other k-1 folds for training and test on the remaining fold
    for k in range(k):
        #create design matrix
        X = create_X(x, y, degree)

        #split data into training and test
        X_test = X[i[k]]
        z_test = z[i[k]]

        X_train = np.delete(X, i[k], axis=0)
        z_train = np.delete(z, i[k])

        #scale data
        X_train_mean = np.mean(X_train,axis=0)
        X_train = X_train - X_train_mean
        X_test = X_test - X_train_mean

        z_train_mean = np.mean(z_train)
        z_train = z_train - z_train_mean
        z_test = z_test - z_train_mean

        if method == 'OLS':
            #find beta and predict OLS
            z_train_predict, z_test_predict = predict_OLS(X_train, X_test, z_train)

            #compute MSE for training and test
            MSE_train[k] = MSE(z_train, z_train_predict)
            MSE_test[k] = MSE(z_test, z_test_predict)
        elif method == 'Ridge':
            MSE_train, MSE_test = ridge(X_train, X_test, lmb, 5)
        elif method =='Lasso':
            MSE_train, MSE_test = ridge(X_train, X_test, lmb, 5)
            
    #return mean MSE
    MSE_train_mean = np.mean(MSE_train)
    MSE_test_mean = np.mean(MSE_test)
    return MSE_train_mean, MSE_test_mean


MSE_train = np.zeros(6)
MSE_test = np.zeros(6)
folds = np.arange(5,11)
for k in folds:
    MSE_train[k-5], MSE_test[k-5] = cross_validation(k, x, y, z, 5)
plt.plot(folds, MSE_train, label='MSE_train')
plt.plot(folds, MSE_test, label='MSE_test')
plt.legend()
plt.show()
