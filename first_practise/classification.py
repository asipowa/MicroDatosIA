import numpy as np
import sklearn 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

#################################################################################
# 1. Read the data and plot the data
#################################################################################

data = pd.read_csv('Data/student_exam_data.csv')
print('############ Read data ############\n')
print(data.head())

#################################################################################
# 2 Plot the data
#################################################################################

color = {'Pass/Fail': {0: 'red', 1: 'green'}}
'''
sns.scatterplot(x='Study Hours', y='Previous Exam Score', data=data, hue='Pass/Fail', palette=color['Pass/Fail'])
plt.title('Student Exam Data')
plt.xlabel('Hours Studied')
plt.ylabel('Passed (1 = Yes, 0 = No)')
plt.show() 
'''
#################################################################################
# 3. Prepare the data for training
#################################################################################

# Split randomly the data into training and test sets
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=35)

print('\n############ Train/Test split ############\n')
print(f'Training data size: {train_data.shape[0]} samples')
print(f'Test data size: {test_data.shape[0]} samples')

# Training data
X_train = train_data[['Study Hours', 'Previous Exam Score']]
y_train = train_data['Pass/Fail']

# Test data
X_test = test_data[['Study Hours', 'Previous Exam Score']]
y_test = test_data['Pass/Fail']

#################################################################################
# 4. Train a classification model
#################################################################################

def feature_mapping(x1, x2, degree=6):
    """
    Maps the two input features to polynomial features up to the specified degree.
    
    Parameters:
    x1 -- First feature (numpy array or pandas Series)
    x2 -- Second feature (numpy array or pandas Series)
    degree -- The maximum degree of polynomial features to generate
    
    Returns:
    A numpy array with the mapped features.
    """
    if len(x1.shape) == 1:
        x1 = x1[:, np.newaxis]
    if len(x2.shape) == 1:
        x2 = x2[:, np.newaxis]
        
    out = np.ones((x1.shape[0], 1))  # Start with the bias term (intercept)
    
    for i in range(1, degree + 1):
        for j in range(i + 1):
            term = (x1 ** (i - j)) * (x2 ** j)
            out = np.hstack((out, term))
    
    return out

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(theta, X, y, lambda_):
    m = len(y)
    h = sigmoid(X.dot(theta))
    
    # Compute the cost
    term1 = -y * np.log(h)
    term2 = (1 - y) * np.log(1 - h)
    cost = (1/m) * np.sum(term1 - term2)
    
    # Regularization (excluding the bias term)
    reg_term = (lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))
    total_cost = cost + reg_term
    
    return total_cost

def gradient(theta, X, y, lambda_):
    m = len(y)
    h = sigmoid(X.dot(theta))
    error = h - y
    grad = (1/m) * (X.T.dot(error))
    
    # Regularization (excluding the bias term)
    reg_term = (lambda_ / m) * theta
    reg_term[0] = 0  # No regularization for the bias term
    grad += reg_term
    
    return grad

degree = 4

normal_X_train = (X_train - X_train.mean()) / X_train.std()
normal_X_test = (X_test - X_test.mean()) / X_test.std()

X_with_intercept = feature_mapping(normal_X_train['Study Hours'].values, normal_X_train['Previous Exam Score'].values, degree=degree)
print("Mapped feature shape:", X_with_intercept.shape)
y = train_data['Pass/Fail'].values
initial_theta = np.zeros(X_with_intercept.shape[1])
lambda_ = 0.1

minimize_compute = minimize(fun=cost_function, 
                  x0=initial_theta, 
                  args=(X_with_intercept, y, lambda_),
                  jac=gradient)




plt.figure(figsize=(8,6))
# Create a grid of values
u = np.linspace(normal_X_test['Study Hours'].min()-0.5, normal_X_test['Study Hours'].max()+0.5, 50)
v = np.linspace(normal_X_test['Previous Exam Score'].min()-0.5, normal_X_test['Previous Exam Score'].max()+0.5, 50)
z = np.zeros((len(u), len(v)))
for i in range(len(u)):
    for j in range(len(v)):
        mapped_feature = feature_mapping(np.array([u[i]]), np.array([v[j]]), degree=degree)
        z[i,j] = mapped_feature.dot(minimize_compute.x)
z = z.T

acepted = data[data['Pass/Fail'] == 1]
not_acepted = data[data['Pass/Fail'] == 0]

# Plot the decision boundary
plt.contour(u, v, z, levels=[0], colors='green')
plt.xlabel('Study Hours')
plt.ylabel('Previous Exam Score')
plt.scatter(normal_X_test['Study Hours'], normal_X_test['Previous Exam Score'], c=y_test, cmap='bwr', edgecolors='k', label='Test Data')
plt.scatter(normal_X_train['Study Hours'], normal_X_train['Previous Exam Score'], c=y_train, cmap='bwr', marker='x', label='Train Data')
plt.legend()
plt.show()

############################################################################
# 5. Prediction of the test set
############################################################################

print('\n############ Model training ############\n')
print('Model coefficients:')
for i, coef in enumerate(minimize_compute.x):
    print(f'Theta {i}: {coef}')

