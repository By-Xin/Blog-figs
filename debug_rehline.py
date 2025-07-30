"""
Test script to understand ReHLine API and debug the implementation
"""

import numpy as np
from rehline import ReHLine

def test_basic_rehline():
    """Test basic ReHLine functionality with a simple example"""
    print("Testing basic ReHLine functionality...")
    
    # Create simple test data
    np.random.seed(42)
    n, d = 20, 2
    X = np.random.randn(n, d)
    y = np.random.randn(n)
    
    # Simple ReLU loss: ReLU(x^T β - y) 
    # This is equivalent to ReLU(1 * x^T β + (-y))
    L = 1  # One ReLU term per sample
    U = np.ones((L, n))  # Coefficient = 1 for all
    V = -y.reshape(1, -1)  # Intercept = -y_i for each sample
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"U shape: {U.shape}")
    print(f"V shape: {V.shape}")
    print(f"U: {U}")
    print(f"V: {V.flatten()[:5]}...")  # Show first 5 values
    
    # Create ReHLine solver
    C = 1.0
    solver = ReHLine(C=C)
    solver._U = U
    solver._V = V
    
    # Fit the model
    solver.fit(X)
    
    print(f"Fitted coefficients: {solver.coef_}")
    print(f"Dual variables Lambda: {getattr(solver, '_Lambda', 'Not available')}")
    
    return solver


def test_svm_example():
    """Test the SVM example from ReHLine documentation"""
    print("\nTesting SVM example from documentation...")
    
    # simulate classification dataset (from ReHLine docs)
    n, d, C = 100, 3, 0.5
    np.random.seed(1024)
    X = np.random.randn(n, d)
    beta0 = np.random.randn(d)
    y = np.sign(X.dot(beta0) + np.random.randn(n))

    # Usage of ReHLine (from docs)
    U = -(C*y).reshape(1,-1)
    L = U.shape[0]
    V = (C*np.array(np.ones(n))).reshape(1,-1)
    
    print(f"SVM U shape: {U.shape}, sample values: {U.flatten()[:5]}")
    print(f"SVM V shape: {V.shape}, sample values: {V.flatten()[:5]}")
    
    clf = ReHLine(C=C)
    clf._U, clf._V = U, V
    clf.fit(X)
    
    print('SVM solution: %s' % clf.coef_)
    print('Decision function sample: %s' % clf.decision_function([[.1,.2,.3]]))
    
    return clf


def analyze_our_problem():
    """Analyze our specific ReLU QP problem"""
    print("\nAnalyzing our ReLU QP problem...")
    
    # Generate the same test data as in main script
    n, d, L = 10, 2, 2  # Smaller for debugging
    np.random.seed(42)
    
    X = np.random.randn(n, d)
    beta_true = np.random.randn(d)
    y = X @ beta_true + 0.1 * np.random.randn(n)
    U = np.random.randn(L, n)
    V = np.random.randn(L, n)
    
    print(f"Our problem - X shape: {X.shape}")
    print(f"Our problem - y: {y}")
    print(f"Our problem - U shape: {U.shape}")
    print(f"Our problem - V shape: {V.shape}")
    print(f"Our problem - beta_true: {beta_true}")
    
    # Manual computation of loss for beta_true
    residuals_true = y - X @ beta_true
    loss_manual = 0
    for i in range(n):
        for l in range(L):
            arg = U[l, i] * residuals_true[i] + V[l, i]
            if arg > 0:
                loss_manual += arg
    print(f"Manual loss with true beta: {loss_manual}")
    
    # Try zero beta
    residuals_zero = y - X @ np.zeros(d)
    loss_zero = 0
    for i in range(n):
        for l in range(L):
            arg = U[l, i] * residuals_zero[i] + V[l, i]
            if arg > 0:
                loss_zero += arg
    print(f"Manual loss with zero beta: {loss_zero}")
    
    # Transform for ReHLine format
    U_rehline = np.zeros((L, n))
    V_rehline = np.zeros((L, n))
    
    for i in range(n):
        for l in range(L):
            U_rehline[l, i] = -U[l, i]  # Note the negative sign
            V_rehline[l, i] = U[l, i] * y[i] + V[l, i]
    
    print(f"ReHLine U shape: {U_rehline.shape}")
    print(f"ReHLine V shape: {V_rehline.shape}")
    print(f"ReHLine U sample: {U_rehline[:, :3]}")
    print(f"ReHLine V sample: {V_rehline[:, :3]}")
    
    # Test ReHLine solver
    C = 100  # High C means low regularization
    solver = ReHLine(C=C)
    solver._U = U_rehline
    solver._V = V_rehline
    
    dummy_y = np.zeros(n)
    solver.fit(X)
    
    print(f"ReHLine solution: {solver.coef_}")
    
    # Verify the solution
    beta_rehline = solver.coef_
    residuals_rehline = y - X @ beta_rehline
    loss_rehline = 0
    for i in range(n):
        for l in range(L):
            arg = U[l, i] * residuals_rehline[i] + V[l, i]
            if arg > 0:
                loss_rehline += arg
    print(f"Verified ReHLine loss: {loss_rehline}")
    
    return solver, beta_true, beta_rehline


if __name__ == "__main__":
    test_basic_rehline()
    test_svm_example()
    analyze_our_problem()