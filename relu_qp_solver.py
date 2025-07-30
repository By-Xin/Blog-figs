"""
ReLU Quadratic Programming Solver Implementation
Original simple iterative solver and ReHLine-based improved solver
"""

import numpy as np
from scipy.optimize import minimize
import warnings

try:
    from rehline import ReHLine
    REHLINE_AVAILABLE = True
except ImportError:
    REHLINE_AVAILABLE = False
    warnings.warn("ReHLine not available, only original solver will work")


def solve_relu_qp(X, y, U, V, L, n, d, reg_lambda=0.01, max_iter=100, tol=1e-6):
    """
    Original simple iterative solver for ReLU regression QP problem
    
    Problem: min_β Σ_i Σ_l ReLU(u_{l,i} * (y_i - x_i^T β) + v_{l,i}) + λ/2 ||β||²
    
    Args:
        X: design matrix (n x d)
        y: response vector (n,)
        U: ReLU coefficients matrix (L x n)
        V: ReLU intercepts matrix (L x n)
        L: number of ReLU terms
        n: number of samples
        d: number of features
        reg_lambda: regularization parameter
        max_iter: maximum iterations
        tol: convergence tolerance
    
    Returns:
        beta_star: optimal primal variables
        residuals: residuals y - X @ beta_star
        loss: final loss value
    """
    
    # Initialize beta with least squares solution
    beta = np.linalg.pinv(X) @ y
    
    for iter_num in range(max_iter):
        beta_old = beta.copy()
        
        # Compute residuals
        residuals = y - X @ beta
        
        # Compute subgradients for each ReLU term
        total_subgrad = np.zeros(d)
        total_loss = 0
        
        for l in range(L):
            for i in range(n):
                arg = U[l, i] * residuals[i] + V[l, i]
                if arg > 0:  # ReLU is active
                    total_subgrad += U[l, i] * (-X[i, :])
                    total_loss += arg
        
        # Add regularization gradient
        grad = total_subgrad + reg_lambda * beta
        
        # Simple gradient descent update
        learning_rate = 0.01 / (1 + 0.1 * iter_num)
        beta = beta - learning_rate * grad
        
        # Check convergence
        if np.linalg.norm(beta - beta_old) < tol:
            break
    
    final_residuals = y - X @ beta
    final_loss = total_loss + 0.5 * reg_lambda * np.linalg.norm(beta)**2
    
    return beta, final_residuals, final_loss


def solve_relu_qp_with_rehline(X, y, U, V, L, n, d, reg_lambda=0.01):
    """
    使用 ReHLine 库求解 ReLU 回归的 QP 问题
    
    ReHLine 问题形式:
    min_β Σ_i Σ_l ReLU(u_{li} * x_i^T β + v_{li}) + 1/(2C) ||β||²
    
    其中我们需要将我们的损失转换为 ReHLine 格式。
    我们的原始问题: min_β Σ_i Σ_l ReLU(u_{l,i} * (y_i - x_i^T β) + v_{l,i}) + λ/2 ||β||²
    需要转换为: min_β Σ_i Σ_l ReLU(u_{l,i} * (-x_i^T β) + (u_{l,i} * y_i + v_{l,i})) + λ/2 ||β||²
    
    Args:
        X: design matrix (n x d)
        y: response vector (n,)
        U: ReLU coefficients matrix (L x n)
        V: ReLU intercepts matrix (L x n)  
        L: number of ReLU terms
        n: number of samples
        d: number of features
        reg_lambda: regularization parameter
    
    Returns:
        beta_star: optimal primal variables
        pi_star: dual variables for ReLU constraints
        lambda_star: dual variables for regularization
        residuals: residuals y - X @ beta_star
        loss: final loss value
    """
    
    if not REHLINE_AVAILABLE:
        raise ImportError("ReHLine library is not available. Please install with: pip install rehline")
    
    # 构建 ReHLine 的损失函数格式
    # ReHLine problem: min_β Σ_i Σ_l ReLU(u_{li} * x_i^T β + v_{li}) + 1/(2C) ||β||²
    # Our problem: min_β Σ_i Σ_l ReLU(u_{l,i} * (y_i - x_i^T β) + v_{l,i}) + λ/2 ||β||²
    #            = min_β Σ_i Σ_l ReLU(-u_{l,i} * x_i^T β + (u_{l,i} * y_i + v_{l,i})) + λ/2 ||β||²
    
    # 构建 ReHLine 格式的参数
    # U_rehline[l,i] = coefficient for x_i^T β in l-th ReLU term of i-th sample
    # V_rehline[l,i] = intercept for l-th ReLU term of i-th sample
    
    U_rehline = np.zeros((L, n))  # (L, n)
    V_rehline = np.zeros((L, n))  # (L, n)
    
    for i in range(n):
        for l in range(L):
            U_rehline[l, i] = -U[l, i]  # Note the negative sign for the transformation
            V_rehline[l, i] = U[l, i] * y[i] + V[l, i]  # Transform intercept
    
    # ReHLine uses C = 1/λ for regularization
    C = 1.0 / reg_lambda if reg_lambda > 0 else 1e6
    
    # 创建 ReHLine 求解器实例 - increase max_iter to improve convergence
    solver = ReHLine(C=C, max_iter=5000, tol=1e-6)
    
    # 设置 ReLU 损失参数
    solver._U = U_rehline  # (L, n)
    solver._V = V_rehline  # (L, n)
    
    # 拟合模型 - ReHLine 只需要 X
    solver.fit(X)
    
    # 获取结果
    beta_star = solver.coef_
    
    # 计算残差
    residuals = y - X @ beta_star
    
    # 获取对偶变量
    pi_star = getattr(solver, '_Lambda', None)  # Dual variables for ReLU constraints
    lambda_star = getattr(solver, '_xi', None)   # Dual variables for linear constraints (if any)
    
    # 计算最终损失 (using original loss formulation)
    final_loss = 0
    for i in range(n):
        for l in range(L):
            arg = U[l, i] * residuals[i] + V[l, i]
            if arg > 0:
                final_loss += arg
    final_loss += 0.5 * reg_lambda * np.linalg.norm(beta_star)**2
    
    return beta_star, pi_star, lambda_star, residuals, final_loss


def generate_test_data(n=50, d=3, L=2, noise_std=0.1, random_seed=42):
    """
    Generate synthetic test data for ReLU QP problem
    
    Args:
        n: number of samples
        d: number of features  
        L: number of ReLU terms per sample
        noise_std: standard deviation of noise
        random_seed: random seed for reproducibility
    
    Returns:
        X, y, U, V: test data matrices
    """
    np.random.seed(random_seed)
    
    # Generate design matrix
    X = np.random.randn(n, d)
    
    # Generate true coefficients
    beta_true = np.random.randn(d)
    
    # Generate response with noise
    y = X @ beta_true + noise_std * np.random.randn(n)
    
    # Generate ReLU parameters
    U = np.random.randn(L, n)  # Coefficients for each ReLU term
    V = np.random.randn(L, n)  # Intercepts for each ReLU term
    
    return X, y, U, V, beta_true


def compare_solvers(X, y, U, V, L, n, d, reg_lambda=0.01):
    """
    Compare the original solver with ReHLine solver
    
    Returns:
        results: dictionary with comparison results
    """
    print("Comparing Original Solver vs ReHLine Solver")
    print("=" * 50)
    
    # Solve with original method
    print("Solving with original iterative method...")
    beta_orig, residuals_orig, loss_orig = solve_relu_qp(
        X, y, U, V, L, n, d, reg_lambda
    )
    
    results = {
        'original': {
            'beta': beta_orig,
            'residuals': residuals_orig,
            'loss': loss_orig,
            'method': 'Simple iterative'
        }
    }
    
    # Solve with ReHLine if available
    if REHLINE_AVAILABLE:
        print("Solving with ReHLine library...")
        try:
            beta_rehline, pi_star, lambda_star, residuals_rehline, loss_rehline = solve_relu_qp_with_rehline(
                X, y, U, V, L, n, d, reg_lambda
            )
            
            results['rehline'] = {
                'beta': beta_rehline,
                'pi_star': pi_star,
                'lambda_star': lambda_star,
                'residuals': residuals_rehline,
                'loss': loss_rehline,
                'method': 'ReHLine'
            }
            
            # Compare results
            beta_diff = np.linalg.norm(beta_orig - beta_rehline)
            loss_diff = abs(loss_orig - loss_rehline)
            
            print(f"\nComparison Results:")
            print(f"Beta difference (L2 norm): {beta_diff:.6f}")
            print(f"Loss difference: {loss_diff:.6f}")
            print(f"Original loss: {loss_orig:.6f}")
            print(f"ReHLine loss: {loss_rehline:.6f}")
            
            results['comparison'] = {
                'beta_diff': beta_diff,
                'loss_diff': loss_diff
            }
            
        except Exception as e:
            print(f"Error with ReHLine solver: {e}")
            results['rehline'] = {'error': str(e)}
    else:
        print("ReHLine not available, skipping ReHLine solver")
        results['rehline'] = {'error': 'ReHLine not installed'}
    
    return results


if __name__ == "__main__":
    # Generate test data
    n, d, L = 50, 3, 2
    X, y, U, V, beta_true = generate_test_data(n, d, L)
    
    print(f"Generated test data: n={n}, d={d}, L={L}")
    print(f"True beta: {beta_true}")
    print()
    
    # Compare solvers
    results = compare_solvers(X, y, U, V, L, n, d, reg_lambda=0.01)
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    if 'original' in results:
        print(f"Original solver beta: {results['original']['beta']}")
        print(f"Original solver loss: {results['original']['loss']:.6f}")
    
    if 'rehline' in results and 'error' not in results['rehline']:
        print(f"ReHLine solver beta: {results['rehline']['beta']}")
        print(f"ReHLine solver loss: {results['rehline']['loss']:.6f}")
        
        if 'comparison' in results:
            print(f"\nSolver comparison:")
            print(f"  Parameter difference: {results['comparison']['beta_diff']:.6f}")
            print(f"  Loss difference: {results['comparison']['loss_diff']:.6f}")
            
            if results['comparison']['beta_diff'] < 1e-3:
                print("  ✓ Solvers agree well!")
            else:
                print("  ⚠ Solvers show significant differences")
    
    print(f"\nTrue beta (reference): {beta_true}")