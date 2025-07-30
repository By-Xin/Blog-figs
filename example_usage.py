"""
Example usage of ReLU QP solvers with visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from relu_qp_solver import (
    solve_relu_qp, 
    solve_relu_qp_with_rehline,
    generate_test_data,
    REHLINE_AVAILABLE
)


def visualize_loss_landscape_1d():
    """Visualize the ReLU loss landscape for a 1D problem"""
    print("Creating 1D loss landscape visualization...")
    
    # Simple 1D problem for visualization
    np.random.seed(42)
    n, d, L = 20, 1, 2
    X = np.random.randn(n, d)
    y = np.random.randn(n)
    U = np.random.randn(L, n)
    V = np.random.randn(L, n)
    
    # Grid of beta values
    beta_range = np.linspace(-3, 3, 100)
    losses = []
    reg_lambda = 0.01
    
    # Compute loss for each beta value
    for beta_val in beta_range:
        beta = np.array([beta_val])
        residuals = y - X.flatten() * beta_val
        
        loss = 0
        for i in range(n):
            for l in range(L):
                arg = U[l, i] * residuals[i] + V[l, i]
                if arg > 0:
                    loss += arg
        loss += 0.5 * reg_lambda * beta_val**2
        losses.append(loss)
    
    # Solve with both methods
    beta_orig, _, loss_orig = solve_relu_qp(X, y, U, V, L, n, d, reg_lambda)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(beta_range, losses, 'b-', linewidth=2, label='ReLU Loss')
    plt.axvline(beta_orig[0], color='red', linestyle='--', label=f'Original solver: β={beta_orig[0]:.3f}')
    
    if REHLINE_AVAILABLE:
        try:
            beta_rehline, _, _, _, loss_rehline = solve_relu_qp_with_rehline(X, y, U, V, L, n, d, reg_lambda)
            plt.axvline(beta_rehline[0], color='green', linestyle=':', label=f'ReHLine: β={beta_rehline[0]:.3f}')
        except Exception as e:
            print(f"ReHLine visualization failed: {e}")
    
    plt.xlabel('β')
    plt.ylabel('Loss')
    plt.title('ReLU Loss Landscape (1D)')
    plt.legend()
    plt.grid(True)
    
    # Zoom in around minimum
    plt.subplot(1, 2, 2)
    min_idx = np.argmin(losses)
    zoom_range = max(10, min_idx - 20), min(len(beta_range) - 10, min_idx + 20)
    zoom_beta = beta_range[zoom_range[0]:zoom_range[1]]
    zoom_losses = losses[zoom_range[0]:zoom_range[1]]
    
    plt.plot(zoom_beta, zoom_losses, 'b-', linewidth=2, label='ReLU Loss')
    plt.axvline(beta_orig[0], color='red', linestyle='--', label=f'Original: β={beta_orig[0]:.3f}')
    
    if REHLINE_AVAILABLE:
        try:
            plt.axvline(beta_rehline[0], color='green', linestyle=':', label=f'ReHLine: β={beta_rehline[0]:.3f}')
        except:
            pass
    
    plt.xlabel('β')
    plt.ylabel('Loss')
    plt.title('Zoomed Loss Landscape')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/runner/work/Blog-figs/Blog-figs/relu_loss_landscape.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Loss landscape plot saved as relu_loss_landscape.png")
    
    return beta_range, losses


def compare_convergence_paths():
    """Compare convergence paths of different solvers"""
    print("Comparing convergence paths...")
    
    # Generate test data
    n, d, L = 30, 2, 2
    X, y, U, V, beta_true = generate_test_data(n, d, L, random_seed=789)
    reg_lambda = 0.01
    
    # Modified version of original solver to track convergence
    def solve_relu_qp_tracked(X, y, U, V, L, n, d, reg_lambda=0.01, max_iter=100):
        beta = np.linalg.pinv(X) @ y
        beta_history = [beta.copy()]
        loss_history = []
        
        for iter_num in range(max_iter):
            beta_old = beta.copy()
            
            residuals = y - X @ beta
            total_subgrad = np.zeros(d)
            total_loss = 0
            
            for l in range(L):
                for i in range(n):
                    arg = U[l, i] * residuals[i] + V[l, i]
                    if arg > 0:
                        total_subgrad += U[l, i] * (-X[i, :])
                        total_loss += arg
            
            grad = total_subgrad + reg_lambda * beta
            learning_rate = 0.01 / (1 + 0.1 * iter_num)
            beta = beta - learning_rate * grad
            
            beta_history.append(beta.copy())
            final_loss = total_loss + 0.5 * reg_lambda * np.linalg.norm(beta)**2
            loss_history.append(final_loss)
            
            if np.linalg.norm(beta - beta_old) < 1e-6:
                break
        
        return beta, beta_history, loss_history
    
    # Track convergence
    beta_orig, beta_hist, loss_hist = solve_relu_qp_tracked(X, y, U, V, L, n, d, reg_lambda)
    
    # Get ReHLine solution
    if REHLINE_AVAILABLE:
        beta_rehline, _, _, _, loss_rehline = solve_relu_qp_with_rehline(X, y, U, V, L, n, d, reg_lambda)
    
    # Create convergence plot
    plt.figure(figsize=(15, 5))
    
    # Loss convergence
    plt.subplot(1, 3, 1)
    plt.plot(loss_hist, 'b-', marker='o', markersize=3, label='Original solver')
    if REHLINE_AVAILABLE:
        plt.axhline(loss_rehline, color='green', linestyle='--', label=f'ReHLine final: {loss_rehline:.3f}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Convergence')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # Parameter convergence (2D trajectory)
    if d == 2:
        plt.subplot(1, 3, 2)
        beta_hist_array = np.array(beta_hist)
        plt.plot(beta_hist_array[:, 0], beta_hist_array[:, 1], 'b-', marker='o', markersize=3, label='Original path')
        plt.plot(beta_hist_array[0, 0], beta_hist_array[0, 1], 'ro', markersize=8, label='Start')
        plt.plot(beta_orig[0], beta_orig[1], 'bs', markersize=8, label='Original final')
        
        if REHLINE_AVAILABLE:
            plt.plot(beta_rehline[0], beta_rehline[1], 'g^', markersize=8, label='ReHLine')
        
        plt.plot(beta_true[0], beta_true[1], 'k*', markersize=10, label='True β')
        plt.xlabel('β₁')
        plt.ylabel('β₂')
        plt.title('Parameter Trajectory')
        plt.legend()
        plt.grid(True)
    
    # Distance to true solution
    plt.subplot(1, 3, 3)
    distances = [np.linalg.norm(beta - beta_true) for beta in beta_hist]
    plt.plot(distances, 'b-', marker='o', markersize=3, label='Original solver')
    if REHLINE_AVAILABLE:
        plt.axhline(np.linalg.norm(beta_rehline - beta_true), color='green', linestyle='--', label='ReHLine distance')
    plt.xlabel('Iteration')
    plt.ylabel('||β - β_true||')
    plt.title('Distance to True Solution')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('/home/runner/work/Blog-figs/Blog-figs/convergence_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Convergence comparison plot saved as convergence_comparison.png")
    
    return beta_hist, loss_hist


def demonstrate_regularization_effect():
    """Demonstrate the effect of regularization parameter"""
    print("Demonstrating regularization effects...")
    
    # Generate test data
    n, d, L = 40, 2, 2
    X, y, U, V, beta_true = generate_test_data(n, d, L, random_seed=101112)
    
    # Test different regularization values
    reg_values = np.logspace(-4, 2, 20)  # From 0.0001 to 100
    
    results_orig = []
    results_rehline = []
    
    for reg_lambda in reg_values:
        # Original solver
        beta_orig, _, loss_orig = solve_relu_qp(X, y, U, V, L, n, d, reg_lambda, max_iter=150)
        results_orig.append({
            'reg_lambda': reg_lambda,
            'beta': beta_orig,
            'loss': loss_orig,
            'distance_to_true': np.linalg.norm(beta_orig - beta_true)
        })
        
        # ReHLine solver
        if REHLINE_AVAILABLE:
            try:
                beta_rehline, _, _, _, loss_rehline = solve_relu_qp_with_rehline(X, y, U, V, L, n, d, reg_lambda)
                results_rehline.append({
                    'reg_lambda': reg_lambda,
                    'beta': beta_rehline,
                    'loss': loss_rehline,
                    'distance_to_true': np.linalg.norm(beta_rehline - beta_true)
                })
            except:
                results_rehline.append(None)
    
    # Create regularization effect plots
    plt.figure(figsize=(15, 10))
    
    # Loss vs regularization
    plt.subplot(2, 3, 1)
    losses_orig = [r['loss'] for r in results_orig]
    plt.semilogx(reg_values, losses_orig, 'b-o', markersize=4, label='Original')
    
    if results_rehline:
        losses_rehline = [r['loss'] if r else np.nan for r in results_rehline]
        plt.semilogx(reg_values, losses_rehline, 'g-s', markersize=4, label='ReHLine')
    
    plt.xlabel('Regularization λ')
    plt.ylabel('Final Loss')
    plt.title('Loss vs Regularization')
    plt.legend()
    plt.grid(True)
    
    # Distance to true solution vs regularization
    plt.subplot(2, 3, 2)
    distances_orig = [r['distance_to_true'] for r in results_orig]
    plt.loglog(reg_values, distances_orig, 'b-o', markersize=4, label='Original')
    
    if results_rehline:
        distances_rehline = [r['distance_to_true'] if r else np.nan for r in results_rehline]
        plt.loglog(reg_values, distances_rehline, 'g-s', markersize=4, label='ReHLine')
    
    plt.xlabel('Regularization λ')
    plt.ylabel('||β - β_true||')
    plt.title('Error vs Regularization')
    plt.legend()
    plt.grid(True)
    
    # Parameter paths (for 2D case)
    if d == 2:
        plt.subplot(2, 3, 3)
        betas_orig = np.array([r['beta'] for r in results_orig])
        plt.plot(betas_orig[:, 0], betas_orig[:, 1], 'b-o', markersize=3, label='Original path')
        
        if results_rehline:
            betas_rehline = np.array([r['beta'] if r else [np.nan, np.nan] for r in results_rehline])
            plt.plot(betas_rehline[:, 0], betas_rehline[:, 1], 'g-s', markersize=3, label='ReHLine path')
        
        plt.plot(beta_true[0], beta_true[1], 'k*', markersize=10, label='True β')
        plt.xlabel('β₁')
        plt.ylabel('β₂')
        plt.title('Regularization Path')
        plt.legend()
        plt.grid(True)
    
    # Individual parameter evolution
    plt.subplot(2, 3, 4)
    for i in range(d):
        params_orig = [r['beta'][i] for r in results_orig]
        plt.semilogx(reg_values, params_orig, f'C{i}-o', markersize=3, label=f'β_{i+1} (Original)')
        
        if results_rehline:
            params_rehline = [r['beta'][i] if r else np.nan for r in results_rehline]
            plt.semilogx(reg_values, params_rehline, f'C{i}--s', markersize=3, label=f'β_{i+1} (ReHLine)')
        
        plt.axhline(beta_true[i], color=f'C{i}', linestyle=':', label=f'True β_{i+1}')
    
    plt.xlabel('Regularization λ')
    plt.ylabel('Parameter Value')
    plt.title('Individual Parameters')
    plt.legend()
    plt.grid(True)
    
    # Solver difference
    if results_rehline:
        plt.subplot(2, 3, 5)
        beta_diffs = []
        loss_diffs = []
        
        for i, (orig, rehline) in enumerate(zip(results_orig, results_rehline)):
            if rehline is not None:
                beta_diff = np.linalg.norm(orig['beta'] - rehline['beta'])
                loss_diff = abs(orig['loss'] - rehline['loss'])
                beta_diffs.append(beta_diff)
                loss_diffs.append(loss_diff)
            else:
                beta_diffs.append(np.nan)
                loss_diffs.append(np.nan)
        
        plt.loglog(reg_values, beta_diffs, 'r-o', markersize=4, label='Parameter difference')
        plt.loglog(reg_values, loss_diffs, 'm-s', markersize=4, label='Loss difference')
        plt.xlabel('Regularization λ')
        plt.ylabel('Difference')
        plt.title('Solver Differences')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/runner/work/Blog-figs/Blog-figs/regularization_effects.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Regularization effects plot saved as regularization_effects.png")
    
    return results_orig, results_rehline


def main():
    """Main demonstration function"""
    print("="*60)
    print("RELU QP SOLVER DEMONSTRATION")
    print("="*60)
    
    print(f"ReHLine library available: {REHLINE_AVAILABLE}")
    
    # Run demonstrations
    visualize_loss_landscape_1d()
    compare_convergence_paths()
    demonstrate_regularization_effect()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETED")
    print("="*60)
    print("\nGenerated plots:")
    print("  - relu_loss_landscape.png: 1D loss landscape visualization")
    print("  - convergence_comparison.png: Convergence behavior comparison")
    print("  - regularization_effects.png: Effect of regularization parameter")
    
    if REHLINE_AVAILABLE:
        print("\n✓ All demonstrations completed successfully with ReHLine!")
    else:
        print("\n⚠ Demonstrations completed with original solver only (ReHLine not available)")


if __name__ == "__main__":
    main()