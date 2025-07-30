"""
Test suite for ReLU QP solver implementations
"""

import numpy as np
import matplotlib.pyplot as plt
from relu_qp_solver import (
    solve_relu_qp, 
    solve_relu_qp_with_rehline, 
    generate_test_data, 
    compare_solvers,
    REHLINE_AVAILABLE
)


def test_convergence_comparison():
    """Test convergence properties of both solvers"""
    print("Testing convergence comparison...")
    
    # Generate test data
    n, d, L = 30, 2, 2
    X, y, U, V, beta_true = generate_test_data(n, d, L, random_seed=123)
    
    # Test different regularization parameters
    reg_lambdas = [0.001, 0.01, 0.1, 1.0]
    results = {}
    
    for reg_lambda in reg_lambdas:
        print(f"\nTesting with λ = {reg_lambda}")
        
        # Original solver
        beta_orig, residuals_orig, loss_orig = solve_relu_qp(
            X, y, U, V, L, n, d, reg_lambda, max_iter=200
        )
        
        results[reg_lambda] = {
            'original': {'beta': beta_orig, 'loss': loss_orig}
        }
        
        # ReHLine solver (if available)
        if REHLINE_AVAILABLE:
            try:
                beta_rehline, pi_star, lambda_star, residuals_rehline, loss_rehline = solve_relu_qp_with_rehline(
                    X, y, U, V, L, n, d, reg_lambda
                )
                
                results[reg_lambda]['rehline'] = {
                    'beta': beta_rehline, 
                    'loss': loss_rehline,
                    'pi_star': pi_star,
                    'lambda_star': lambda_star
                }
                
                # Compute differences
                beta_diff = np.linalg.norm(beta_orig - beta_rehline)
                loss_diff = abs(loss_orig - loss_rehline)
                
                print(f"  Original loss: {loss_orig:.6f}")
                print(f"  ReHLine loss:  {loss_rehline:.6f}")
                print(f"  Difference:    {loss_diff:.6f}")
                print(f"  Beta L2 diff:  {beta_diff:.6f}")
                
                results[reg_lambda]['comparison'] = {
                    'beta_diff': beta_diff,
                    'loss_diff': loss_diff
                }
                
            except Exception as e:
                print(f"  ReHLine failed: {e}")
                results[reg_lambda]['rehline'] = {'error': str(e)}
    
    return results


def test_dual_variables():
    """Test extraction and validity of dual variables from ReHLine"""
    if not REHLINE_AVAILABLE:
        print("ReHLine not available, skipping dual variable test")
        return None
    
    print("Testing dual variables extraction...")
    
    # Generate simple test data
    n, d, L = 20, 2, 1  # Single ReLU term for simplicity
    X, y, U, V, beta_true = generate_test_data(n, d, L, random_seed=456)
    
    beta_rehline, pi_star, lambda_star, residuals, loss = solve_relu_qp_with_rehline(
        X, y, U, V, L, n, d, reg_lambda=0.1
    )
    
    print(f"Solution beta: {beta_rehline}")
    print(f"Dual variables pi (Lambda): {pi_star}")
    print(f"Shape of pi: {pi_star.shape if pi_star is not None else 'None'}")
    
    # Verify KKT conditions (basic check)
    if pi_star is not None:
        print("Checking complementary slackness...")
        for i in range(n):
            for l in range(L):
                arg = U[l, i] * residuals[i] + V[l, i]
                dual_val = pi_star[l, i] if pi_star.shape == (L, n) else pi_star[i] if L == 1 else 0
                
                if i < 5:  # Print first few for inspection
                    print(f"  Sample {i}, ReLU {l}: arg={arg:.4f}, dual={dual_val:.4f}")
                
                # Complementary slackness: pi_i * max(0, arg_i) should equal pi_i * arg_i
                if arg <= 0 and dual_val > 1e-6:
                    print(f"  WARNING: Complementary slackness violation at sample {i}, ReLU {l}")
    
    return {
        'beta': beta_rehline,
        'pi_star': pi_star,
        'lambda_star': lambda_star,
        'residuals': residuals,
        'loss': loss
    }


def test_scaling_performance():
    """Test performance with different problem sizes"""
    print("Testing scaling performance...")
    
    sizes = [(20, 2, 1), (50, 3, 2), (100, 5, 2)]
    timing_results = {}
    
    for n, d, L in sizes:
        print(f"\nTesting size n={n}, d={d}, L={L}")
        
        # Generate data
        X, y, U, V, beta_true = generate_test_data(n, d, L)
        
        # Time original solver
        import time
        
        start_time = time.time()
        beta_orig, _, loss_orig = solve_relu_qp(X, y, U, V, L, n, d, max_iter=100)
        orig_time = time.time() - start_time
        
        timing_results[(n, d, L)] = {
            'original': {'time': orig_time, 'loss': loss_orig}
        }
        
        print(f"  Original solver: {orig_time:.4f}s, loss: {loss_orig:.6f}")
        
        # Time ReHLine solver
        if REHLINE_AVAILABLE:
            try:
                start_time = time.time()
                beta_rehline, _, _, _, loss_rehline = solve_relu_qp_with_rehline(X, y, U, V, L, n, d)
                rehline_time = time.time() - start_time
                
                timing_results[(n, d, L)]['rehline'] = {
                    'time': rehline_time, 
                    'loss': loss_rehline
                }
                
                print(f"  ReHLine solver:  {rehline_time:.4f}s, loss: {loss_rehline:.6f}")
                print(f"  Speedup:         {orig_time/rehline_time:.2f}x")
                
            except Exception as e:
                print(f"  ReHLine failed: {e}")
                timing_results[(n, d, L)]['rehline'] = {'error': str(e)}
    
    return timing_results


def test_edge_cases():
    """Test edge cases and robustness"""
    print("Testing edge cases...")
    
    test_cases = []
    
    # Case 1: Very small regularization
    print("\n1. Testing very small regularization...")
    n, d, L = 20, 2, 1
    X, y, U, V, _ = generate_test_data(n, d, L)
    
    try:
        results = compare_solvers(X, y, U, V, L, n, d, reg_lambda=1e-6)
        test_cases.append(('small_reg', results))
        print("   ✓ Small regularization test passed")
    except Exception as e:
        print(f"   ✗ Small regularization test failed: {e}")
        test_cases.append(('small_reg', {'error': str(e)}))
    
    # Case 2: Large regularization
    print("\n2. Testing large regularization...")
    try:
        results = compare_solvers(X, y, U, V, L, n, d, reg_lambda=100.0)
        test_cases.append(('large_reg', results))
        print("   ✓ Large regularization test passed")
    except Exception as e:
        print(f"   ✗ Large regularization test failed: {e}")
        test_cases.append(('large_reg', {'error': str(e)}))
    
    # Case 3: All positive/negative ReLU arguments
    print("\n3. Testing extreme ReLU configurations...")
    U_pos = np.ones((L, n))  # All positive coefficients
    V_pos = np.ones((L, n))  # All positive intercepts
    
    try:
        results = compare_solvers(X, y, U_pos, V_pos, L, n, d, reg_lambda=0.01)
        test_cases.append(('positive_relu', results))
        print("   ✓ Positive ReLU test passed")
    except Exception as e:
        print(f"   ✗ Positive ReLU test failed: {e}")
        test_cases.append(('positive_relu', {'error': str(e)}))
    
    return test_cases


def run_all_tests():
    """Run all test suites"""
    print("="*60)
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    all_results = {}
    
    # Test 1: Convergence comparison
    all_results['convergence'] = test_convergence_comparison()
    
    print("\n" + "="*60)
    
    # Test 2: Dual variables
    all_results['dual_variables'] = test_dual_variables()
    
    print("\n" + "="*60)
    
    # Test 3: Scaling performance
    all_results['scaling'] = test_scaling_performance()
    
    print("\n" + "="*60)
    
    # Test 4: Edge cases
    all_results['edge_cases'] = test_edge_cases()
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETED")
    print("="*60)
    
    return all_results


if __name__ == "__main__":
    results = run_all_tests()
    
    # Summary
    print("\nSUMMARY:")
    print(f"ReHLine available: {REHLINE_AVAILABLE}")
    
    if 'convergence' in results:
        conv_results = results['convergence']
        print(f"Convergence tests completed for {len(conv_results)} regularization values")
        
        # Check if results are consistent
        all_consistent = True
        for reg_lambda, result in conv_results.items():
            if 'comparison' in result:
                if result['comparison']['beta_diff'] > 0.1:
                    all_consistent = False
                    print(f"  WARNING: Large difference at λ={reg_lambda}")
        
        if all_consistent:
            print("  ✓ All solvers show consistent results")
        else:
            print("  ⚠ Some solver differences detected")
    
    if 'dual_variables' in results and results['dual_variables']:
        print("  ✓ Dual variables successfully extracted")
    
    if 'scaling' in results:
        print(f"  ✓ Scaling tests completed for {len(results['scaling'])} problem sizes")
    
    if 'edge_cases' in results:
        passed_cases = sum(1 for name, result in results['edge_cases'] if 'error' not in result)
        total_cases = len(results['edge_cases'])
        print(f"  ✓ Edge case tests: {passed_cases}/{total_cases} passed")