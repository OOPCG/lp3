def fib_iterative(n):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1

    for i in range(2, n + 1):
        next_fib = a + b
        a = b
        b = next_fib
        
    return b

# ==============================================================================
# 2. Recursive Approach
# Time Complexity: O(2^n) | Space Complexity: O(n)
# This method is conceptually simpler but highly inefficient due to redundant calls.
# ==============================================================================
def fib_recursive(n):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return fib_recursive(n - 1) + fib_recursive(n - 2)

# --- Example Usage ---
n_value = 10

print(f"Calculating the {n_value}th Fibonacci number:")

# Iterative Test
iterative_result = fib_iterative(n_value)
print(f"Iterative Result (O({n_value})): {iterative_result}") 

# Recursive Test
recursive_result = fib_recursive(n_value)
print(f"Recursive Result (O(2^{n_value})): {recursive_result}")
