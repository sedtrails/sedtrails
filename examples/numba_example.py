import time
from numba import jit
 
# A simple function to calculate sum of squares in a range
def sum_squares_python(n):
    result = 0
    for i in range(n):
        result += i * i
    return result
 
# Same function with Numba optimization
@jit(nopython=True)
def sum_squares_numba(n):
    result = 0
    for i in range(n):
        result += i * i
    return result
 
# Test the performance
if __name__ == "__main__":
    n = 1000000000  
   
    # Time the standard Python function
    start = time.time()
    result_python = sum_squares_python(n)
    python_time = time.time() - start
   
    # Time the Numba function (first run includes compilation)
    start = time.time()
    result_numba = sum_squares_numba(n)
    numba_time = time.time() - start
   
    # Print results
    print(f"Sum of squares from 0 to {n:,}: {result_python}")
    print(f"Python time: {python_time:.4f} seconds")
    print(f"Numba time: {numba_time:.4f} seconds")
    print(f"Speedup: {python_time / numba_time:.2f}x faster with Numba")