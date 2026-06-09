import time
import platform
import os
import inspect
import sys
import pandas as pd
from typing import Callable, List

class Benchmark:
    """
    The Benchmark class is used to perform a set of benchmark tests.
    It measures and reports the average computational time of selected 
    functions over multiple iterations.

    Example usage:
        # Define benchmark tests as functions in the same script:
        def run_validation_TP_CEA_6():
            # Example calculation / simulation
            pass
        
        # Initialize (it will automatically find run_validation_TP_CEA_6)
        bench = Benchmark(num_iterations=20)
        
        # Run the tests and display the report
        bench.run().report()
    """

    def __init__(self, tests: List[Callable] = None, num_iterations: int = 10):
        # Set properties
        self.num_iterations = num_iterations
        self.metadata = []
        self.system = self._get_system_info()
        
        # Must be set after num_iterations and metadata initialization
        self.tests = tests if tests is not None else self.get_default_tests()

    @property
    def num_test(self) -> int:
        """Get number of tests"""
        return len(self.tests)

    def set(self, **kwargs):
        """
        Set properties of the Benchmark object using keyword arguments.
        
        Examples:
            bench.set(num_iterations=20)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Property '{key}' not found in Benchmark class.")
        return self

    def run(self):
        """
        Executes the benchmark tests and records the average 
        execution time for each test function.
        """
        if not self.tests:
            print("No matching validation tests found to run.")
            return self

        for test_func in self.tests:

            test_name = test_func.__name__ if hasattr(test_func, '__name__') else str(test_func)
            print(f"Running test: {test_name}...")

            start_time = time.perf_counter()
            result = None
            for _ in range(self.num_iterations):
                result = test_func()
            end_time = time.perf_counter()

            avg_time = (end_time - start_time) / self.num_iterations

            # Default base metadata values
            meta_record = {
                'Module': 'Validation',       
                'Problem': test_name,
                'Cases': 1,                   
                'Species': 0,                 
                'AvgTime': avg_time
            }

            if isinstance(result, dict):
                meta_record.update(result)
            elif hasattr(result, '__dict__'):
                meta_record.update(result.__dict__)
            
            # Ensure final tracked average time is written
            meta_record['AvgTime'] = avg_time
            self.metadata.append(meta_record)

            print(f"{test_name:<40} | Average Time = {avg_time:.6f} seconds")
            
        return self

    def report(self):
        """
        Displays a formatted benchmark report, showing the 
        average execution times for each test.
        """
        if not self.metadata:
            print("Warning: No benchmark results available. Run the benchmarks first.")
            return

        # 1. System Information Table
        sys_df = pd.DataFrame([{
            'App Version': 'v1.2.9',  
            'Python Version': platform.python_version(),
            'OS Version': self.system['OSVersion'],
            'CPU Name': self.system['CPUName'],
            'Cores': self.system['TotalCores']
        }])

        # 2. Extended Benchmark Table
        bench_df = pd.DataFrame(self.metadata)

        # 3. Summary Benchmark Table
        summary_df = bench_df.groupby(['Module', 'Problem']).agg(
            Cases=('Cases', 'sum'),
            Species=('Species', 'mean'),
            AvgTime=('AvgTime', 'mean')
        ).reset_index()

        # Print Output
        print("\n" + "="*80)
        print("BENCHMARK REPORT".center(80))
        print("="*80 + "\n")

        print("System Information")
        print(sys_df.to_string(index=False))
        print("\n")

        print("Benchmarking")
        print(bench_df.to_string(index=False))
        print("\n")

        print("Summary Benchmarking")
        print(summary_df.to_string(index=False))
        print("\n")

    # HELPER FUNCTIONS

    def _get_system_info(self) -> dict:
        """Helper to replace MATLAB's cpuinfo()"""
        return {
            'OSVersion': platform.system() + " " + platform.release(),
            'CPUName': platform.processor() or "Unknown Processor",
            'TotalCores': os.cpu_count() or 1
        }

    @staticmethod
    def get_default_tests() -> List[Callable]:
        """
        Automatically grabs all validation functions defined or imported
        within the current file context.
        """
        test_list = []

        current_module = sys.modules[__name__]
        
        for name, func in inspect.getmembers(current_module, inspect.isfunction):
            if name.startswith('run_validation'):
                test_list.append(func)
                
        return test_list