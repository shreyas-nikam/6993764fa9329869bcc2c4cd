import time
import sys

print("Starting import test...")
start = time.time()

try:
    from source import prepare_credit_data
    print(f"Import completed in {time.time() - start:.2f} seconds")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
