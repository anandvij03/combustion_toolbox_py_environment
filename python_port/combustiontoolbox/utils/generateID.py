import hashlib
import numpy as np

def generate_id_numpy(value: str) -> float:
    # Compute 128-bit MD5 hash
    md5_bytes = hashlib.md5(value.encode('utf-8')).digest()
    
    # Equivalent to MATLAB's typecast(..., 'uint32')
    hash_array = np.frombuffer(md5_bytes, dtype=np.uint32)
    
    # Vectorized XOR reduction over the array
    id_int = np.bitwise_xor.reduce(hash_array)
    
    # Return as a float to match MATLAB's double() cast
    return float(id_int)