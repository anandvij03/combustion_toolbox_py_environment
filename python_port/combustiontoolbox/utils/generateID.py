import hashlib
import struct

def generateID(value: str) -> float:
    """
    Generate a deterministic numeric identifier from input data.
    
    Args:
       value (str): Input character array (string) used to generate the ID
    
    Returns:
       id (float): Deterministic 32-bit numeric identifier (returned as a float to match MATLAB's double)
    
    Example:
       id = generateID('this_is_an_example')
    """
    # Compute 128-bit MD5 hash from input string
    md5_hash = hashlib.md5(value.encode('utf-8')).digest()

    # Convert the 16-byte MD5 hash into four uint32 integers
    # '<IIII' unpacks 16 bytes into 4 little-endian unsigned 32-bit integers
    hash_ints = struct.unpack('<IIII', md5_hash)

    # Collapse the 128-bit digest into a single 32-bit numeric identifier
    id_int = hash_ints[0] ^ hash_ints[1] ^ hash_ints[2] ^ hash_ints[3]
    
    return float(id_int)
