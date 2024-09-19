import numpy as np

def pack_int4(int4_array):
    # Ensure values are within the range of 0 to 15
    int4_array = np.clip(int4_array + 8, 0, 15).astype(np.uint8)
    
    # Calculate the number of bytes needed
    num_bytes = (len(int4_array) + 1) // 2
    
    # Initialize packed array
    packed_array = np.zeros(num_bytes, dtype=np.uint8)
    
    # Pack int4 values into bytes
    for i in range(0, len(int4_array), 2):
        high_bits = (int4_array[i] & 0xF) << 4
        low_bits = int4_array[i + 1] & 0xF if i + 1 < len(int4_array) else 0
        packed_array[i // 2] = high_bits | low_bits
    
    return packed_array

def quantize_vector(fp32_vectors, target_precision=''):
    if target_precision == 'int8':
        scale_factor = 127.0 / np.max(np.abs(fp32_vectors))  # Scale FP32 to [-127, 127] for INT8
        int8_quantized = np.clip(np.round(fp32_vectors * scale_factor), -128, 127).astype(np.int8)
        print("INT8 Quantized Values:", int8_quantized[0])
        print("INT8 Length:", len(int8_quantized[0]))
        return int8_quantized
    
    elif target_precision == 'int4':
        scale_factor = 7.0 / np.max(np.abs(fp32_vectors))
        int4_quantized = np.clip(np.round(fp32_vectors * scale_factor), -8, 7).astype(np.int8)
        print("INT4 Quantized Values:", int4_quantized[0])
        print("INT4 Length Before Packing:", len(int4_quantized.flatten()))

        # Pack INT4 values
        int4_quantized_packed = pack_int4(int4_quantized.flatten())
        
        print("Packed INT4 Vector Length:", len(int4_quantized_packed))
        return int4_quantized_packed
    
    elif target_precision == 'binary':
        binary_vectors = (fp32_vectors > 0).astype(np.uint8)
        return binary_vectors
    
    else:
        raise TypeError('Unsupported precision type')

# Example FP32 vector for testing
fp32_vector = np.array([[1.234, -0.567, 3.456, -2.345, 0.678]], dtype=np.float32)

# Test the function
quantize_vector(fp32_vector, 'int8')
quantize_vector(fp32_vector, 'int4')
