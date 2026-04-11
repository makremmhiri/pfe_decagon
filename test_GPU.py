import tensorflow.compat.v1 as tf
import sys
import os

def test_gpu_setup():
    print("--- TensorFlow GPU Diagnostic ---")
    
    # Check Python and TF version
    print(f"Python version: {sys.version}")
    print(f"TensorFlow version: {tf.__version__}")

    # Check for visible GPU devices
    gpu_devices = tf.config.list_physical_devices('GPU')
    
    print("-" * 30)
    if not gpu_devices:
        print("RESULT: No GPU detected by TensorFlow.")
        print("\nPOSSIBLE PROBLEMS:")
        
        # 1. Check for the common DLLs missing in your previous logs
        print("Checking for common missing drivers...")
        important_dlls = [
            'cudart64_110.dll', 
            'cublas64_11.dll', 
            'cudnn64_8.dll', 
            'cusolver64_11.dll'
        ]
        
        # This checks the system PATH to see if CUDA is actually there
        path_env = os.environ.get('PATH', '').lower()
        found_cuda = "cuda" in path_env or "nvidia" in path_env
        
        if not found_cuda:
            print("  [!] CUDA not found in System PATH. Have you installed the CUDA Toolkit?")
        else:
            print("  [?] CUDA folders found in PATH, but DLLs might be the wrong version.")
            
        print("\nREQUIRED ACTION:")
        print("1. Ensure NVIDIA Drivers are updated.")
        print("2. Install CUDA Toolkit (Recommended: 11.2 or 11.8 for your TF version).")
        print("3. Download cuDNN and copy the DLL files into the CUDA folder.")
    else:
        print(f"SUCCESS: TensorFlow found {len(gpu_devices)} GPU(s):")
        for gpu in gpu_devices:
            print(f"  - {gpu}")
            
    print("-" * 30)

if __name__ == "__main__":
    test_gpu_setup()