import torch
import sys

def check_cuda():
    print("PyTorch version:", torch.__version__)
    
    print("\nCUDA information:")
    print("CUDA available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("Number of GPUs:", torch.cuda.device_count())
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i} information:")
            print("  Name:", torch.cuda.get_device_name(i))
            print("  Capability:", torch.cuda.get_device_capability(i))
            
            # Get memory information
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            
            print(f"  Memory: Total {mem_total:.2f} GB | Reserved {mem_reserved:.2f} GB | Allocated {mem_allocated:.2f} GB")
            
        # Set current device to 0
        torch.cuda.set_device(0)
        print(f"\nCurrent device: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
        
        # Test CUDA with a simple operation
        print("\nPerforming a simple CUDA operation:")
        try:
            x = torch.rand(1000, 1000).cuda()
            y = torch.rand(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("  CUDA operation successful!")
        except Exception as e:
            print(f"  CUDA operation failed: {e}")
    else:
        print("CUDA is not available. Please check your installation.")
        
if __name__ == "__main__":
    check_cuda()
    sys.exit(0) 