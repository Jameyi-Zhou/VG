import torch
torch.cuda.reset_max_memory_allocated()  # Resets the starting point of memory usage accounting
torch.cuda.reset_accumulated_memory_stats()  # Resets the historical memory usage accounting