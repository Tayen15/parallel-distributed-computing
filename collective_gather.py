from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
N_local = 3

send_data = np.full(N_local, rank + 1, dtype="i")
print(f"Rank {rank}: Data yang akan di Gather: {send_data}")

if rank == 0:
    recv_data = np.empty(size * N_local, dtype="i")
else:
    recv_data = None  # Hanya Root yang perlu buffer ini
    
comm.Gather([send_data, MPI.INT], [recv_data, MPI.INT], root=0)
if rank == 0:
    print(f"Rank {rank}: Data hasil Gather (total array): {recv_data}")