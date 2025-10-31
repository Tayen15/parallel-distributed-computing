# File: collective_gather.py
from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
N_local = 3 
# Data lokal unik yang dihasilkan oleh setiap Rank
send_data = np.full(N_local, rank + 1, dtype='i') 
print(f"Rank {rank}: Data lokal yang akan di Gather: {send_data}")
if rank == 0:
    # Rank 0: Alokasikan buffer besar untuk menerima SEMUA data
    recv_data = np.empty(size * N_local, dtype='i')
else:
    recv_data = None

# Gather: Semua proses mengirim, hanya Root yang menerima
comm.Gather([send_data, MPI.INT], [recv_data, MPI.INT], root=0)
if rank == 0:
    print(f"Rank {rank}: Data hasil Gather (total array): {recv_data}")

