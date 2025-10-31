# File: collective_allreduce.py
from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
local_value = np.array([rank + 1], dtype='f')
global_sum = np.empty(1, dtype='f') 
# Allreduce: Menghitung jumlah total dan mendistribusikannya ke SEMUA Rank.
# Semua Rank perlu buffer penerima (global_sum).
comm.Allreduce([local_value, MPI.FLOAT], [global_sum,MPI.FLOAT],
               op=MPI.SUM)
# Semua Rank sekarang memiliki hasil yang sama
print(f"Rank {rank}: Total jumlah global (hasil Allreduce) adalah: {global_sum[0]}")

