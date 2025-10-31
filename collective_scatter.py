from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
N_local = 4

if rank == 0:
    # Total elemen: size * N_local. Misal size=4, total 16.
    send_data = np.arange(size * N_local, dtype="i")
    print(f"Rank {rank}: Data total yang akan di Scatter: {send_data}")
else:
    send_data = None  # Hanya Root yang perlu data ini

recv_data = np.empty(N_local, dtype="i")

# Scatter: Setiap proses menerima N_local elemen
comm.Scatter([send_data, MPI.INT], [recv_data, MPI.INT], root=0)
print(f"Rank {rank}: Menerima chunk data: {recv_data}")
