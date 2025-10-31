# File: collective_reduce.py
from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# Setiap Rank memiliki nilai tunggal yang unik
local_value = np.array([rank + 1], dtype='f') # Menggunakan float untuk fleksibilitas
if rank == 0:
    # Rank 0: Alokasikan buffer penerima untuk hasil akhir (1 elemen)
    global_sum = np.empty(1, dtype='f') 
else:
    global_sum = None
# Reduce: Menjumlahkan semua local_value, hasilnya di global_sum (Rank 0)
comm.Reduce([local_value, MPI.FLOAT], [global_sum, MPI.FLOAT], 
            op=MPI.SUM, root=0)
if rank == 0:
    # Jika size = 4, maka Sum = 1 + 2 + 3 + 4 = 10.0
    print(f"Rank {rank}: Total jumlah global adalah: {global_sum[0]}")
else:
    print(f"Rank {rank}: Nilai lokal ({local_value[0]}) telah dikirim ke Root.")

