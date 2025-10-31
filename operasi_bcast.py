# File: collective_bcast.py
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    # Rank 0: Inisialisasi data. Ukuran 5 integer.
    data = np.array([10, 20, 30, 40, 50], dtype='i')
    print(f"Rank {rank}: Mengirim data {data}")

else:
    # Rank lain: Alokasikan buffer kosong dengan ukuran yang sama
    data = np.empty(5, dtype='i')

# Semua proses berpartisipasi dalam Bcast.
# Rank 0 adalah Root, yang lain menerima.
comm.Bcast([data, MPI.INT], root=0)

# Verifikasi hasil di semua proses
print(f"Rank {rank}: Menerima data hasil Bcast: {data}")

