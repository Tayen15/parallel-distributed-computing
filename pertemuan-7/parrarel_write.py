from mpi4py import MPI
import numpy as np
import os  # untuk menghapus berkas

# Inisialisasi MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
filename = "data_output.bin"
data_per_proc = 10  # Jumlah elemen yang ditulis setiap proses
total_data = data_per_proc * size

# 1. Definisi Data
# Setiap proses membuat array lokal
local_array = np.arange(
    rank * data_per_proc, (rank + 1) * data_per_proc, dtype=np.int32
)
buffer = local_array

# Hapus berkas lama jika ada (hanya proses 0)
if rank == 0:
    if os.path.exists(filename):
        os.remove(filename)

# Sinkronisasi agar semua proses menunggu berkas dihapus
comm.Barrier()

# 2. Buka Berkas (Operasi Kolektif)
# Gunakan mode TULIS, BUAT, dan OVERWRITE
amode = MPI.MODE_WRONLY | MPI.MODE_CREATE | MPI.MODE_EXCL
try:
    fh = MPI.File.Open(comm, filename, amode)
except MPI.Exception as e:
    print(f"Error pada rank {rank}: {e}")
    comm.Abort()


# 3. Hitung Offset dan Ukuran Data
# Offset adalah posisi awal penulisan data proses ini dalam byte.
# Setiap elemen int32 berukuran 4 byte.
datatype = MPI.INT  # Tipe data MPI untuk np.int32
element_size = datatype.Get_size()
offset = rank * data_per_proc * element_size

# 4. Tulis Data Kolektif (Write_at_all)
# Write_at_all: menulis data pada 'offset' tanpa perlu mengubah file pointer.
# Ini adalah operasi kolektif, semua proses harus memanggilnya.
# Data ditulis langsung dari 'buffer'
fh.Write_at_all(offset, buffer)

# 5. Tutup Berkas (Operasi Kolektif)
fh.Close()

# 6. Verifikasi (Hanya Proses 0)
if rank == 0:
    print(f"Semua proses telah selesai menulis. Total data: {total_data} elemen.")
    # Membaca kembali semua data untuk verifikasi
    read_data = np.empty(total_data, dtype=np.int32)
    with open(filename, "rb") as f:
        read_data = np.fromfile(f, dtype=np.int32)

    print("\nData yang tersimpan di berkas (seharusnya 0 sampai N-1):")
    print(read_data)

    # Hapus berkas setelah selesai
    # os.remove(filename)
