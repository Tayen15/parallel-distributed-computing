# ============================================================
# Program: parallel_column_write.py
# Deskripsi: Demonstrasi penulisan file paralel (column-wise)
#            menggunakan MPI-IO dengan mpi4py.
# Setiap proses MPI menulis satu kolom dari matriks 4x4 ke file biner tunggal.
# ============================================================

from mpi4py import MPI
import numpy as np
import os

# ------------------------------------------------------------
# 1. Inisialisasi MPI
# ------------------------------------------------------------
comm = MPI.COMM_WORLD  # Komunikator global (semua proses)
rank = comm.Get_rank()  # Nomor proses (0, 1, 2, 3)
size = comm.Get_size()  # Jumlah total proses yang dijalankan

# Program ini dirancang khusus untuk dijalankan dengan 4 proses
if size != 4:
    if rank == 0:
        print("Program ini harus dijalankan dengan 4 proses (mpiexec -n 4).")
    comm.Abort()  # Hentikan seluruh proses jika jumlah tidak sesuai

# ------------------------------------------------------------
# 2. Parameter global matriks dan nama file keluaran
# ------------------------------------------------------------
filename = "matrix_output.bin"
rows, cols = 4, 4  # Matriks 4x4 → total 16 elemen integer

# ------------------------------------------------------------
# 3. Persiapan data lokal (tiap proses menulis satu kolom)
# ------------------------------------------------------------
# Misal rank 0 menulis kolom 0: [0, 1, 2, 3]
# rank 1 menulis kolom 1: [10, 11, 12, 13]
# dst.
local_data = np.arange(rows, dtype=np.int32) + rank * 10

# Buat array Fortran-ordered (kolom-major) agar sesuai dengan file view
local_buffer = np.asfortranarray(local_data)

# ------------------------------------------------------------
# 4. Hapus file lama (hanya dilakukan oleh rank 0)
# ------------------------------------------------------------
if rank == 0 and os.path.exists(filename):
    os.remove(filename)
comm.Barrier()  # Sinkronisasi: semua proses menunggu sebelum lanjut

# ------------------------------------------------------------
# 5. Buka file paralel secara kolektif
# ------------------------------------------------------------
# MODE_WRONLY : tulis saja
# MODE_CREATE : buat file baru jika belum ada
amode = MPI.MODE_WRONLY | MPI.MODE_CREATE
fh = MPI.File.Open(comm, filename, amode)

# ------------------------------------------------------------
# 6. Definisikan "file view" menggunakan subarray (non-kontigu)
# ------------------------------------------------------------
# Kita ingin setiap proses menulis 1 kolom vertikal dari matriks global
# Dengan kata lain:
#   Proses 0 → kolom ke-0
#   Proses 1 → kolom ke-1
#   Proses 2 → kolom ke-2
#   Proses 3 → kolom ke-3

# Ukuran global matriks (rows x cols)
sizes = (rows, cols)

# Ukuran subarray lokal per proses: 1 kolom penuh (4 baris, 1 kolom)
subsizes = (rows, 1)

# Posisi awal (starting index) kolom per proses
starts = (0, rank)

# Buat tipe data turunan (derived datatype) untuk pola kolom
filetype = MPI.INT.Create_subarray(sizes, subsizes, starts, order=MPI.ORDER_FORTRAN)
filetype.Commit()

# ------------------------------------------------------------
# 7. Tetapkan "view" file untuk tiap proses
# ------------------------------------------------------------
# Dengan Set_view, setiap proses hanya melihat bagian file miliknya
# (yaitu kolom tertentu dari matriks global)
fh.Set_view(0, MPI.INT, filetype)

# ------------------------------------------------------------
# 8. Tulis data secara kolektif (efisien)
# ------------------------------------------------------------
# Write_all → semua proses menulis secara sinkron, MPI akan mengoptimalkan
fh.Write_all(local_buffer)

# ------------------------------------------------------------
# 9. Tutup file dan bebaskan tipe data
# ------------------------------------------------------------
fh.Close()
filetype.Free()

# ------------------------------------------------------------
# 10. Verifikasi hasil (dilakukan hanya oleh Rank 0)
# ------------------------------------------------------------
if rank == 0:
    print(f"\nPenulisan matriks 4x4 selesai ke {filename}.")

    # Membaca kembali file hasil penulisan
    # Perhatikan penggunaan order='F' (Fortran order) karena layout kolom
    data = np.fromfile(filename, dtype=np.int32).reshape(rows, cols, order="F")

    print("Isi matriks hasil (tersusun berdasarkan kolom):")
    print(data)

    # Contoh hasil yang diharapkan:
    # [[ 0 10 20 30]
    #  [ 1 11 21 31]
    #  [ 2 12 22 32]
    #  [ 3 13 23 33]]
