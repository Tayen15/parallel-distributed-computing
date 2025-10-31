# =====================================================
# Workshop MPI - Operasi Kolektif (Versi Diperbaiki)
# Topik: Simulasi Pengumpulan Data Sensor Cuaca Terdistribusi
# Jalankan dengan: mpirun -np 4 python mpi_workshop.py
# =====================================================

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# -----------------------------------------------------
# 1. Broadcast: Root mendistribusikan ambang batas suhu
# -----------------------------------------------------
if rank == 0:
    threshold = 30.0  # suhu batas panas
    print(f"Root Menyiapkan ambang batas suhu: {threshold}C")
else:
    threshold = None

threshold = comm.bcast(threshold, root=0)
print(f"[Proses {rank}] Menerima ambang batas suhu = {threshold}C")

# -----------------------------------------------------
# 2. Scatter: Root membagikan data suhu ke tiap node
# -----------------------------------------------------
if rank == 0:
    # total jumlah data harus = size × n_data_per_proses
    n_data_per_proc = 2  # misal tiap proses dapat 2 data sensor
    total_data = n_data_per_proc * size
    sensor_data = np.linspace(25.0, 34.0, total_data, dtype=np.float64)
    print(f"[Root] Menyiapkan {total_data} data sensor untuk dibagikan: {sensor_data}")
else:
    n_data_per_proc = None
    sensor_data = None

# broadcast n_data_per_proc supaya semua tahu ukuran yang diterima
n_data_per_proc = comm.bcast(n_data_per_proc, root=0)

# buffer penerima di tiap proses
recvbuf = np.zeros(n_data_per_proc, dtype=np.float64)

# distribusi data secara kolektif
comm.Scatter([sensor_data, MPI.DOUBLE], [recvbuf, MPI.DOUBLE], root=0)
print(f"[Proses {rank}] Menerima data suhu lokal: {np.round(recvbuf, 2)}°C")

# -----------------------------------------------------
# 3. Analisis lokal: deteksi suhu tinggi
# -----------------------------------------------------
local_flags = recvbuf > threshold
local_status = ["TINGGI" if val else "NORMAL" for val in local_flags]
print(f"[Proses {rank}] Status suhu lokal: {local_status}")

# -----------------------------------------------------
# 4. Gather: Kumpulkan hasil ke root
# -----------------------------------------------------
# tiap proses mengirim array boolean (1 = tinggi, 0 = normal)
send_int = local_flags.astype(np.int32)
recv_flags = None
if rank == 0:
    recv_flags = np.empty(n_data_per_proc * size, dtype=np.int32)

comm.Gather(send_int, recv_flags, root=0)

if rank == 0:
    print("\n[Root] Hasil deteksi suhu tinggi dari semua sensor:")
    for i, val in enumerate(recv_flags):
        print(f"  Sensor {i:02d}: {'TINGGI' if val else 'NORMAL'}")

# -----------------------------------------------------
# 5. Reduce: Hitung rata-rata suhu global
# -----------------------------------------------------
local_avg = np.mean(recvbuf)
global_sum = comm.reduce(local_avg, op=MPI.SUM, root=0)

if rank == 0:
    global_avg = global_sum / size
    print(f"\n[Root] Rata-rata suhu global = {global_avg:.2f}°C")

# -----------------------------------------------------
# 6. Allgather: Bagikan rata-rata suhu lokal ke semua node
# -----------------------------------------------------
all_local_avg = comm.allgather(local_avg)
print(
    f"[Proses {rank}] Mengetahui rata-rata suhu tiap node: {np.round(all_local_avg, 2)}"
)

# -----------------------------------------------------
# 7. Akhiri program
# -----------------------------------------------------
if rank == 0:
    print("\nSimulasi selesai")
