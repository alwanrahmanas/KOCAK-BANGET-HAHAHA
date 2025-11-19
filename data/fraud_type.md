# 📘 Jenis-Jenis Fraud dalam Program Jaminan Kesehatan (BPJS)

Dokumen ini merangkum **tipe-tipe kecurangan (fraud)** yang diuraikan dalam:

- Permenkes No. 16 Tahun 2019 tentang Pencegahan dan Penanganan Kecurangan  
- Peraturan BPJS Kesehatan No. 6 Tahun 2020  
- Pedoman Anti-Fraud JKN  
- SPO Audit Klaim BPJS  

Tujuan ringkasan ini adalah menyediakan **ground truth legal-based** untuk pengembangan model *AI fraud detection* dan sistem *rule-based compliance*.

---

## 🏥 Fraud oleh Fasilitas Kesehatan / Pemberi Pelayanan

| Kategori Fraud | Deskripsi | Contoh | Level Pelanggaran |
|----------------|------------|---------|-------------------|
| **Upcoding Diagnosis** | Mengubah kode diagnosis agar tarif klaim lebih tinggi dari seharusnya. | Pasien apendisitis tanpa komplikasi diklaim sebagai apendisitis perforasi. | Sedang – Berat |
| **Phantom Billing (Klaim Fiktif)** | Mengajukan klaim atas layanan yang tidak pernah dilakukan. | Klaim tindakan operasi yang tidak pernah dilakukan. | Berat |
| **Cloning Claim** | Menyalin rekam medis atau data pasien lain untuk klaim baru. | Copy-paste data pasien lain ke klaim berbeda. | Berat |
| **Inflated Bill** | Menggelembungkan tagihan obat atau alat kesehatan. | Tagihan 4 vial obat padahal hanya diberikan 2. | Sedang |
| **Service Unbundling** | Memecah paket pelayanan menjadi beberapa klaim terpisah untuk menaikkan total pembayaran. | Pemeriksaan laboratorium yang seharusnya satu paket dibagi menjadi 4 klaim. | Sedang |
| **Self-Referral** | Merujuk pasien ke fasilitas milik sendiri tanpa alasan medis. | Dokter merujuk pasien ke RS tempat ia juga bekerja. | Sedang |
| **Repeat Billing** | Menagih kembali klaim yang sudah dibayar. | Klaim rawat inap diajukan dua kali. | Berat |
| **Prolonged Length of Stay** | Memperpanjang lama rawat inap tanpa indikasi medis. | Pasien ventilator dipertahankan tanpa indikasi. | Sedang |
| **Manipulation of Room Charge** | Memanipulasi kelas perawatan untuk menaikkan biaya klaim. | Klaim kelas I padahal pasien dirawat di kelas II. | Sedang |
| **Unnecessary Services** | Memberikan layanan medis yang tidak sesuai indikasi. | Pemeriksaan berlebihan agar tagihan naik. | Ringan – Sedang |
| **Fake License / Illegal Practice** | Menggunakan izin praktik atau izin operasional palsu. | Dokter tanpa SIP aktif tetap mencatat layanan. | Berat |

---

## 👤 Fraud oleh Peserta (Pasien)

| Kategori Fraud | Deskripsi | Contoh | Level Pelanggaran |
|----------------|------------|---------|-------------------|
| **False Identity** | Memalsukan data kepesertaan untuk memperoleh layanan. | Menggunakan NIK fiktif. | Berat |
| **Card Misuse** | Meminjamkan, menjual, atau memperdagangkan kartu JKN-KIS. | Peserta meminjamkan kartu ke saudara. | Ringan – Sedang |
| **Unnecessary Claim** | Memanfaatkan hak layanan tanpa indikasi medis. | Meminta rujukan ke FKRTL tanpa alasan medis. | Ringan |
| **False Information** | Memberikan informasi palsu saat diagnosis atau pemeriksaan. | Mengaku belum pernah berobat untuk klaim ulang. | Sedang |
| **Bribery / Gratifikasi** | Memberi imbalan untuk mempercepat atau menambah layanan. | Menyuap petugas agar diberi obat lebih. | Berat |
| **Reselling Medicines** | Menjual kembali obat/alat kesehatan yang diperoleh dari JKN. | Menjual obat kronis ke pihak lain. | Sedang |

---

## 🏢 Fraud oleh BPJS Kesehatan (Internal)

| Kategori Fraud | Deskripsi | Contoh | Level |
|----------------|------------|---------|-------|
| **Collusion with Provider** | Bekerjasama dengan Faskes untuk menyetujui klaim tidak sah. | Verifikator meloloskan klaim palsu. | Berat |
| **Manipulation of Benefit** | Menyetujui manfaat di luar ketentuan. | Menyetujui tindakan non-covered. | Berat |
| **Conflict of Interest** | Pengambilan keputusan dipengaruhi kepentingan pribadi. | Petugas memperlambat pembayaran untuk meminta imbalan. | Sedang |
| **Credentialing Fraud** | Manipulasi hasil uji kelayakan Faskes. | RS tidak layak tetap disetujui kontraknya. | Sedang |
| **Misuse of Funds** | Menggunakan dana JKN untuk kepentingan pribadi. | Penarikan dana kapitasi tanpa dasar. | Berat |

---

## 💊 Fraud oleh Penyedia Obat & Alat Kesehatan

| Kategori | Deskripsi | Contoh | Level |
|-----------|------------|---------|-------|
| **Refusal / Delay without Reason** | Menolak atau menunda pengiriman tanpa alasan jelas. | Supplier menunda stok obat. | Ringan |
| **Kickback / Suap** | Memberi imbalan agar produk dipakai Faskes tertentu. | Vendor memberi komisi pada dokter. | Sedang |
| **Inflated Price** | Menagih harga lebih tinggi dari sebenarnya. | Klaim 10 unit alat kesehatan padahal hanya 8. | Sedang |
| **False Product / Data Manipulation** | Mengubah hasil uji atau dokumen kualitas produk. | Memalsukan laporan sertifikasi alat. | Berat |

---

## 👔 Fraud oleh Pemberi Kerja / Pemangku Kepentingan Lain

| Kategori | Deskripsi | Contoh | Level |
|-----------|------------|---------|-------|
| **Manipulasi Data Pegawai** | Tidak mendaftarkan pegawai atau menurunkan upah dilaporkan. | Melapor upah separuh dari sebenarnya. | Sedang |
| **Tidak Menyetorkan Iuran** | Menahan iuran yang sudah dipotong dari karyawan. | Iuran ditahan untuk keperluan perusahaan. | Berat |
| **Bribery / Collusion** | Suap untuk memanipulasi data PBI. | Memberi uang agar pegawai non-eligible jadi PBI. | Berat |

---

## ⚖️ Jenis Sanksi Administratif (Permenkes 16/2019 Pasal 6-8)

| Kategori Pelanggaran | Jenis Sanksi | Contoh |
|-----------------------|--------------|---------|
| **Ringan** | Teguran lisan / tertulis | Keterlambatan minor, kesalahan input klaim |
| **Sedang** | Teguran tertulis + pengembalian dana / denda 25% | Upcoding ringan, unbundling kecil |
| **Berat** | Pengembalian dana + denda 50% + pencabutan izin | Phantom billing, collusion, cloning |

---

## 📚 Referensi Utama
1. **Permenkes No. 16 Tahun 2019** – Bab II Jenis Kecurangan & Bab V Sanksi  
2. **Peraturan BPJS Kesehatan No. 6 Tahun 2020** – Sistem Pencegahan Kecurangan  
3. **Pedoman Anti-Fraud JKN** – Jenis Fraud dan Mekanisme Pencegahan  
4. **SPO Audit Klaim BPJS** – Prosedur Audit dan Validasi Klaim  

---

> **Catatan:**  
> Tabel ini dapat dijadikan _label taxonomy_ untuk pengembangan *Fraud Knowledge Base* dalam proyek AI, serta dasar pembuatan dataset sintetis untuk pelatihan model deteksi fraud berbasis regulasi.
