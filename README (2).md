# Reproduksi Notebook: House Prices Prediction using TFDF

Dokumen ini menjelaskan cara menjalankan ulang notebook Kaggle
*House Prices Prediction using TFDF* (Random Forest dengan TensorFlow Decision
Forests) di Google Colab, beserta perbaikan agar hasilnya dapat diulang.

## 1. Ringkasan
- **Tujuan:** memprediksi harga jual rumah (`SalePrice`) dari 79 fitur.
- **Data:** Kaggle *House Prices - Advanced Regression Techniques*
  (`train.csv`, `test.csv`).
- **Model:** Random Forest (TFDF), tugas regresi.
- **Metrik:** RMSE out-of-bag dan MSE pada data validasi.

## 2. Versi yang dipakai
| Komponen | Notebook asli | Reproduksi (Colab) |
|---|---|---|
| Python | tidak dicatat | 3.12.13 |
| TensorFlow | 2.11.0 | 2.19.0 |
| TFDF | 1.2.0 | 1.12.0 |

> TFDF 1.12.0 belum tersedia untuk Python 3.13. Jika runtime Colab memakai
> Python 3.13, ubah runtime ke Python 3.12 terlebih dahulu.

## 3. Cara menjalankan di Google Colab
1. Unduh `train.csv` dan `test.csv` dari halaman kompetisi Kaggle
   (perlu akun Kaggle), lalu unggah ke Colab.
2. Pasang pustaka:
   ```
   !pip install tensorflow==2.19.0 tensorflow_decision_forests==1.12.0
   ```
3. Ganti path data pada notebook:
   ```python
   train_file_path = "train.csv"          # asli: ../input/house-prices-.../train.csv
   test_file_path  = "test.csv"
   ```
4. Ganti path penyimpanan hasil:
   ```python
   sample_submission_df.to_csv("submission.csv", index=False)   # asli: /kaggle/working/submission.csv
   ```
5. **Tetapkan seed** sebelum membagi data (perbaikan utama):
   ```python
   import numpy as np
   SEED = 42
   np.random.seed(SEED)

   rf = tfdf.keras.RandomForestModel(
       task=tfdf.keras.Task.REGRESSION,
       random_seed=SEED,
       num_trees=300,          # tulis hyperparameter secara eksplisit
   )
   ```
6. Jalankan seluruh sel dari atas ke bawah.

## 4. Perbaikan terhadap notebook asli
| Masalah | Perbaikan |
|---|---|
| Split acak tanpa seed (jumlah data berubah tiap run) | `np.random.seed(SEED)` sebelum split |
| Seed dan hyperparameter model tidak dicatat | `random_seed=SEED`, `num_trees` ditulis eksplisit |
| Path `/kaggle/input` dan `/kaggle/working` | Path lokal Colab |
| Versi pustaka tidak didokumentasikan | Tabel versi pada bagian 2 |

## 5. Hasil yang diharapkan
Dengan seed yang sama, dua kali eksekusi memberi jumlah data latih/validasi dan
metrik yang sama. Hasil audit sebelumnya (tanpa seed) sebagai pembanding:

| | Colab | Kaggle asli |
|---|---|---|
| Data latih / validasi | 1.018 / 442 | 1.016 / 444 |
| RMSE out-of-bag | 27.352,65 | 27.943,25 |
| MSE validasi | 1.019.549.056 | 1.075.197.952 |

Selisih beberapa persen antar lingkungan atau antar seed masih wajar.

## 6. Keterbatasan
- Hasil hanya sebanding pada versi pustaka yang sama.
- TFDF sudah dalam mode pemeliharaan; penggantinya adalah YDF.
- Notebook melaporkan RMSE pada skala harga asli, sedangkan metrik resmi
  kompetisi Kaggle adalah RMSE pada log harga.

## 7. Sumber
- Notebook asli: *House Prices Prediction using TFDF* (Kaggle) https://www.kaggle.com/code/gusthema/house-prices-prediction-using-tfdf/notebook 
- Dataset: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
