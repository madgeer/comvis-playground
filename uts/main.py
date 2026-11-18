import cv2
import numpy as np
import time

def process_frame(frame):
    # konversi video ke grayscale supaya lebih mudah untuk mendeteksi fitur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ---------------------------------------------------------
    # 1. HARRIS CORNER DETECTION
    # ---------------------------------------------------------
    start_time = time.time()
    
    # Konversi ke float32 untuk Harris Corner
    gray_float = np.float32(gray)
    
    # cornerHarris(image, blockSize, ksize, k)
    dst = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None) # Memperjelas titik sudut
    
    # Threshold untuk menentukan titik sudut
    harris_img = frame.copy()
    harris_mask = dst > 0.01 * dst.max()
    harris_img[harris_mask] = [0, 0, 255] # Tandai titik Harris dengan warna merah
    
    harris_count = np.sum(harris_mask)
    harris_time = (time.time() - start_time) * 1000 # Dalam milidetik (ms)
    
    # ---------------------------------------------------------
    # 2. SIFT (Scale-Invariant Feature Transform)
    # ---------------------------------------------------------
    start_time = time.time()
    
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(gray, None)
    
    # Gambar keypoints SIFT
    sift_img = cv2.drawKeypoints(
        frame, keypoints, None, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    sift_count = len(keypoints)
    sift_time = (time.time() - start_time) * 1000 # Dalam milidetik (ms)
    
    # ---------------------------------------------------------
    # 3. MENAMPILKAN INFORMASI & MENGGABUNGKAN FRAME
    # ---------------------------------------------------------
    # Teks Overlay Harris
    cv2.putText(harris_img, f"Harris Corner", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(harris_img, f"Fitur: {harris_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(harris_img, f"Waktu: {harris_time:.2f} ms", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Teks Overlay SIFT
    cv2.putText(sift_img, f"SIFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(sift_img, f"Fitur: {sift_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(sift_img, f"Waktu: {sift_time:.2f} ms", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Gabungkan dua gambar secara berdampingan (side-by-side)
    combined_view = np.hstack((harris_img, sift_img))
    return combined_view


def main():
    video_path = "input.mp4"  # File video input
    output_path = "output.mp4"  # Nama file video hasil simpanan
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Video '{video_path}' tidak ditemukan.")
        return

    # Ambil nilai FPS dari video asli
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0  # Default jika FPS tidak terbaca

    # Ukuran resolusi video gabungan (dua frame 640x360 digabung secara horizontal jadi 1280x360)
    frame_width = 640 * 2
    frame_height = 360

    # Inisialisasi VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print("Memproses video dan menyimpan output...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
            
        # Resize frame agar muat di layar saat digabung
        frame_resized = cv2.resize(frame, (640, 360))
        result = process_frame(frame_resized)
        
        # Simpan frame hasil ke video output
        out.write(result)
        
        # Tampilkan di layar
        cv2.imshow("Komparasi Harris Corner vs SIFT", result)
        
        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release semua resource
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Selesai! Video output berhasil disimpan ke: '{output_path}'")


if __name__ == "__main__":
    main()