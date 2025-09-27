# Lc0 Kaynak Kodunda Native Path-Integral Sampling — Teknik Öneri

## Amaç
Lc0 motorunun C++/CUDA kaynak koduna, path-integral sampling’i sample_size kadar paralel ve native olarak yapacak bir fonksiyon eklenmesi. Böylece, Python tarafında sample_size kadar ayrı motor çağrısı yerine, tek seferde toplu sampling alınabilir.

## Teknik Akış
1. **Yeni API Parametresi:**
   - `--path_integral_sample_size N` (veya UCI opsiyonu)
   - `--lambda` (softmax sıcaklığı)
2. **Root Node Sampling:**
   - Root node’da, policy head ve value head ile, N adet olasılıksal yol örneklemesi yapılır.
   - Her örnek, softmax(λ) ile ağırlıklandırılır ve MCTS yolunu takip eder.
3. **Sampling Fonksiyonu:**
   - C++ tarafında, root node’da N adet yol için random seed ile sampling yapılır.
   - Her yolun hamle dizisi ve centipawn değeri kaydedilir.
   - Sonuçlar JSON veya CSV olarak dışarıya aktarılır.
4. **Python’dan Çağrı:**
   - Tek bir motor başlatılır, pozisyon yüklenir, `sample_size` ve `lambda` parametreleri ile sampling istenir.
   - Sonuçlar doğrudan toplu olarak alınır.

## Akış Diyagramı
```
[Python] → [Lc0 başlat] → [Pozisyon yükle] → [Path-integral sampling fonksiyonu]
    ↓
[N adet yol örneklemesi, softmax(λ) ile]
    ↓
[Hamle dizileri + centipawn değerleri]
    ↓
[JSON/CSV ile Python’a geri]
```

## Kodda Değişecek Yerler
- UCI/CLI parametreleri: Yeni sampling opsiyonu
- Root node sampling: C++ fonksiyonu
- Output: JSON/CSV formatı
- (Opsiyonel) CUDA kernel optimizasyonu

## Python’dan Kullanım Örneği
```python
import subprocess
subprocess.run([
    r"lc0.exe", "--path_integral_sample_size=100", "--lambda=0.1", "--weights=...", "--position=..."
])
# Sonuçları JSON/CSV’den oku
```

## Avantajlar
- Zaman ve kaynak verimliliği (tek motor, toplu sampling)
- Tutarlı random seed ve state yönetimi
- GPU üzerinde paralel sampling
- Daha az IO ve başlatma maliyeti

## Zorluklar
- Lc0 kaynak koduna müdahale gerektirir (C++/CUDA)
- Sampling fonksiyonunun MCTS ve policy head ile uyumlu olması gerekir
- API ve output formatı için ek geliştirme

---
Bu öneri, path-integral sampling’in Lc0 motorunda native ve verimli şekilde yapılmasını sağlar. Detaylı teknik dokümantasyon ve kod örnekleri için Lc0’nın resmi kaynak koduna bakılmalı ve toplulukla tartışılmalıdır.
