# MNIST ile MLP Tabanlı El Yazısı Rakam Tanıma

Bu projede MNIST veri seti ile **sıfırdan eğitilen bir Multi-Layer Perceptron (MLP)** geliştirilmiş, farklı
hiperparametre/mimari seçimleri karşılaştırılmış ve model **kendi el yazımla yazdığım öğrenci numarası** üzerinde test edilmiştir.
CNN ve hazır eğitilmiş model **kullanılmamıştır** (PyTorch).

- **Öğrenci No:** 24120205046
- **Ad Soyad:** `Volkan Taştemir`

## Klasör Yapısı

```
├── src/
│   ├── common.py               # veri bölme, MLP modeli, eğitim/değerlendirme fonksiyonları
│   ├── exp1_baseline.py        # Deney 1: temel MLP
│   ├── exp2_compare.py         # Deney 2: hiperparametre karşılaştırması
│   ├── prepare_handwriting.py  # el yazısı fotoğrafını MNIST formatına çevirir
│   └── eval_handwriting.py     # modeli el yazısı rakamlarda test eder
├── my_handwriting/
│   ├── student_number_original.jpeg  # kendi el yazımla yazdığım öğrenci numarası
│   ├── digits/                       # 28x28 MNIST formatındaki rakamlar (digit_00.png ...)
│   └── preview.png                   # ön işleme kontrol görseli
├── results/                    # grafikler, metrikler, tablolar (kodlar çalıştırılınca oluşur)
├── requirements.txt
└── README.md
```

## Çalıştırma

```bash
pip install -r requirements.txt
python src/prepare_handwriting.py   # el yazısı -> 28x28 MNIST formatı
python src/exp1_baseline.py         # Deney 1 (temel MLP)
python src/exp2_compare.py          # Deney 2 (karşılaştırma + en iyi model seçimi)
python src/eval_handwriting.py      # kendi el yazımda test
```

MNIST ilk çalıştırmada otomatik indirilir (`data/` klasörü).

## Veri Bölme (Train / Validation / Test)

| Küme | Boyut | Kaynak | Kullanım |
|---|---|---|---|
| Train | 54.000 | MNIST eğitim setinin %90'ı (sabit seed ile rastgele) | ağırlıkların öğrenilmesi |
| Validation | 6.000 | MNIST eğitim setinin %10'u | epoch/model/hiperparametre seçimi |
| Test | 10.000 | resmi MNIST test seti | **sadece** final değerlendirme (geliştirme sürecinde kullanılmadı) |

Deney 2'de tüm konfigürasyonlar yalnızca validation başarımına göre karşılaştırılmış; test seti yalnızca seçilen
nihai model için en sonda bir kez kullanılmıştır.

## Deney 1 – Temel MLP

**Mimari:** `784 → 256 → ReLU → 128 → ReLU → 10` (2 hidden layer)

| Hiperparametre | Değer |
|---|---|
| Loss | CrossEntropyLoss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 128 |
| Epoch | 20 (en yüksek validation accuracy'li epoch'un ağırlıkları kullanılır) |
| Dropout / Weight decay | 0 / 0 |
| Normalizasyon | (x/255 − 0.1307) / 0.3081 |
| Seed | 42 |

**Sonuçlar** (`python src/exp1_baseline.py` çıktısından / `results/exp1_metrics.json`):

| Metrik | Değer |
|---|---|
| Training loss | `0.011690545428923711` |
| Validation loss | `0.094945068359375` |
| Training accuracy | `0.9960185185185185` |
| Validation accuracy | `0.9796666666666667` |
| Test accuracy | `0.9796` |
| Eğitim süresi | `13.190529599998627` sn (cihaz: `AMD Ryzen 7 7840HS w Radeon 780M Graphics 3.80 GHz / NVIDIA GeForce RTX 4050 Laptop GPU`) |

![Deney 1 grafikleri](results/exp1_curves.png)

## Deney 2 – Konfigürasyon Karşılaştırması

Temel modelden başlayarak her seferinde tek bir şey değiştirilmiştir (15 epoch): katman genişliği, derinlik, dropout,
öğrenme oranı, optimizer (SGD+momentum) ve batch size.

| Deney | En iyi epoch | Train loss | Val loss | Train acc | Val acc | Süre (sn) |
|---|---|---|---|---|---|---|
| baseline (256-128, Adam 1e-3) | 15 | 0.0117 | 0.0949 | 0.9960 | 0.9797 | 9.8 |
| dar (64-32) | 15 | 0.0248 | 0.1018 | 0.9917 | 0.9707 | 7.8 |
| geniş (512-256) | 13 | 0.0161 | 0.0907 | 0.9948 | 0.9813 | 15.0 |
| derin (256-128-64) | 11 | 0.0206 | 0.0865 | 0.9926 | 0.9802 | 10.8 |
| dropout 0.3 | 14 | 0.0483 | 0.0768 | 0.9843 | 0.9802 | 12.9 |
| Adam lr=1e-2 (yüksek) | 14 | 0.0922 | 0.2128 | 0.9774 | 0.9655 | 10.1 |
| Adam lr=1e-4 (düşük) | 15 | 0.0528 | 0.0884 | 0.9854 | 0.9723 | 9.9 |
| SGD lr=0.01 momentum=0.9 | 15 | 0.0061 | 0.0724 | 0.9993 | 0.9805 | 8.6 |
| batch 32 | 12 | 0.0202 | 0.1011 | 0.9936 | 0.9800 | 22.5 |
| batch 512 | 15 | 0.0083 | 0.0793 | 0.9979 | 0.9803 | 7.0 |

![Karşılaştırma](results/exp2_comparison.png)

**Seçilen en iyi konfigürasyon:** `<results/exp2_best.json>`  – Test accuracy: `0.9781`

## Kendi El Yazım ile Test

Öğrenci numaram kareli kağıda mavi tükenmez kalemle yazılıp fotoğraflanmıştır (`my_handwriting/student_number_original.jpeg`).
Ön işleme (`src/prepare_handwriting.py`), görüntüleri MNIST'e benzetmek için şu adımları uygular:

1. Gri kareli kağıt çizgilerinden ayırmak için mavi mürekkep maskelenir (mavi − kırmızı kanal farkı).
2. Bağlı bileşenlerle rakamlar ayrılır; iki vuruşlu rakamlar (ör. `4`) birleştirilir.
3. Gri tonlamalı hale getirilir, **siyah arka plan üzerine beyaz rakam** (MNIST formatı) olacak şekilde ayarlanır.
4. Rakam oranı korunarak 20×20 kutuya sığdırılır, ağırlık merkezine göre 28×28 tuvalin ortasına yerleştirilir.
5. [0,1] aralığına ölçeklenir ve MNIST ile **aynı normalizasyon** uygulanır.

![Ön işlenmiş rakamlar](my_handwriting/preview.png)

**Model tahmini** (`python src/eval_handwriting.py`):

| | |
|---|---|
| Gerçek numara | 24120205046 |
| Tahmin | `24120208946` |
| Doğruluk | `9` / 11 |

![Tahminler](results/handwriting_predictions.png)

## Genel Değerlendirme

**Eğitim eğrileri ve overfitting.** Deney 1'in sonunda training accuracy %99,60'a, training loss 0.0117'ye ulaşırken validation accuracy %97,97'de, validation loss ise 0.0949'da kaldı. Validation loss'un train loss'tan yaklaşık sekiz kat yüksek olması ve train/validation accuracy arasında yaklaşık 1,6 puanlık fark bulunması, modelin eğitim verisine validation verisinden belirgin biçimde daha iyi uyduğunu, yani hafif bir overfitting olduğunu gösteriyor. Train accuracy %100'e yaklaşırken validation accuracy'nin %98 civarında doymuş olması da aynı şeye işaret ediyor: eğitim verisini daha fazla öğrenmek artık yeni örneklere genellemeyi iyileştirmiyor. Deney 2'de de aynı örüntü görülüyor; örneğin SGD+momentum train accuracy'yi %99,93'e çıkardı ama validation accuracy %98,05'te kaldı.

**Test ve validation tutarlılığı.** Deney 1'de test accuracy (%97,96) validation accuracy (%97,97) ile neredeyse birebir aynı. Bu beklenen bir durum, çünkü validation ve test kümeleri aynı MNIST dağılımından geliyor ve test seti eğitim sürecinde hiç kullanılmadı; dolayısıyla modelin gerçekten görmediği veride nasıl davrandığını dürüstçe ölçüyor. Deney 2'de seçilen en iyi modelde ise validation %98,13, test %97,81 çıktı. Aradaki küçük düşüş şunu gösteriyor: konfigürasyonu validation üzerinde en yüksek skora göre seçtiğim için validation skoru bir miktar iyimser (seçim yanlılığı). Hatta Deney 1'deki baseline modelin test başarısı (%97,96), "en iyi" seçilen modelden biraz daha yüksek. Bu da konfigürasyonlar arasındaki farkların tesadüf aralığında olduğunu gösteriyor.

**Deney 2 – hangi değişiklik ne etkiledi?** En belirgin etki learning rate'ten geldi. Adam ile lr=1e-2 çok büyük adımlar nedeniyle kararsız kaldı: val loss 0.2128 ile en yüksek, val accuracy %96,55 ile en düşük değeri aldı ve train accuracy bile yalnızca %97,74'te kaldı. lr=1e-4 ise çok yavaş öğrendi; en iyi epoch 15 yani eğitimin son epoch'u, dolayısıyla model hâlâ iyileşiyordu ve yetersiz eğitildi (val accuracy %97,23). İkinci büyük etki model kapasitesinden geldi: çok dar ağ (64-32) %97,07 ile en düşük sonuçlardan birini verdi, yani bu ağ problemi öğrenmek için fazla küçük. Bunun dışında genişlik (512-256: %98,13), derinlik (%98,02), dropout (%98,02), SGD+momentum (%98,05) ve batch size (%98,00–98,03) denemeleri baseline'a (%97,97) çok yakın sonuç verdi; aradaki fark en fazla yaklaşık 0,0016, yani 6000 örnekli validation setinde 10 örnek civarı. Tek seed ile yaptığım bu denemede bu farkların anlamlı olduğunu söyleyemem. Dropout'un train accuracy'yi düşürüp (%98,43) validation loss'u baseline'dan belirgin biçimde azalttığı (0.0768, baseline 0.0949) ise overfitting'e karşı düzenlileştirme etkisinin tutarlı bir göstergesi. SGD+momentum'un val loss'u (0.0724) en düşük çıktı, ancak accuracy'de ayırt edici bir fark yaratmadı.

**Süre ve başarım dengesi.** En küçük model 7,8 sn'de %97,07'ye ulaştı; baseline 9,8 sn'de %97,97'ye çıktı, yani küçük bir süre artışı belirgin kazanç sağladı. Sonrasında getiri hızla azalıyor: en iyi validation sonucunu veren geniş ağ (%98,13) baseline'dan yaklaşık %50 daha uzun sürdü (15,0 sn) ve test'te daha iyi çıkmadı. Batch 32 yaklaşık iki buçuk kat uzun sürdü (22,5 sn) ve aynı doğruluğa ulaştı (%98,00). Batch 512 ise yalnızca 7,0 sn ile %98,03 verdi; bu denemeler arasında süre/başarım açısından en verimli seçenek oldu. Süreler küçük bir modelde ve bu donanımda ölçüldüğü için mutlak değerleri değil, birbirine oranlarını yorumluyorum.

**Kendi el yazım ve distribution shift.** Model MNIST test setinde %97,81–97,96 başarı gösterirken öğrenci numaramda 11 rakamın 9'unu doğru tanıdı (%81,8). Tahmin `24120208946`, gerçek numara `24120205046`; yani hatalar numaranın 8. ve 9. hanesinde: `5` rakamını `8`, ardından gelen `0` rakamını `9` olarak tahmin etti. Bu iki rakam yan yana ve bitişik yazılmıştı; `5`'in alt kıvrımı kapalı bir halkaya benzediği için `8` ile, kuyruğu açık yazdığım `0` ise ters bir "ɔ"/`9` şekline benzediği için `9` ile karışmış olabilir. Hataların ne kadarının benim yazış tarzımdan, ne kadarının ön işleme adımlarından (bitişik rakamların ayrılması, kırpma ve ortalama) kaynaklandığını kesin olarak ayıramıyorum. Doğru tanınan `4`, `1`, `2`, `6` gibi rakamlar ise MNIST'te sık görülen biçimlere yakındı. Buradaki temel neden distribution shift: model, MNIST yazarlarının kalem kalınlığı, eğim, rakam şekli ve ortalanma düzeniyle eğitildi; benim yazım stilim (spiral/açık `0`, iki vuruşlu `4`, bitişik `50`), fotoğrafın ışığı ve kareli kağıt gibi etkenler bu dağılımın dışında kalıyor. Yalnızca 11 örnek olduğu için her hata başarıyı yaklaşık %9 oynatıyor; bu yüzden %81,8 ile %97,8 arasındaki farkı kesin bir istatistik olarak değil, modelin gerçek dünya verisine genellemesinin MNIST test setinden daha zayıf olduğuna dair bir gösterge olarak değerlendiriyorum.

**MLP'nin sınırı ve CNN ile karşılaştırma.** MLP girdi görüntüsünü 784 elemanlı düz bir vektöre çevirdiği için piksellerin komşuluk ilişkisini kullanamaz; her piksel birbirinden bağımsız bir özellik gibi ele alınır. Rakamın birkaç piksel kayması, biraz daha kalın ya da eğik yazılması girdi vektörünü büyük ölçüde değiştirir ve ağın bunu "aynı şekil" olarak tanımasını sağlayacak yapısal bir mekanizma yoktur. Bu yüzden MLP, MNIST'in düzgün ortalanmış yapısına ve benim ön işleme adımlarımın kalitesine güçlü biçimde bağımlı. Bir CNN ise küçük filtreleri görüntünün her yerinde kullanarak (ağırlık paylaşımı) ve pooling uygulayarak çizgi, köşe, halka gibi yerel desenleri konumdan büyük ölçüde bağımsız öğrenir; daha az parametreyle el yazısı çeşitliliğine daha dayanıklıdır. Bu ödevde CNN kullanılmadı, ancak el yazısı testindeki hatalar bu farkın pratikte neden önemli olduğunu gösteriyor.

**Notlar.** Tüm sonuçlar tek seed (42) ve tek çalıştırma ile elde edilmiştir; farklı seed'lerde ondalık düzeyindeki farklar değişebilir. Test seti yalnızca Deney 1'in nihai raporunda ve Deney 2'de seçilen en iyi model için kullanılmıştır.
