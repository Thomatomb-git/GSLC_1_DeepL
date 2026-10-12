Nama : Frederick Kamsono <br>
NIM : 2802549205

1. **Efek Regularisasi pada Batch Normalization**

    Kita tau bahwa Batch Normalization (BN) menghitung rata-rata mini-batch $\mu_B$ dan variansi $\sigma_B^2$ secara independen pada setiap iterasi pelatihan:

    $$\mu_B = \frac{1}{m} \sum_{i=1}^m x_i, \quad \sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2$$

    Karena mini-batch merupakan sampel acak terbatas dari keseluruhan populasi data latih, statistik mean dan variansi mengandung derau stokastik. Saat aktivasi dinormalisasi ($\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$), derau ini bertindak sebagai regularisasi inheren yang mencegah coadaptation berlebih antar-fitur, serupa dengan efek penambahan derau acak pada aktivasi.

    Asumsi ini gugur ketika ukuran batch terlampau kecil ($m < 8$), di mana estimasi variansi menjadi sangat tidak stabil dan memicu degradasi konvergensi. Selain itu, penggabungan BN dan Dropout secara bertumpukan sering memicu fenomena variance shift saat inferensi, karena pemutusan unit neuron acak oleh Dropout mengubah ekspektasi variansi yang tidak selaras dengan akumulasi moving statistics pada BN.

2. **Degradasi Jaringan Dalam dan Formulasi Residual**

    Penurunan akurasi latih pada plain CNN yang sangat dalam bukan disebabkan oleh overfitting, melainkan kesulitan optimasi akibat permukaan fungsi objektif yang menjadi sangat non-konveks dan bergelombang. Meskipun secara teoritis jaringan yang lebih dalam mampu mendekati fungsi identitas, optimasi berbasis turunan orde pertama secara empiris gagal menemukan pemetaan identitas melalui tumpukan fungsi aktivasi non-linear non-konveks.

    ResNet merumuskan blok residual sebagai $\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)$. Berdasarkan chain rule, aliran gradien backpropagation terhadap aktivasi $\mathbf{x}_l$ dirumuskan sebagai:

    $$\frac{\partial \mathcal{E}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{E}}{\partial \mathbf{x}_{l+1}} \left( \mathbf{I} + \frac{\partial \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)}{\partial \mathbf{x}_l} \right)$$

    Hubungkan mekanisme tersebut dengan mitigasi fenomena vanishing gradient yakni matriks identitas $\mathbf{I}$ menyediakan jalur pintas tanpa hambatan (gradient highway). Gradien dari lapisan atas $\frac{\partial \mathcal{E}}{\partial \mathbf{x}_{l+1}}$ dapat merambat langsung ke lapisan awal tanpa harus melewati perkalian berantai dengan bobot $\vert{}\mathcal{W}\vert{} < 1$, sehingga mencegah degradasi sinyal gradien menuju nol (vanishing gradient).

3. **Feature Extraction vs. Fine-Tuning**

    - Trainable Parameters: Feature Extraction hanya melatih lapisan kepala klasifikasi akhir ($< 1\%$ total bobot model). Fine-Tuning memperbarui lapisan akhir klasifikasi bersama sebagian atau seluruh blok konvolusi backbone ($10\% - 100\%$ total bobot).

    - Kebutuhan Volume Data Latih: Feature Extraction sangat efektif pada rezim data sangat terbatas ($< 200$ sampel per kelas). Fine-Tuning membutuhkan sampel moderat hingga besar agar bobot backbone tidak mengalami overfitting.

    - Beban Komputasi dan Memori: Feature Extraction menghemat VRAM dan waktu pelatihan karena backbone tidak memerlukan kalkulasi propagasi balik. Fine-Tuning membutuhkan komputasi gradien penuh dan memori aktivasi yang signifikan.

    - Risiko Rusaknya Fitur Bawaan: Nol pada Feature Extraction karena representasi dibekukan (frozen). Sangat tinggi pada Fine-Tuning jika learning rate terlalu agresif.

    Gunakan Feature Extraction jika ukuran dataset sangat kecil dan domain data sangat mirip dengan ImageNet. Terapkan Progressive Fine-Tuning jika dataset cukup memadai dan domain data memiliki keunikan visual spesifik.

4. **Kaidah Penerapan Data Augmentation**

    Augmentasi data dirancang sebagai regularisasi stokastik guna memperluas distribusi ruang sampel data latih dan menekan generalization gap. Sebaliknya, himpunan validasi dan uji berfungsi sebagai tolok ukur objektif performa model pada distribusi empiris dunia nyata. Menerapkan transformasi stokastik pada data validasi/uji akan memasukkan noise evaluasi buatan (measurement noise), merusak determinisme metrik, serta menghasilkan estimasi performa yang bias dan tidak konsisten.

    Penerapan distorsi warna ekstrem (Extreme Color Jittering / Solarization) atau konversi skala abu-abu (Grayscale) tidak layak diterapkan. Pola pigmen dan nuansa warna bulu—seperti corak titik pada kucing Siamese, pola belang Calico, atau rona keemasan Golden Retriever—merupakan atribut semantik pembeda utama antarras. Mengubah atau menghilangkan informasi warna merusak invariansi semantik (semantic label invariance), sehingga model kehilangan fitur diskriminatif yang krusial untuk klasifikasi.

5. **Mitigasi Catastrophic Forgetting**

    Fenomena ini terjadi ketika backbone pretrained menerima pembaruan gradien bernilai besar dari classifier head yang diinisialisasi secara acak pada awal adaptasi. Sinyal gradien yang belum terkalibrasi ini merusak representasi visual umum (seperti detektor tepi dan tekstur) yang telah dipelajari dari ImageNet, menimpa ruang parameter berharga dengan pola spesifik data baru secara destruktif.

    Dua Strategi Mitigasi Konkret:

    - Differential Learning Rate: Menetapkan laju pembelajaran bertingkat, dengan $\eta_{\text{backbone}}$ jauh lebih kecil (misal $10^{-5}$ atau $10^{-4}$) dibandingkan $\eta_{\text{fc}}$ (misal $10^{-3}$). Secara teoritis, ini menjaga kestabilan manifold representasi tingkat rendah sekaligus memberi fleksibilitas bagi lapisan atas untuk beradaptasi (Howard & Ruder, 2018).

    - Gradual Unfreezing: Melatih classifier head terlebih dahulu hingga mencapai konvergensi awal, baru kemudian membuka pembekuan blok residual teratas (misalnya layer4) secara bertahap. Hal ini memastikan gradien yang masuk ke backbone sudah stabil dan terarah, mencegah lonjakan gradien acak.
