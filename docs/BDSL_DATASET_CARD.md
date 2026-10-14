# Dataset Card: BDSLW131 (Bangla Sign Language Word Dataset)

## 1. Dataset Summary
- **Dataset Name**: BDSLW131 (Bangla Sign Language Word Dataset)
- **Publication Source**: [Kaggle Dataset by Musaddique Ali Erfan](https://www.kaggle.com/datasets/musaddiquealierfan/bdslw131-bangla-sign-language-word-dataset)
- **License**: Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))
- **Language**: Bengali (বাংলা) / BdSL
- **Total Samples**: 29,930 still images
- **Total Classes**: 131 unique words/concepts
- **Class Formats**: Bilingual folder names `<Bengali_Unicode>_<English_Gloss>`

---

## 2. Dataset Structure & Modality
- **Modality**: Still RGB images (JPG format)
- **Capture Setup**: Captured via smartphone camera sensors across natural indoor environments with diverse backgrounds and lighting conditions.
- **Directory Layout**:
  ```
  data/raw/bangla-sign-dataset/
  ├── metadata.csv                   # 29,930 filepath-to-label records
  ├── frequency.csv                  # 131 class frequency counts
  ├── README.md                      # Official dataset documentation
  ├── অপেক্ষা করো_Wait/              # 230 images
  ├── আমি_I (Me)/                    # 230 images
  ├── ...                            # 127 intermediate word classes
  ├── শূন্য_Zero/                    # 140 images
  └── হৃদয়_Heart/                    # 230 images
  ```
- **Local Location**: `data/raw/bangla-sign-dataset/` (aliased via Windows directory junction `data/raw/bdslw131/` for configuration portability).

---

## 3. Class Distribution & Balance Statistics

| Metric | Measured Value |
|---|---|
| **Total Classes** | 131 |
| **Total Images Discovered & Validated** | 29,930 |
| **Excluded / Corrupt Images** | 0 |
| **Mean Images per Class** | 228.47 |
| **Median Images per Class** | 230.0 |
| **Min Images in Class** | 140 (`শূন্য_Zero`, Class ID: 119) |
| **Max Images in Class** | 230 (119 classes have exactly 230 images) |
| **Total Dataset Storage Size** | ~134.9 GB |

### Vocabulary Categorization
1. **Numbers (১০টি)**: ০ (Zero) to ৯ (Nine)
2. **Pronouns & Relations**: আমি (I), তুমি (You), সে (He/She), মা (Mother), বাবা (Father), ভাই (Brother), বোন (Sister), চাচা/মামা (Uncle), দাদী/নানী (Grandmother)
3. **Common Actions**: খাওয়া (Eating), ঘুমানো (Sleeping), দাঁড়াও (Stand), থামো (Stop), সাহায্য (Help)
4. **Days & Time**: রবিবার (Sunday) to শনিবার (Saturday), আজ (Today), এখন (Now), সময় (Time)
5. **Body Parts**: হাত (Hand), চোখ (Eye), কান (Ear), নাক (Nose), মাথা (Head), বুক (Chest), পা (Leg), জিহ্বা (Tongue), দাঁত (Teeth)
6. **Abstract & Modern Concepts**: ভালোবাসা (Love), প্রজ্ঞা (Wisdom), ইন্টারনেট/ওয়াই-ফাই (Wi-Fi), ফোন (Phone), ইনস্টাগ্রাম (Instagram), গণিত (Mathematics)

---

## 4. Academic Integrity & Limitation Statements

### Modality Limitation
- **Static vs. Dynamic Signs**: BDSLW131 consists strictly of **still photograph images**, representing single-frame poses/gestures for words. It does **not** contain temporal video sequences or multi-frame video clips.
- **Scientific Claim**: We evaluate static landmark feature classification models on this dataset. We do **not** claim to recognize continuous dynamic signing videos using training on BDSLW131 alone. Real-time webcam inference in SignFlow uses MediaPipe holistic tracking with stability debouncing to classify the user's holding gesture into one of the 131 vocabulary words.

### Signer Demographics & Splitting Methodology
- **Signer Identifiers**: The official dataset publishers do not provide per-image signer IDs or demographic metadata in `metadata.csv`.
- **Generalization Claim**: Because individual signers cannot be mathematically separated across splits without signer IDs, true signer-independent generalization cannot be measured on BDSLW131. Stratified class splitting (70% train, 15% validation, 15% test) is enforced with deterministic seeds to prevent sample leakage across splits.
