# Medical Prescription Analyzer (Vaidya AI)

A deep-learning pipeline that reads prescription images and extracts the medical information in them.
It combines OCR with a Donut (Document Understanding Transformer) model and biomedical named entity recognition (NER).

---

## Problem

Prescriptions, especially handwritten ones, are hard to read and slow to digitize.
Misreading a medicine name or dosage is a patient safety risk.
This project explores automating the first step: turning a prescription image into text and tagged medical entities.

---

## How It Works

```
Prescription image
        |
        v
Donut OCR (vision encoder-decoder, no separate text detection step)
        |
        v
Raw text
        |
        v
Biomedical NER (d4data/biomedical-ner-all)
        |
        v
Tagged entities (medicines, dosages, and similar)
```

1. **OCR with Donut.**
   Donut reads the image directly and generates text, so no separate text detection step is needed.
2. **Biomedical NER.**
   The OCR text is passed to a token classification model that tags medical entities.
3. **Evaluation.**
   The pipeline notebook compares the OCR output with a ground-truth text using `difflib`.
   The fine-tuning notebook tracks training loss, validation loss and WER (word error rate).

---

## Repository Contents

| File | Purpose |
|------|---------|
| [Fine_Tuning_Model_Gamma.ipynb](Fine_Tuning_Model_Gamma.ipynb) | Fine-tunes `naver-clova-ix/donut-base` on the prescription dataset |
| [Vaidya_AI_alpha.ipynb](Vaidya_AI_alpha.ipynb) | End-to-end pipeline: Donut OCR, then biomedical NER, then evaluation |
| [dataSet.txt](dataSet.txt) | Links to the Kaggle datasets |
| [requirements.txt](requirements.txt) | Python dependencies |
| [LICENSE](LICENSE) | MIT License |

---

## Datasets

Both datasets are hosted on Kaggle (see [dataSet.txt](dataSet.txt)):

- [prescription-data-set](https://www.kaggle.com/datasets/soham3ripathy/prescription-data-set) - prescription images
- [ground-truth-seed](https://www.kaggle.com/datasets/soham3ripathy/ground-truth-seed) - ground-truth text for the images

---

## Setup

The notebooks are designed for Google Colab or Kaggle with a GPU runtime.
Both open directly in Colab through the badge at the top of each notebook.

To run locally instead:

```bash
git clone https://github.com/SOHAM-3T/Medical-Prescription-Analyzer-.git
cd Medical-Prescription-Analyzer-
pip install -r requirements.txt
```

The notebooks download the data with `kagglehub`, so you need to log in to Kaggle when prompted.
Some data sources are private, so you may need access from the dataset owner.

---

## Usage

### 1. Fine-tune Donut

Open [Fine_Tuning_Model_Gamma.ipynb](Fine_Tuning_Model_Gamma.ipynb) and run the cells in order.
Update `IMAGE_DIR` and `OUTPUT_MODEL_DIR` in the configuration cell to match your environment.

Training setup:

- Base model: `naver-clova-ix/donut-base`
- Epochs: 10
- Batch size: 1 (train and eval)
- Max sequence length: 384
- Metric: WER

### 2. Run the pipeline

Open [Vaidya_AI_alpha.ipynb](Vaidya_AI_alpha.ipynb) and set `TEST_IMAGE_PATH` and the ground-truth text in the last cell.
Run all cells to see the OCR output, the NER tags and the comparison score.

Note: the alpha pipeline currently loads `naver-clova-ix/donut-base-finetuned-cord-v2` for OCR.
To use your own fine-tuned model, point it at your `donut-finetuned-prescription-ocr` output directory:

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

MODEL_DIR = "path/to/donut-finetuned-prescription-ocr"
processor = DonutProcessor.from_pretrained(MODEL_DIR)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
```

---

## Results

| Metric | Value (10 epochs) |
|--------|-------------------|
| Training loss | ~4.4 |
| Validation loss | ~4.4 |
| WER | ~0.87 to 1.15 (unstable) |

The dataset is small, so these results are a proof of concept and not production quality.
More data and more training are expected to improve them significantly.

---

## Future Work

- [ ] Expand the dataset for better generalization
- [ ] Use the fine-tuned model inside the pipeline notebook
- [ ] Output structured JSON (medicine, dosage, duration)
- [ ] Deploy as an API (Flask or FastAPI)
- [ ] Support multiple languages

---

## Contributing

Pull requests are welcome.
For major changes, please open an issue first to discuss what you would like to change.

1. Fork the project
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m "feat: add my feature"`)
4. Push the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/SOHAM-3T">
        <img src="https://github.com/SOHAM-3T.png" width="100px;" alt="SOHAM-3T"/>
        <br />
        <sub><b>Soham Tripathy</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Ajay-Kumar-Prasad">
        <img src="https://github.com/Ajay-Kumar-Prasad.png" width="100px;" alt="Ajay-Kumar-Prasad"/>
        <br />
        <sub><b>Ajay Kumar Prasad</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/SaiPrithvi1278">
        <img src="https://github.com/SaiPrithvi1278.png" width="100px;" alt="SaiPrithvi1278"/>
        <br />
        <sub><b>Sai Prithvi</b></sub>
      </a>
    </td>
  </tr>
</table>

---

## Acknowledgments

- [Naver Clova IX](https://github.com/clovaai/donut) for the original Donut model
- Hugging Face for the Transformers library
- [d4data/biomedical-ner-all](https://huggingface.co/d4data/biomedical-ner-all) for the biomedical NER model
