<h1 align="center">EyeROV Assessment Solution</h1>

<p align="center">
<b>Author:</b> Nandana S Madhu <br>
<b>Date:</b> 25-10-2025
</p>


---

## **📘 Overview**

This project contains three independent Python scripts designed to perform data segmentation, image processing, and basic neural network modeling.
All tasks were completed as part of the **EyeROV interview assessment** within the given 24-hour timeline.

---

## **🧩 Project Components**

### **1️⃣ segment_xyz.py**

* Reads the `data.xyz` file and splits it into **three segments** using two given timestamps.
* If `data.xyz` is missing, it **automatically creates a synthetic file** for testing.
* Outputs three CSV files:

  * `segment_before_t1.csv`
  * `segment_between_t1_t2.csv`
  * `segment_after_t2.csv`

### **2️⃣ image_process.py**

* Performs **image enhancement and denoising** on `oculus.jpg`.
* Uses OpenCV and scikit-image techniques such as:

  * Histogram Equalization
  * CLAHE (Contrast Limited Adaptive Histogram Equalization)
  * Bilateral Filtering
  * Non-Local Means Denoising
  * Total Variation Denoising
  * Median Filtering
  * Unsharp Masking (sharpening)
* Processed outputs are saved in the folder `oculus_output/`.

### **3️⃣ nn_positive_negative_tf.py**

* Implements a **simple neural network** using TensorFlow.
* Classifies integers from -100 to 100 as **positive (1)** or **non-positive (0)**.
* Demonstrates basic model training, normalization, and prediction.

---

## **⚙️ Setup Instructions**

### **Step 1 — Create and Activate Virtual Environment**

Open VS Code Terminal (`Terminal → New Terminal`) and run:

```bash
python -m venv venv
venv\Scripts\activate
```

After activation, your prompt should begin with `(venv)`.

---

### **Step 2 — Install Dependencies**

Install the required Python packages:

```bash
pip install numpy pandas opencv-python scikit-image matplotlib
```

Optional (for deep learning examples):

```bash
pip install tensorflow
pip install torch torchvision
```

---

### **Step 3 — Place Input Files**

* Place `oculus.jpg` in the project folder.
* Place `data.xyz` (if provided) in the same folder.
* If `data.xyz` is not available, the script will create a **demo version automatically**.

---

## **🚀 Running the Scripts**

### **1️⃣ Segment the Data**

```bash
python segment_xyz.py
```

Outputs:

* `segment_before_t1.csv`
* `segment_between_t1_t2.csv`
* `segment_after_t2.csv`

---

### **2️⃣ Process the Image**

```bash
python image_process.py
```

Outputs: several enhanced images saved in the `oculus_output/` folder, such as:

* `hist_equal_gray.png`
* `clahe_gray.png`
* `bilateral.png`
* `nlmeans_color.png`
* `tv_denoise.png`
* `unsharp.png`

---

### **3️⃣ Run the Neural Network (Optional)**

```bash
python nn_positive_negative_tf.py
```

Outputs predictions for test values:

```
Tests: [-5, 0, 1, 50]
Predicted probabilities: [...]
Predicted classes: [0, 0, 1, 1]
```

---

## **🧠 Troubleshooting Tips**

* **Module not found:** Ensure `(venv)` is active and dependencies are installed.
* **Header issue in data.xyz:**

  * If your file includes a header, change:

    ```python
    pd.read_csv(infile, header=None, names=[...])
    ```

    to:

    ```python
    pd.read_csv(infile, header=0)
    ```
* **Timestamp format issues:**

  * If timestamps include dates, provide 5–10 sample lines to adjust the parser.
* **Noisy or dark images:**

  * Adjust parameters such as:

    * `h` in `fastNlMeansDenoisingColored` (e.g., 7–20)
    * `weight` in `denoise_tv_chambolle` (e.g., 0.05–0.5)

---

## **📂 Output Summary**

| Script                     | Input          | Output Files                     | Description                            |
| -------------------------- | -------------- | -------------------------------- | -------------------------------------- |
| segment_xyz.py             | data.xyz       | 3 CSV files                      | Timestamp-based segmentation           |
| image_process.py           | oculus.jpg     | Multiple PNGs in `oculus_output` | Image denoising and enhancement        |
| nn_positive_negative_tf.py | Generated data | Console output                   | Simple classification using TensorFlow |





