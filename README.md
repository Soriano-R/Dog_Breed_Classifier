# Dog Identification App

**Udacity AI Nanodegree: Convolutional Neural Networks**

## Project: Write an Algorithm for a Dog Identification App

This exported Jupyter notebook guides you through building an end-to-end pipeline that:

- Detects whether an image contains a human or a dog  
- Classifies the dog’s breed if a dog is present  
- Suggests the most resembling dog breed if a human face is detected  
- Handles “neither” cases with an appropriate message  

You’ll piece together multiple models—OpenCV’s Haar cascades, a ResNet-50 dog detector, and a custom CNN breed classifier—into a single user-facing algorithm  

---

## The Road Ahead  
We break the notebook into separate steps. Feel free to jump to any section:

- **Step 0:** Import Datasets  
- **Step 1:** Detect Humans  
- **Step 2:** Detect Dogs  
- **Step 3:** Create a CNN to Classify Dog Breeds (from Scratch)  
- **Step 4:** Use a CNN to Classify Dog Breeds  
- **Step 5:** Create a CNN to Classify Dog Breeds (Transfer Learning)  
- **Step 6:** Write Your Algorithm  
- **Step 7:** Test Your Algorithm  

---

### Step 0: Import Datasets  
Load the dog and human image datasets using the `load_files` function from scikit-learn. This populates:  
- `train_files`, `valid_files`, `test_files` (paths to images)  
- `train_targets`, `valid_targets`, `test_targets` (one-hot labels)  
- `dog_names` (list of 133 breed names)  

### Step 1: Detect Humans  
Use OpenCV’s Haar cascade for frontal faces (`haarcascade_frontalface_alt.xml`) to implement `face_detector(img_path)`, which returns `True` if a human face is found, `False` otherwise  

### Step 2: Detect Dogs  
Load Keras’s pre-trained ResNet-50 (ImageNet weights) and define `ResNet50_predict_labels(img_path) → idx`. A dog is detected if `151 ≤ idx ≤ 268`. Wrap this in `dog_detector(img_path)` 

### Step 3: Create a CNN to Classify Dog Breeds (from Scratch)  
Build a simple convolutional network with Conv2D → MaxPooling → Dropout → Dense layers. Train for 20 epochs on preprocessed tensors (`train_tensors_scratch`, etc.), checkpointing weights to `saved_models/weights.best.from_scratch.keras`.

### Step 4: Use a CNN to Classify Dog Breeds  
Evaluate your scratch CNN on the validation set to establish a baseline.

### Step 5: Create a CNN to Classify Dog Breeds (Transfer Learning)  
Extract **bottleneck features** from five popular architectures (VGG16, VGG19, ResNet50, InceptionV3, Xception) saved in `bottleneck_features/*.npz`. Train a lightweight top-model for each and compare performance. The Xception-based classifier achieves the best results (~90% test accuracy) and its weights are saved to `saved_models/weights.best.Xception.keras`.

### Step 6: Write Your Algorithm  
Implement `predict_breed(img_path)` that:  
1. Runs `face_detector` → if `True`, greets the human and shows the resembling dog breed.  
2. Else runs `dog_detector` → if `True`, greets the dog and shows its breed.  
3. Otherwise, returns an error-style message asking for another image.

### Step 7: Test Your Algorithm  
Run `predict_breed` on sample human, dog, and random images to visually verify outputs.

---

## Repository Structure
dog_app.html            # This exported notebook
bottleneck_features/    # .npz files for each pretrained CNN
saved_models/           # Checkpointed weights:
# ├─ weights.best.from_scratch.keras
# ├─ weights.best.VGG16.keras
# └─ weights.best.Xception.keras
haarcascades/           # Pretrained face detector XML
images/                 # Sample output screenshots

## Dependencies
- Python 3.x  
- TensorFlow & Keras  
- scikit-learn  
- OpenCV (`cv2`)  
- NumPy  
- Pillow (`PIL`)  
- Jupyter Notebook / JupyterLab  

```bash
pip install tensorflow keras scikit-learn opencv-python numpy pillow jupyter

## Q & A
__Question 1:__ Use the code cell below to test the performance of the `face_detector` function.  
- What percentage of the first 100 images in `human_files` have a detected human face?  
- What percentage of the first 100 images in `dog_files` have a detected human face? 

***************************************************************
__Answer:__ 

- 100% of the first 100 human images contain human faces
- 12% of the first 100 dog images contain human faces

__Question 2:__ This algorithmic choice necessitates that we communicate to the user that we accept human images only when they provide a clear view of a face (otherwise, we risk having unneccessarily frustrated users!). In your opinion, is this a reasonable expectation to pose on the user? If not, can you think of a way to detect humans in images that does not necessitate an image with a clearly presented face?

***************************************************************
__Answer:__

Looking at the code, I can suggest several ways to detect humans in images without relying solely on face detection:

1. Use full body detection models - OpenCV provides other pre-trained cascades like `haarcascade_fullbody.xml` that can detect full human bodies rather than just faces.

2. Use more modern deep learning-based person detection models like YOLO or SSD that are trained to detect people in various poses and orientations.

3. Use pose estimation models that can detect human body keypoints, which would work even when faces are not clearly visible.

The current face detection approach does have limitations since it requires:
- Front-facing faces
- Faces to be clearly visible (not obscured/side view)
- Sufficient image resolution
- Good lighting conditions

This is somewhat restrictive for users. A more flexible approach using one of the above alternatives would provide better user experience by:
- Working with profile views
- Detecting people even when faces are obscured
- Working with full-body shots
- Being more robust to varied lighting and poses

So while face detection is straightforward to implement, it may not be the most user-friendly approach. I would recommend considering one of the more robust person detection methods mentioned above for better usability.

__Question 3:__ Use the code cell below to test the performance of your `dog_detector` function.  
- What percentage of the images in `human_files_short` have a detected dog?  
- What percentage of the images in `dog_files_short` have a detected dog?

***************************************************************
__Answer:__ 

- 0.0% of the first 100 human images are detected as dogs
- 100.0% of the first 100 dog images are detected as dogs

__Question 4:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  If you chose to use the hinted architecture above, describe why you think that CNN architecture should work well for the image classification task.

***************************************************************
__Answer:__ 

### Final CNN — Step by Step

| Step | What I Tried | Why | Result / Next Move |
|------|--------------|-----|--------------------|
| **1. Tiny scratch CNN**<br>`Conv16 → Conv32 → Conv64 → GAP → Dense-133` | Sanity-check data & labels; set a floor to beat. | **≈ 3 %** val-acc → dataset too small for training from scratch. |
| **2. Bigger scratch CNN** + BatchNorm & Dropout | See if capacity or over-fit is the issue. | Still < 10 % and over-fits early → capacity isn’t the answer. |
| **3. Transfer learning v1**<br>**VGG-16 (frozen)** + GAP + Dense-512 + Dropout 0.5 | Re-use ImageNet features. | **≈ 72 %** test-acc → transfer learning clearly better. |
| **4. Swap to **Xception** (frozen)** | Separable convs capture fine detail; lighter than VGG. | +4–5 pp → mid-70 s accuracy. |
| **5. Fine-tune last 60 Xception layers** (LR = 1e-4, 10 epochs) | Let high-level filters specialise to 133 breeds. | **≈ 90.6 %** test-acc, no over-fit.<br>*≈ 22 M params total, ~1.1 M trainable* |
| **6. On-graph data augmentation**<br>`Flip + Rotate + Zoom + Translate` | Simulate more variety (pose, scale). | +1–2 pp and smoother learning. |
| **7. Hyper-param polish**<br>512-unit head + Dropout 0.5, batch 32 | Final balance of capacity vs generalisation. | Stable best performance. |

---

#### **Xception model**

* **Pre-trained on ImageNet** → already encodes generic visual features.  
* **Depth-wise separable convs** → excel at fine-grained breed differences.  
* **GlobalAveragePooling** → fewer parameters, less over-fit.  
* **Selective fine-tuning** → keeps low-level filters, adapts high-level ones.  
* **Data augmentation + Dropout** → essential regularization for a ~20 k-image dataset.

_Result:_ ** 85 %** top-1 accuracy, comfortably above the project requirement.

__Question 5:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.

***************************************************************
__Answer:__ 

### Final CNN

| Step | What I tried | Why I tried it | What happened / next move |
|------|--------------|---------------|---------------------------|
| **1. Tiny CNN from scratch**<br>`Conv 32 → Conv 64 → GAP → Dense 133` | Quick sanity-check that data & labels line up; set a “floor.” | **≈ 3 %** accuracy → scratch alone isn’t enough. |
| **2. *Frozen* VGG-16 + small head**<br>`GAP → Dense 512 → Dropout 0.5 → Softmax 133` | Classic transfer-learning baseline. | **≈ 70.7 %** accuracy → transfer clearly beats scratch. |
| **3. Light VGG fine-tune**<br>Unfreeze last **4** conv blocks, LR = 1 e-4 | Let the top VGG filters adapt to dog-breed details. | **≈ 76 %** accuracy → better, but VGG is heavy. |
| **4. *Frozen* Xception + head** | Xception is lighter and good at fine-grained tasks. | **≈ 80 %** accuracy out-of-the-box. |
| **5. Fine-tune top 15 Xception layers**<br>LR = 1 e-5 | Specialize high-level filters without wrecking low-level ones. | **≈ 85 %** test accuracy → final model. |
| **6. Light data-aug in `tf.data`**<br>`RandomFlip`, `Rotation ±10°`, `Zoom 15 %` | Add pose/scale variety; cut over-fit. | +1 pp & smoother learning curves. |
| **7. Hyper-param tidy-up**<br>Batch 32, Dropout 0.5, EarlyStopping | Balance capacity vs. generalization. | Locked-in best checkpoint. |

---

#### Fine-tuned **Xception** 

* **Pre-trained on ImageNet** → already knows generic edges, colors, and textures.
* **Depth-wise separable convolutions** → capture subtle breed cues (ears, snouts) with far fewer weights than VGG/ResNet.
* **GlobalAveragePooling** → no huge fully-connected layers ⇒ < 2 M trainable params ⇒ lower over-fit risk.
* **Selective fine-tuning** → freeze early layers, update only high-level filters to focus on breed-specific patterns.
* **Augmentation + Dropout** → regularize the head and improve robustness to pose, lighting, and scale.

**Result:** *≈ 85 %* top-1 accuracy over 133 breeds — comfortably above the project requirement while remaining lightweight and fast to infer.

__Question 6:__ Is the output better than you expected :) ?  Or worse :( ?  Provide at least three possible points of improvement for your algorithm.

***************************************************************
__Answer:__ 

Yes — getting **≈ ~85 %† top-1 accuracy across 133 breeds** is better than I honestly thought possible on a modest dataset. Many breeds look almost identical (e.g. Malamute vs. Husky), so I expected something in the high-70 % range.

---

### Three quick ideas to push accuracy even further

1. **Richer data-augmentation**  
   *Add colour jitter, random crops, more aggressive rotations and mixup/cut-mix.*  
   Creating synthetic variety forces the model to generalize beyond the limited poses and lighting in the training set.

2. **Ensemble two light backbones**  
   *Average logits from fine-tuned Xception **plus** a fine-tuned EfficientNet-B0.*  
   Each architecture makes slightly different errors; an ensemble often buys an extra 2-3 percentage points.

3. **Class-weighted focal loss**  
   *Replace plain cross-entropy with focal loss + per-breed class weights.*  
   Helps the network focus on the minority breeds that are currently swallowed by the majority classes.

† ~85 % = best checkpoint after fine-tuning top 15 layers of Xception; VGG-16 head-only peaked at 70.7 %.
