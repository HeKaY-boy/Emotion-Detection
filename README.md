# Facial Expression Recognition on FER2013 and KDEF

**5ARE0 - Assignment 2 | Group 6**  
Yuhan Shao (2034964), Himanshu Khartade (2431459), Aniket Zade (2431491)

## Project Overview

This project compares facial expression recognition on FER2013 and KDEF. We evaluate Logistic Regression, SVM, and Random Forest with HOG, facial landmarks, and PCA features. The project also includes cross-dataset experiments, an explainable fuzzy classifier, improvements to the FER2013 baseline, and a live webcam demo.

The final webcam model is an RBF SVM with HOG and landmark features. Its fitted StandardScaler and classifier are saved together in `models/svm_fer_final.pkl`.

## Submission Contents

| File or folder | Purpose |
| --- | --- |
| `Emotion_Detection_Report.pdf` | Final report, including confusion matrices in the appendix |
| `z_load_split_fer2013.ipynb` | Load the FER2013 CSV and create the original training, validation, and test splits |
| `baseline_fer2013_landmarks.ipynb` | FER2013 baseline comparisons, improvement experiments, and final evaluation |
| `z_load_kdef_student.ipynb` | KDEF preprocessing and subject-independent model evaluation |
| `Cross-Dataset Generalization.ipynb` | FER2013-to-KDEF and KDEF-to-FER2013 experiments |
| `new_fuzzy_classifier.ipynb` | Geometric features, fuzzy rules, and validation/test evaluation |
| `baseline_demo_fer2013.py` | Run the final FER2013 model with a webcam |
| `models/` | Saved trained models and fitted preprocessing objects |
| `faceAndLandmarkModels/` | Haar cascade and dlib landmark predictors |
| `auxFiles/kdef_subject_split.pkl` | Fixed KDEF subject split |
| `src/` | Helper functions imported by the KDEF notebook |

Large model files are not all included in the GitHub repository. They must be included in the submitted ZIP. A GitHub download alone does not contain everything needed to run the demo.

## Requirements and Setup

Use a Python environment with the following packages:

- numpy
- pandas
- scipy
- scikit-learn
- scikit-image
- opencv-python
- dlib
- matplotlib
- seaborn
- joblib
- jupyterlab

For an Anaconda setup, one option is:

```bash
conda create -n emotion python=3.10 -y
conda activate emotion
conda install -c conda-forge dlib -y
python -m pip install numpy pandas scipy scikit-learn scikit-image opencv-python matplotlib seaborn joblib jupyterlab
```

Use the package versions from the training environment when loading saved models, especially scikit-learn and numpy. The commands above provide a general setup rather than an exact reproduction of the training environment.

## Datasets and File Locations

Run notebooks and commands from the project root so that relative paths resolve correctly.

### FER2013

Download the single-CSV version of FER2013 from:

https://www.kaggle.com/datasets/deadskull7/fer2013

Place the original CSV at:

```text
data/fer/fer2013.csv
```

The splitting notebook creates:

```text
data/fer2013_train.csv
data/fer2013_validation.csv
data/fer2013_test.csv
```

Training contains 28,709 images, PublicTest validation contains 3,589 images, and PrivateTest contains 3,589 images. Validation is used for model selection; PrivateTest is used for final evaluation.

### KDEF

Obtain KDEF through its official registration page:

https://www.kdef.se/download-2/register

Place the extracted KDEF images under:

```text
data/KDEF_and_AKDEF/KDEF/
```

Only frontal images are used. Sessions A and B of each person stay in the same split. The ordinary classifiers use 56 training subjects and 14 test subjects. The fuzzy experiment further divides the training subjects into 44 training subjects and 12 validation subjects.

The KDEF notebook generates `data/kdef_preprocessed.pkl`, which is required by the cross-dataset and fuzzy notebooks. Raw datasets are obtained separately unless included in the submission.

### Face and Landmark Resources

Keep these files at their existing paths:

```text
faceAndLandmarkModels/haarcascade_frontalface_default.xml
faceAndLandmarkModels/shape_predictor_68_face_landmarks_GTX.dat
faceAndLandmarkModels/shape_predictor_68_face_landmarks.dat
```

The FER2013 notebook, KDEF notebook, and webcam demo use the `_GTX.dat` predictor. The fuzzy notebook uses the predictor without the `_GTX` suffix. Both files are therefore needed to run all experiments.

## Running the Notebooks

From the project root, start JupyterLab:

```bash
python -m jupyterlab
```

Run each notebook's cells from top to bottom in this order:

1. `z_load_split_fer2013.ipynb`
2. `baseline_fer2013_landmarks.ipynb`
3. `z_load_kdef_student.ipynb`
4. `Cross-Dataset Generalization.ipynb`
5. `new_fuzzy_classifier.ipynb`

The last two notebooks can be run in either order after KDEF preprocessing is complete. Create the `data/` and `models/` folders if they do not already exist. Full model training, particularly the FER2013 SVM experiments, may take substantial time.

## Running the Webcam Demo

The demo does not require retraining or the raw datasets. It requires the saved final model, the landmark predictor, the Haar cascade, and a working webcam.

Check that these files are present:

```text
models/svm_fer_final.pkl
faceAndLandmarkModels/shape_predictor_68_face_landmarks_GTX.dat
faceAndLandmarkModels/haarcascade_frontalface_default.xml
```

Then run from the project root:

```bash
conda activate emotion
python baseline_demo_fer2013.py
```

The webcam window displays detected faces and predicted expression labels. Press **Q** while the webcam window is focused to close it. The script uses camera index `0`; if the wrong camera is selected, adjust `cv2.VideoCapture(0)` in the demo script.

If a file-not-found error occurs, check the working directory and the required file paths. If the camera does not open, check camera permissions and close other applications using it.
