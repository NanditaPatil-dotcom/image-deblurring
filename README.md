#  Image Deblurring using Linear Algebra

##  Overview
This project focuses on restoring blurred images using **linear algebra techniques**, specifically **least squares approximation** and **linear transformations**.

Blurring is modeled as a mathematical operation applied to an original image. The goal is to reconstruct the original image from its blurred version using matrix-based methods.

---

## Problem Statement
Images often become blurred due to:
- Motion
- Camera shake
- Defocus

We model the problem as:

A X = B

Where:
- **X** → Original image (unknown)
- **A** → Blur transformation matrix
- **B** → Observed blurred image

Since the system is usually **non-invertible**, we approximate the solution using **least squares**.

---

## Mathematical Concepts Used

### 1. Image as Matrix
- Image → Matrix of pixel intensities (0–255)
- Enables mathematical operations on images

### 2. Linear Transformation
- Blur is modeled as a transformation matrix **A**
- Each pixel spreads into neighboring pixels

### 3. System of Linear Equations
- Formulated as: `AX = B`
- Often inconsistent or underdetermined

### 4. Row Reduction (RREF)
- Used to analyze:
  - Existence of solutions
  - Rank of matrix

### 5. Vector Spaces
- **Column Space** → Possible blurred outputs
- **Null Space** → Lost image details

### 6. Linear Independence
- Identifies independent pixel relationships
- Removes redundant information

### 7. Orthogonalization
- Gram-Schmidt process used to:
  - Separate image components
  - Improve reconstruction

### 8. Least Squares Approximation
We compute:

X̂ = (AᵀA)⁻¹ Aᵀ B

- Gives best possible reconstruction
- Minimizes error

### 9. Eigenvalues & Eigenvectors
- Computed from `AᵀA`
- Helps identify:
  - Important image features
  - Noise components

### 10. Dimensionality Reduction
- Retain dominant eigenvectors
- Improve clarity and reduce noise

---

## ⚙️ Implementation

### Tools Used
- Python
- NumPy
- OpenCV
- Jupyter Notebook

### Workflow
1. Load blurred image
2. Convert to matrix form
3. Apply blur model
4. Solve using least squares
5. Extract features using eigenvalues
6. Reconstruct image

---

##  Output
- Deblurred image
- Error comparison (original vs reconstructed)
- Feature analysis using eigenvalues

---

##  Applications
- Medical Imaging (MRI, CT scans)
- Satellite Image Processing
- Surveillance Systems
- Smartphone Camera Enhancement

---

## 📁 Project Structure
image-deblurring/
│── main.py / notebook.ipynb
│── images/
│── results/
│── README.md


---

## 💡 Key Insight
Blurring causes **loss of information**, making exact recovery impossible.  
Using linear algebra, we compute the **best approximation** rather than exact reconstruction.

---

## 📌 Future Improvements
- Use advanced deconvolution techniques
- Implement regularization (Ridge/Lasso)
- Extend to color images
- Optimize performance for large images

---
