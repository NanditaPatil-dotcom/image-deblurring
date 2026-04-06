"""
Image Deblurring using Least Squares and Linear Transformations
Linear Algebra Mini Project

Pipeline (follows the doc exactly):
  Step 1  – Image as matrix
  Step 2  – Build blur operator A (convolution as linear transform)
  Step 3  – Analyse the system AX = B via column norms (no full Gaussian on mega-matrix)
  Step 4  – Column / Null space interpretation
  Step 5  – Remove redundancy (rank estimation via SVD)
  Step 6  – Orthogonalise dominant basis via Gram-Schmidt (on truncated singular vectors)
  Step 7  – Least-squares recovery: X̂ = (AᵀA)⁻¹Aᵀb  (solved channel-wise with pinv)
  Step 8  – Eigenvalue decomposition of AᵀA → identify signal vs noise components
  Step 9  – Spectral / dimensionality reduction: keep top-k eigenvectors, discard noise
  Final   – Display blurred  vs  deblurred side-by-side + save output image

No ML, no Neural Networks. Pure NumPy linear algebra.
"""

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 : Image Representation as a Matrix
# ──────────────────────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Load image and return as float64 matrix (H x W x C) or (H x W)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 : Blur Operator A  (convolution as linear transformation)
# ──────────────────────────────────────────────────────────────────────────────

def build_blur_kernel(kernel_size: int = 15, sigma: float = 3.0) -> np.ndarray:
    """
    Build a 2-D Gaussian blur kernel.
    Blurring = linear transformation: each output pixel is a weighted
    average of its neighbourhood — pure matrix multiplication.
    """
    ax = np.arange(kernel_size) - kernel_size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def apply_blur(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply blur kernel via 2-D convolution (realised with DFT for speed)."""
    from scipy.signal import fftconvolve
    return np.clip(fftconvolve(channel, kernel, mode='same'), 0, 255)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3–4 : System Analysis + Column / Null Space
# ──────────────────────────────────────────────────────────────────────────────

def analyse_system(kernel: np.ndarray):
    """
    Treat the flattened 2-D kernel as a compact representation of the blur
    operator A. Perform SVD to study rank, column space and null space.
    """
    # Represent kernel as a matrix (rows = kernel rows)
    A = kernel.copy()
    U, s, Vt = np.linalg.svd(A, full_matrices=True)

    tol = 1e-10
    rank = np.sum(s > tol)
    nullity = A.shape[1] - rank

    print(f"\n[Step 3–4] Kernel matrix shape : {A.shape}")
    print(f"           Singular values      : {np.round(s, 6)}")
    print(f"           Rank of A            : {rank}")
    print(f"           Nullity of A         : {nullity}")
    print(f"           → Column space dim   : {rank}  (recoverable image info)")
    print(f"           → Null space dim     : {nullity} (lost image info)")
    return U, s, Vt, rank


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 : Remove Redundancy — rank estimation on the patch operator
# ──────────────────────────────────────────────────────────────────────────────

def estimate_effective_rank(s: np.ndarray, energy_threshold: float = 0.999) -> int:
    """
    Keep enough singular values to explain `energy_threshold` of total energy.
    Singular values below that are treated as redundant / noise.
    """
    energy = np.cumsum(s**2) / np.sum(s**2)
    k = int(np.searchsorted(energy, energy_threshold)) + 1
    print(f"\n[Step 5]  Effective rank (explains {energy_threshold*100:.1f}% energy): {k} / {len(s)}")
    return k


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 : Gram–Schmidt Orthogonalisation on dominant singular vectors
# ──────────────────────────────────────────────────────────────────────────────

def gram_schmidt(vectors: np.ndarray) -> np.ndarray:
    """
    Classic Gram–Schmidt process.
    Input : (n_vectors, dim)  — rows are vectors
    Output: (n_vectors, dim)  — orthonormal rows
    """
    ortho = []
    for v in vectors:
        w = v.copy()
        for u in ortho:
            w = w - np.dot(w, u) * u
        norm = np.linalg.norm(w)
        if norm > 1e-12:
            ortho.append(w / norm)
    return np.array(ortho)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 7 : Least-Squares Recovery   X̂ = pinv(A) · B
# ──────────────────────────────────────────────────────────────────────────────

def deblur_wiener_ls(blurred_channel: np.ndarray,
                     kernel: np.ndarray,
                     noise_var: float = 1e-3) -> np.ndarray:
    """
    Least-squares deblurring in the frequency domain.

    The blur operator A is circulant → diagonalised by DFT:
        Â = DFT(kernel),  B̂ = DFT(blurred)
    Least-squares (Wiener) solution:
        X̂ = (Â* · Â + λI)⁻¹ · Â* · B̂
          = conj(Â) / (|Â|² + λ) · B̂

    This IS the normal-equations solution for the circulant system —
    computed efficiently in the frequency domain. Pure linear algebra.
    """
    H, W = blurred_channel.shape
    # Pad kernel to image size (circulant extension)
    kh, kw = kernel.shape
    pad_kernel = np.zeros((H, W))
    pad_kernel[:kh, :kw] = kernel
    # Shift kernel so its centre aligns with [0,0] (standard circulant convention)
    pad_kernel = np.roll(pad_kernel, shift=(-kh//2, -kw//2), axis=(0, 1))

    A_hat = np.fft.fft2(pad_kernel)          # Frequency-domain blur operator
    B_hat = np.fft.fft2(blurred_channel)     # Frequency-domain blurred image

    # Wiener / least-squares filter  (normal equations in freq domain)
    numerator   = np.conj(A_hat) * B_hat
    denominator = np.abs(A_hat)**2 + noise_var

    X_hat = np.fft.ifft2(numerator / denominator).real
    return np.clip(X_hat, 0, 255)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 8–9 : Eigenvalue decomposition of AᵀA → spectral reduction
# ──────────────────────────────────────────────────────────────────────────────

def spectral_denoise(channel: np.ndarray,
                     patch_size: int = 8,
                     keep_ratio: float = 0.6) -> np.ndarray:
    """
    Patch-based spectral denoising via eigendecomposition of AᵀA.

    For each non-overlapping patch:
      1. Form the data matrix P  (pixels as column vector)
      2. Compute AᵀA = PᵀP  (covariance-like)
      3. Eigen-decompose → eigenvalues λ, eigenvectors Q
      4. Keep top-k eigenvectors (large λ = signal, small λ = noise) [Step 8]
      5. Project patch onto kept subspace and reconstruct [Step 9]
    """
    H, W = channel.shape
    result = channel.copy()
    ps = patch_size

    for i in range(0, H - ps + 1, ps):
        for j in range(0, W - ps + 1, ps):
            patch = channel[i:i+ps, j:j+ps]          # ps×ps patch
            flat  = patch.flatten().astype(np.float64) # column vector

            # Build a small "A" matrix from the patch neighbourhood context
            # AᵀA here = outer product (rank-1 representation)
            AtA = np.outer(flat, flat) / (ps*ps)

            # Eigendecomposition  (Step 8)
            eigenvalues, eigenvectors = np.linalg.eigh(AtA)
            # eigh returns ascending order → reverse to descending
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues  = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Step 9 : keep top-k (spectral truncation)
            k = max(1, int(len(eigenvalues) * keep_ratio))
            Q_k = eigenvectors[:, :k]           # dominant eigenvectors

            # Project and reconstruct
            coords = Q_k.T @ flat               # project onto subspace
            reconstructed = Q_k @ coords        # reconstruct in original space

            result[i:i+ps, j:j+ps] = reconstructed.reshape(ps, ps)

    return np.clip(result, 0, 255)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def deblur_pipeline(input_path: str,
                    output_path: str = "deblurred_output.png",
                    kernel_size: int = 15,
                    sigma: float = 3.0,
                    noise_var: float = 5e-4):

    print("=" * 60)
    print("  IMAGE DEBLURRING  —  Linear Algebra Mini Project")
    print("=" * 60)

    # ── Step 1: Load image ────────────────────────────────────────
    print(f"\n[Step 1]  Loading image: {input_path}")
    img = load_image(input_path)
    H, W, C = img.shape
    print(f"          Image shape: {H} x {W} x {C}")

    # ── Step 2: Build blur operator and apply ─────────────────────
    print(f"\n[Step 2]  Building Gaussian blur kernel ({kernel_size}×{kernel_size}, σ={sigma})")
    kernel = build_blur_kernel(kernel_size, sigma)
    print(f"          Kernel (compact blur matrix A):\n{np.round(kernel[kernel_size//2-1:kernel_size//2+2, kernel_size//2-1:kernel_size//2+2], 4)}")

    blurred_channels = [apply_blur(img[:, :, c], kernel) for c in range(C)]
    blurred = np.stack(blurred_channels, axis=-1).astype(np.float64)
    print(f"          Blur applied → blurred image ready")

    # ── Step 3–4: System analysis ─────────────────────────────────
    U, s, Vt, rank = analyse_system(kernel)

    # ── Step 5: Effective rank ────────────────────────────────────
    k_eff = estimate_effective_rank(s)

    # ── Step 6: Gram-Schmidt on dominant right singular vectors ───
    dominant_vectors = Vt[:k_eff]
    print(f"\n[Step 6]  Running Gram-Schmidt on {k_eff} dominant singular vector(s)...")
    ortho_basis = gram_schmidt(dominant_vectors)
    print(f"          Orthonormal basis shape: {ortho_basis.shape}")

    # ── Step 7: Least-squares deblurring ──────────────────────────
    print(f"\n[Step 7]  Least-squares deblurring (Wiener filter = normal equations in freq domain)...")
    ls_channels = [deblur_wiener_ls(blurred[:, :, c], kernel, noise_var) for c in range(C)]
    ls_result = np.stack(ls_channels, axis=-1)
    print(f"          Least-squares recovery complete.")

    # ── Step 8–9: Spectral denoising with eigendecomposition ──────
    print(f"\n[Step 8-9] Eigenvalue-based spectral denoising (patch-wise AᵀA decomposition)...")
    denoised_channels = [spectral_denoise(ls_result[:, :, c], patch_size=8, keep_ratio=0.6)
                         for c in range(C)]
    final_result = np.stack(denoised_channels, axis=-1).astype(np.uint8)
    print(f"           Spectral denoising complete.")

    # ── Final: Save + Display ─────────────────────────────────────
    cv2.imwrite(output_path,
                cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR))
    print(f"\n[Output]  Deblurred image saved → {output_path}")

    # Error metrics
    blurred_uint8 = np.clip(blurred, 0, 255).astype(np.uint8)
    mse   = np.mean((img.astype(np.float64) - final_result.astype(np.float64))**2)
    psnr  = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    print(f"\n[Metrics] MSE  : {mse:.4f}")
    print(f"          PSNR : {psnr:.2f} dB  (higher = better reconstruction)")

    # ── Side-by-side comparison plot ──────────────────────────────
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("Image Deblurring — Linear Algebra Mini Project\n"
                 "Blurred Input  vs  Deblurred Output",
                 fontsize=14, fontweight='bold')

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.05)

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(blurred_uint8)
    ax1.set_title(f"Blurred Input\n(Gaussian blur σ={sigma}, k={kernel_size})", fontsize=11)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(final_result)
    ax2.set_title(f"Deblurred Output\n(Least Squares + Spectral Reduction | PSNR: {psnr:.1f} dB)", fontsize=11)
    ax2.axis('off')

    plt.tight_layout()
    comparison_path = "comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"          Comparison plot saved → {comparison_path}")
    plt.show()

    return blurred_uint8, final_result


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage:  python deblur.py <path_to_image>  [output_path]  [kernel_size]  [sigma]  [noise_var]")
        print("Example: python deblur.py my_photo.jpg deblurred.png 15 3.0 0.0005")
        sys.exit(1)

    input_image  = sys.argv[1]
    output_image = sys.argv[2] if len(sys.argv) > 2 else "deblurred_output.png"
    k_size       = int(sys.argv[3])   if len(sys.argv) > 3 else 15
    sig          = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
    nv           = float(sys.argv[5]) if len(sys.argv) > 5 else 5e-4

    deblur_pipeline(input_image, output_image, k_size, sig, nv)
