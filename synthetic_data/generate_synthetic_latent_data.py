"""Generate simple synthetic latent-feature arrays for smoke tests.

The generated files are local test artifacts. They are ignored by git and should
not be committed.
"""

from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "synthetic_latent"

NUM_CLASSES = 3
SAMPLES_PER_CLASS = 120
LATENT_DIM = 32


def main():
    rng = np.random.default_rng()
    total_samples = NUM_CLASSES * SAMPLES_PER_CLASS

    gaussian_features = rng.normal(0, 1, size=(total_samples, LATENT_DIM))
    gaussian_labels = rng.integers(0, NUM_CLASSES, size=total_samples)

    class_labels = np.repeat(np.arange(NUM_CLASSES), SAMPLES_PER_CLASS)

    class_centers = np.zeros((NUM_CLASSES, LATENT_DIM))
    class_centers[0, 0] = 4
    class_centers[1, 1] = 4
    class_centers[2, 2] = 4

    class_features = class_centers[class_labels]
    class_features += rng.normal(0, 0.6, size=class_features.shape)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "synthetic_gaussian_noise_features.npy", gaussian_features)
    np.save(OUTPUT_DIR / "synthetic_gaussian_noise_labels.npy", gaussian_labels)
    np.save(OUTPUT_DIR / "synthetic_class_features.npy", class_features)
    np.save(OUTPUT_DIR / "synthetic_class_labels.npy", class_labels)

    print(f"Wrote synthetic arrays to {OUTPUT_DIR}")
    print(f"Feature shape: ({total_samples}, {LATENT_DIM})")
    print(f"Classes: {NUM_CLASSES}, samples per class: {SAMPLES_PER_CLASS}")


if __name__ == "__main__":
    main()
