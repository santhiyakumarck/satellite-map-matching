
import os
import json
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

# Drone image to be matched
DRONE_IMAGE = "data/drone/frame_02795.jpg"

# Precomputed satellite embeddings
SATELLITE_EMBEDDINGS = "embeddings/satellite_embeddings_1024.npy"

# Satellite tile metadata
SATELLITE_METADATA = "embeddings/satellite_metadata_1024.json"

# Number of results to return
TOP_K = 3

# DINOv2 model
MODEL_NAME = "dinov2_vitb14"

# DINOv2 input size
IMAGE_SIZE = 518


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("OFFLINE DINOv2 SATELLITE MAP MATCHING")
print("=" * 60)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# LOAD DINOv2
# ============================================================

print("\nLoading DINOv2...")

model = torch.hub.load(
    "facebookresearch/dinov2",
    MODEL_NAME
)

model = model.to(device)
model.eval()

print("DINOv2 loaded successfully.")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def resize_and_pad(image, size=518):
    """
    Resize image while preserving aspect ratio,
    then pad it to a square image.

    Example:
        1280 x 720
             ↓
        resize
             ↓
        ~518 x 291
             ↓
        padding
             ↓
        518 x 518
    """

    width, height = image.size

    scale = size / max(width, height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.BICUBIC
    )

    # Create square canvas
    canvas = Image.new(
        "RGB",
        (size, size),
        (0, 0, 0)
    )

    # Center the resized image
    x = (size - new_width) // 2
    y = (size - new_height) // 2

    canvas.paste(image, (x, y))

    return canvas


# ============================================================
# DINOv2 PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD SATELLITE DATABASE
# ============================================================

print("\nLoading satellite vector database...")

if not os.path.exists(SATELLITE_EMBEDDINGS):
    raise FileNotFoundError(
        f"Satellite embedding file not found:\n"
        f"{SATELLITE_EMBEDDINGS}"
    )

if not os.path.exists(SATELLITE_METADATA):
    raise FileNotFoundError(
        f"Satellite metadata file not found:\n"
        f"{SATELLITE_METADATA}"
    )


satellite_embeddings = np.load(
    SATELLITE_EMBEDDINGS
).astype(np.float32)


with open(SATELLITE_METADATA, "r") as f:
    satellite_metadata = json.load(f)


print(
    f"Satellite embedding shape: "
    f"{satellite_embeddings.shape}"
)

print(
    f"Number of satellite vectors: "
    f"{len(satellite_embeddings)}"
)

print(
    f"Embedding dimension: "
    f"{satellite_embeddings.shape[1]}"
)


# ============================================================
# NORMALIZE SATELLITE EMBEDDINGS
# ============================================================

satellite_norms = np.linalg.norm(
    satellite_embeddings,
    axis=1,
    keepdims=True
)

satellite_norms = np.maximum(
    satellite_norms,
    1e-12
)

satellite_embeddings = (
    satellite_embeddings / satellite_norms
)


# ============================================================
# LOAD DRONE IMAGE
# ============================================================

print("\nLoading drone image...")

if not os.path.exists(DRONE_IMAGE):
    raise FileNotFoundError(
        f"Drone image not found:\n"
        f"{DRONE_IMAGE}"
    )

drone_image = Image.open(
    DRONE_IMAGE
).convert("RGB")

print(
    f"Drone image: {DRONE_IMAGE}"
)

print(
    f"Original image size: "
    f"{drone_image.size}"
)


# ============================================================
# PREPROCESS DRONE IMAGE
# ============================================================

print("\nPreprocessing drone image...")

start_preprocess = time.perf_counter()

drone_processed = resize_and_pad(
    drone_image,
    IMAGE_SIZE
)

drone_tensor = transform(
    drone_processed
)

# Add batch dimension
drone_tensor = drone_tensor.unsqueeze(0)

drone_tensor = drone_tensor.to(device)

if torch.cuda.is_available():
    torch.cuda.synchronize()

end_preprocess = time.perf_counter()

preprocess_time = (
    end_preprocess - start_preprocess
) * 1000


# ============================================================
# EXTRACT DINOv2 DRONE EMBEDDING
# ============================================================

print("\nExtracting DINOv2 feature...")

if torch.cuda.is_available():
    torch.cuda.synchronize()

start_dino = time.perf_counter()

with torch.inference_mode():

    drone_embedding = model(
        drone_tensor
    )

if torch.cuda.is_available():
    torch.cuda.synchronize()

end_dino = time.perf_counter()

dino_time = (
    end_dino - start_dino
) * 1000


# ============================================================
# NORMALIZE DRONE EMBEDDING
# ============================================================

drone_embedding = (
    drone_embedding /
    drone_embedding.norm(
        dim=1,
        keepdim=True
    )
)


# Convert GPU tensor → NumPy
drone_vector = (
    drone_embedding
    .squeeze(0)
    .cpu()
    .numpy()
    .astype(np.float32)
)


print(
    f"DINOv2 vector dimension: "
    f"{drone_vector.shape[0]}"
)


# ============================================================
# COSINE SIMILARITY SEARCH
# ============================================================

print("\nSearching satellite database...")

start_search = time.perf_counter()

# Since both vectors are L2 normalized:
#
# cosine similarity =
# satellite_vector dot drone_vector
#
similarities = (
    satellite_embeddings @ drone_vector
)

# Get Top-K indices
top_indices = np.argsort(
    similarities
)[::-1][:TOP_K]

end_search = time.perf_counter()

search_time = (
    end_search - start_search
) * 1000


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 60)
print(f"TOP {TOP_K} SATELLITE MATCHES")
print("=" * 60)


for rank, index in enumerate(
    top_indices,
    start=1
):

    similarity = float(
        similarities[index]
    )

    metadata = satellite_metadata[index]

    # Metadata keys may depend on the JSON file.
    # These are the keys used by your current pipeline.

    tile_name = metadata.get(
        "tile",
        metadata.get(
            "filename",
            f"index_{index}"
        )
    )

    latitude = metadata.get(
        "latitude",
        metadata.get(
            "lat",
            None
        )
    )

    longitude = metadata.get(
        "longitude",
        metadata.get(
            "lon",
            None
        )
    )

    print(f"\nRank: {rank}")

    print(
        f"Tile: {tile_name}"
    )

    print(
        f"Similarity: {similarity:.6f}"
    )

    if latitude is not None:
        print(
            f"Latitude: {float(latitude):.8f}"
        )

    if longitude is not None:
        print(
            f"Longitude: {float(longitude):.8f}"
        )


# ============================================================
# TIMING
# ============================================================

total_time = (
    preprocess_time
    + dino_time
    + search_time
)

fps = 1000.0 / total_time


print("\n")
print("=" * 60)
print("PROCESSING TIME")
print("=" * 60)

print(
    f"Preprocessing time : "
    f"{preprocess_time:.2f} ms"
)

print(
    f"DINOv2 vector time : "
    f"{dino_time:.2f} ms"
)

print(
    f"Vector search time : "
    f"{search_time:.2f} ms"
)

print(
    f"TOTAL TIME         : "
    f"{total_time:.2f} ms"
)

print(
    f"Approx FPS         : "
    f"{fps:.2f}"
)

print("=" * 60)
