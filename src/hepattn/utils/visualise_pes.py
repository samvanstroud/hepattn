"""Standalone positional encoding visualization functions for analysis and testing."""

import matplotlib.pyplot as plt
import torch


def create_simple_pos_enc_visualization(
    posenc: torch.Tensor,
    title: str = "Positional Encoding Matrix",
    save_path: str = "./posenc_visualization.png",
) -> None:
    """Create a simple positional encoding visualization.

    Args:
        posenc: Positional encoding tensor [num_positions, dim]
        title: Title for the plot
        save_path: Optional path to save the figure
    """
    _, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(posenc.T.numpy(), origin="lower", aspect="auto", cmap="viridis")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Position Index", fontsize=12)
    ax.set_ylabel("Positional Encoding Dimension", fontsize=12)
    plt.colorbar(im, ax=ax)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_similarity_matrix_visualization(
    posenc1: torch.Tensor,
    posenc2: torch.Tensor,
    title: str = "Positional Encoding Similarity Matrix",
    save_path: str = "./posenc_similarity.png",
) -> None:
    """Create a similarity matrix visualization for positional encodings.

    Args:
        posenc1: First positional encoding tensor [num_positions, dim]
        posenc2: Second positional encoding tensor [num_positions, dim]
        title: Title for the plot
        save_path: Optional path to save the figure
    """
    # Compute similarity matrix
    similarity = torch.matmul(posenc1, posenc2.T) / posenc2.shape[1]

    _, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(similarity.numpy(), origin="lower", aspect="auto", cmap="viridis")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Position Index", fontsize=12)
    ax.set_ylabel("Position Index", fontsize=12)
    plt.colorbar(im, ax=ax)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
