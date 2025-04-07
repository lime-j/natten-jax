import os
# Environment variables typically not needed for pure JAX vmap approach
os.environ["TPU_CHIPS_PER_PROCESS_BOUNDS"] = os.environ.get("TPU_CHIPS_PER_PROCESS_BOUNDS", "1,1,1")
os.environ["TPU_PROCESS_BOUNDS"] = os.environ.get("TPU_PROCESS_BOUNDS", "1,1,1")
os.environ["TPU_VISIBLE_DEVICES"] = os.environ.get("TPU_VISIBLE_DEVICES", "0")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
import numpy as np # Only for sqrt in kernel_size calculation if needed
import torch # For comparison reference
from jax2torch import jax2torch
# --- Helper Functions ---
def get_hw_indices(idx, H, W):
    """Calculates 2D coordinates from linear index."""
    rows = idx // W
    cols = idx % W
    return rows, cols

# --- Core Single-Query NA Computation (using padding) ---
def compute_na_single_query_padded(
    q_query,           # Single query vector (D,)
    k_padded_bh,       # Padded K tensor for the current batch/head (H+2p, W+2p, D)
    v_padded_bh,       # Padded V tensor for the current batch/head (H+2p, W+2p, D)
    q_row: int,        # Row index of the query
    q_col: int,        # Column index of the query
    kernel_size: int,  # Size of the neighborhood (e.g., 3)
    head_dim: int,     # Dimension of the head
    scale: float       # Attention scaling factor (1/sqrt(head_dim))
    ):
    """Computes attention for a single query using padded K/V."""
    neigh_size = kernel_size * kernel_size
    # Note: padding `pad = kernel_size // 2` is implicitly handled by indexing

    # --- 1. Extract Neighborhood using Slicing ---
    # Use dynamic_slice to extract the KxK neighborhood around the query position (q_row, q_col)
    # The starting indices in the padded tensor are (q_row, q_col) because padding shifts coords.
    # Slice shape: (kernel_size, kernel_size, head_dim)
    neighborhood_k = jax.lax.dynamic_slice(
        k_padded_bh,
        (q_row, q_col, 0),           # Start indices (row, col, dim)
        (kernel_size, kernel_size, head_dim) # Slice sizes
    )
    neighborhood_v = jax.lax.dynamic_slice(
        v_padded_bh,
        (q_row, q_col, 0),
        (kernel_size, kernel_size, head_dim)
    )

    # Reshape neighborhood to (neigh_size, head_dim)
    k_gathered = jnp.reshape(neighborhood_k, (neigh_size, head_dim))
    v_gathered = jnp.reshape(neighborhood_v, (neigh_size, head_dim))

    # --- 2. Compute Attention Scores ---
    # q_query shape: (head_dim,)
    # k_gathered shape: (neigh_size, head_dim)
    # Result shape: (neigh_size,)
    # Use einsum for clarity: sum_k(q[k] * K[n, k]) -> score[n]
    scores = jnp.einsum('k,nk->n', q_query, k_gathered) * scale
    scores = scores.astype(jnp.float32) # Use float32 for softmax stability

    # --- 3. Compute Softmax ---
    # No masking needed here as out-of-bounds areas were implicitly handled by padding (usually with 0)
    # If padding value affects results, masking might still be needed depending on padding value.
    probs = jax.nn.softmax(scores).astype(q_query.dtype) # Cast back

    # --- 4. Compute Output ---
    # probs shape: (neigh_size,)
    # v_gathered shape: (neigh_size, head_dim)
    # Result shape: (head_dim,)
    # Use einsum: sum_n(prob[n] * V[n, k]) -> out[k]
    output = jnp.einsum('n,nk->k', probs, v_gathered)

    return output

# --- VMAP Wrapper for Neighborhood Attention ---
def neighborhood_attention_vmap(
    q: jax.Array, k: jax.Array, v: jax.Array, kernel_size: int
    ):
    """Neighborhood Attention implementation using jax.vmap and padding."""
    BATCH, HEADS, H, W, HEAD_DIM = q.shape
    assert q.shape == k.shape == v.shape, "Input shapes must match"
    assert kernel_size % 2 == 1, "Kernel size must be odd"

    pad = kernel_size // 2
    BH = BATCH * HEADS
    scale = float(1.0 / HEAD_DIM**0.5)
    q = q * scale
    # --- 1. Reshape Inputs ---
    # Merge Batch and Head dimensions for easier vmap application
    q_bh = jnp.reshape(q, (BH, H, W, HEAD_DIM))
    k_bh = jnp.reshape(k, (BH, H, W, HEAD_DIM))
    v_bh = jnp.reshape(v, (BH, H, W, HEAD_DIM))

    # --- 2. Pad K and V ---
    # Pad spatial dimensions (H, W)
    # Padding format: ((before_axis1, after_axis1), (before_axis2, after_axis2), ...)
    padding_config = (
        (0, 0, 0),       # No padding for Batch*Head dim
        (pad, pad, 0),   # Pad Height
        (pad, pad, 0),   # Pad Width
        (0, 0, 0)        # No padding for Head_Dim dim
    )
    k_padded = jax.lax.pad(k_bh, padding_value=0.0, padding_config=padding_config)
    v_padded = jax.lax.pad(v_bh, padding_value=0.0, padding_config=padding_config)
    # Padded shape: (BH, H + 2*pad, W + 2*pad, HEAD_DIM)

    # --- 3. Prepare Query Coordinates ---
    # Generate linear indices and corresponding 2D coordinates for all possible query positions
    all_q_indices = jnp.arange(H * W)
    all_q_rows, all_q_cols = get_hw_indices(all_q_indices, H, W) # Shapes: (N_CTX,)

    # --- 4. Define Vmapped Function ---
    # We need to vmap over two dimensions: Batch*Head (BH) and Query Index (N_CTX)

    # Inner vmap: Maps over query indices (rows and columns) for a *single* BH element
    # It receives the Q tensor reshaped for spatial access (H, W, D) and the *single* padded K/V
    # vmap_over_queries = jax.vmap(
    #     compute_na_single_query_padded,
    #     in_axes=(0, None, None, 0, 0, None, None, None), # q_query(0), k_pad(N), v_pad(N), row(0), col(0), ksize(N), head_dim(N), scale(N)
    #     # axis_size=N_CTX # Optional: specify size if needed, usually inferred
    # )
    # Expected input signature for vmap_over_queries after this:
    # (q_queries_hw: (H, W, D), k_padded_single_bh: (H+2p, W+2p, D), v_padded_single_bh: (H+2p, W+2p, D),
    #  q_rows: (N,), q_cols: (N,), kernel_size, head_dim, scale)
    # But this is tricky because the first arg needs spatial indexing.

    # Alternative: Flatten Q for the inner vmap
    vmap_over_queries_flat = jax.vmap(
        compute_na_single_query_padded,
        in_axes=(0, None, None, 0, 0, None, None, None), # q_flat(0), k_pad(N), v_pad(N), row(0), col(0), ksize(N), head_dim(N), scale(N)
    )
    # Expected input signature for vmap_over_queries_flat:
    # (q_queries_flat: (N, D), k_padded_single_bh: (H+2p, W+2p, D), v_padded_single_bh: (H+2p, W+2p, D),
    #  q_rows: (N,), q_cols: (N,), kernel_size, head_dim, scale)
    # Output shape: (N, D)


    # Outer vmap: Maps over the Batch*Head dimension
    # It passes the corresponding slice of q_bh, k_padded, v_padded to the inner vmap
    # We also need to pass the constant coordinate arrays (all_q_rows, all_q_cols)
    vmap_over_bh = jax.vmap(
        # Apply the inner vmap function
        lambda q_single_bh, k_pad_single_bh, v_pad_single_bh: vmap_over_queries_flat(
            jnp.reshape(q_single_bh, (H * W, HEAD_DIM)), # Flatten Q for this BH element
            k_pad_single_bh,
            v_pad_single_bh,
            all_q_rows,      # Broadcast rows
            all_q_cols,      # Broadcast cols
            kernel_size,
            HEAD_DIM,
            scale
        ),
        in_axes=(0, 0, 0) # Map over axis 0 of q_bh, k_padded, v_padded
    )
    # Expected input signature for vmap_over_bh:
    # (q_bh: (BH, H, W, D), k_padded: (BH, H+2p, W+2p, D), v_padded: (BH, H+2p, W+2p, D))
    # Output shape: (BH, N_CTX, HEAD_DIM)

    # --- 5. Execute ---
    output_bh_flat = vmap_over_bh(q_bh, k_padded, v_padded)

    # --- 6. Reshape Output ---
    output = jnp.reshape(output_bh_flat, (BATCH, HEADS, H, W, HEAD_DIM))

    return output


# --- PyTorch Reference Implementation (Copied from previous examples) ---
def torch_neighborhood_attention(q_th, k_th, v_th, kernel_size):
    """Plain PyTorch implementation for neighborhood attention."""
    print("Running PyTorch reference...")
    B, HD, H, W, D = q_th.shape
    q_th = q_th.float() # Use float32 for stable reference calculation
    k_th = k_th.float()
    v_th = v_th.float()
    output = torch.zeros_like(q_th)
    k_size = kernel_size
    pad = k_size // 2
    scale = float(1.0 / D**0.5)
    q_th = q_th * scale
    q_reshaped = q_th.reshape(B * HD, H, W, D)
    k_reshaped = k_th.reshape(B * HD, H, W, D)
    v_reshaped = v_th.reshape(B * HD, H, W, D)

    k_padded = torch.nn.functional.pad(k_reshaped.permute(0, 3, 1, 2), (pad, pad, pad, pad), mode='constant', value=0).permute(0, 2, 3, 1)
    v_padded = torch.nn.functional.pad(v_reshaped.permute(0, 3, 1, 2), (pad, pad, pad, pad), mode='constant', value=0).permute(0, 2, 3, 1)

    out_reshaped = output.reshape(B * HD, H, W, D)

    for r in range(H):
        for c in range(W):
            neighborhood_k = k_padded[:, r:r+k_size, c:c+k_size, :].reshape(B * HD, k_size * k_size, D)
            neighborhood_v = v_padded[:, r:r+k_size, c:c+k_size, :].reshape(B * HD, k_size * k_size, D)
            query_pix = q_reshaped[:, r, c, :].unsqueeze(1)
            scores = torch.matmul(query_pix * scale, neighborhood_k.transpose(-1, -2))
            attn_probs = torch.softmax(scores, dim=-1)
            result = torch.matmul(attn_probs, neighborhood_v).squeeze(1)
            out_reshaped[:, r, c, :] = result

    print("PyTorch reference finished.")
    return out_reshaped.reshape(B, HD, H, W, D).to(v_th.device).to(q_th.dtype)


# --- Example Usage ---
if __name__ == '__main__':
    DTYPE = jnp.float32
    # DTYPE = jnp.bfloat16 # Good choice for TPUs

    key = jax.random.PRNGKey(0)
    BATCH = 2
    HEADS = 4
    H = 256   # Use smaller dimensions for quicker testing if needed
    W = 256
    HEAD_DIM = 64
    KERNEL_SIZE = 3

    q_key, k_key, v_key = jax.random.split(key, 3)
    q = jax.device_put(jax.random.normal(q_key, (BATCH, HEADS, H, W, HEAD_DIM), dtype=DTYPE))
    k = jax.device_put(jax.random.normal(k_key, (BATCH, HEADS, H, W, HEAD_DIM), dtype=DTYPE))
    v = jax.device_put(jax.random.normal(v_key, (BATCH, HEADS, H, W, HEAD_DIM), dtype=DTYPE))
    print(f"Using device: {q.device}")

    print("Running JAX VMAP Neighborhood Attention...")
    # JIT compile the vmap wrapper function
    compiled_na_vmap = jax.jit(neighborhood_attention_vmap, static_argnums=(-1))
    # output_vmap = compiled_na_vmap(q, k, v, H, W, KERNEL_SIZE)
    j2t_func = jax2torch(compiled_na_vmap)
    #output_vmap.block_until_ready() # Ensure execution finishes
    # print(f"VMAP output shape: {output_vmap.shape}, "
    #       f"min: {output_vmap.min()}, max: {output_vmap.max()}")
    # print("VMAP execution finished.")

    # # --- Verification against PyTorch Reference ---
    # print("\nComparing outputs...")
    # output_vmap_np = np.array(output_vmap)
    # output_vmap_th = torch.from_numpy(output_vmap_np)
    q_th = torch.from_numpy(np.array(q))
    k_th = torch.from_numpy(np.array(k))
    v_th = torch.from_numpy(np.array(v))
    for i in range(10):
        output_vmap_th = j2t_func(q_th, k_th, v_th, KERNEL_SIZE)
        
        # output_torch = torch_neighborhood_attention(q_th, k_th, v_th, KERNEL_SIZE)

    # print(f"PyTorch output shape: {output_torch.shape}, "
    #       f"min: {output_torch.min()}, max: {output_torch.max()}")

    # atol = 1e-2 if DTYPE != jnp.float32 else 1e-5
    # rtol = 1e-3 if DTYPE != jnp.float32 else 1e-4
    # try:
    #     output_torch_device = output_torch.to(output_vmap_th.device)
    #     torch.testing.assert_close(output_vmap_th, output_torch_device, atol=atol, rtol=rtol)
    #     print(f"✅ Outputs match! (atol={atol}, rtol={rtol})")
    # except AssertionError as e:
    #     print(f"❌ Outputs differ!")
    #     print(e)
    #     max_diff = torch.max(torch.abs(output_vmap_th - output_torch_device))
    #     print(f"Max difference: {max_diff.item()}")
