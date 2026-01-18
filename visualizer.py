import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
from transformers.image_utils import ChannelDimension
from matplotlib.colors import to_rgba
import matplotlib.gridspec as gridspec
from PIL import Image
import hdbscan
import spacy
from spacy.matcher import Matcher
from typing import List, Dict, Tuple, Optional, Any
import argparse
import os
from dataclasses import dataclass
from collections import defaultdict
import colorsys
from src.model.processor import Qwen2_VL_process_fn
from sklearn.cluster import DBSCAN

from transformers import AutoTokenizer, AutoModel

def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch

# ============ Data Classes ============
@dataclass
class ClusterInfo:
    """Information about a single cluster."""
    token_indices: torch.Tensor
    cluster_ids: torch.Tensor
    num_clusters: int
    cluster_mapping: Dict[int, int]
    original_labels: np.ndarray
    
@dataclass
class TextSpanInfo:
    """Information about text spans"""
    token_indices: torch.Tensor
    span_ids: torch.Tensor
    num_spans: int
    token_to_span_map: torch.Tensor
    span_texts: List[str]
    span_char_offsets: List[Tuple[int, int]]
    
@dataclass
class CrossModalAttention:
    """Cross-modal attention weights between text tokens and image regions."""
    span_to_cluster_attn: torch.Tensor
    text_span_info: TextSpanInfo
    vision_cluster_info: ClusterInfo
    
@dataclass
class VisualizationData:
    """All data needed for visualization."""
    layer_idx: int
    vision_cluster_info: ClusterInfo
    text_span_info_words: TextSpanInfo
    text_span_info_spans: TextSpanInfo
    cross_modal_attn_words: Optional[CrossModalAttention]
    cross_modal_attn_spans: Optional[CrossModalAttention]
    hidden_states_vision: torch.Tensor
    hidden_states_text: torch.Tensor
    
# ============Color Utilities ============

def generate_distinct_colors(n: int, saturation: float = 0.7, value: float = 0.9) -> List[Tuple[float, float, float]]:
    """Generate n visually distinct colors using HSV color space"""
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    return colors


def get_cluster_colormap(num_clusters: int) -> Dict[int, Tuple[float, float, float, float]]:
    """Create a colormap for clusters with RGBA values"""
    colors = generate_distinct_colors(num_clusters)
    colormap = {}
    for i in range(num_clusters):
        colormap[i] = (*colors[i], 0.6)  # Add alpha
    colormap[-1] = (0.5, 0.5, 0.5, 0.3)  # Gray for noise/unassigned
    return colormap

# ============Clustering Functions ============

def get_patch_coordinates(patch_idx: int, num_patch_per_row: int, patch_size: int) -> Tuple[float, float]:
    """Calculate center coordinates of a patch on the image"""
    row = patch_idx // num_patch_per_row
    col = patch_idx % num_patch_per_row
    center_x = col * patch_size + patch_size / 2
    center_y = row * patch_size + patch_size / 2
    return center_x, center_y


def compute_vision_distance_matrix(hidden_states: torch.Tensor, 
                                   num_patches_per_row: int, 
                                   patch_size: int,
                                   image_width: int, 
                                   image_height: int, 
                                   spatial_weight: float = 0.1) -> np.ndarray:
    """Compute distance matrix for HDBSCAN clustering"""
    num_tokens = hidden_states.size(0)
    device = hidden_states.device
    
    # Cosine distance
    hidden_norm = F.normalize(hidden_states, p=2, dim=-1)
    sim_matrix = hidden_norm @ hidden_norm.T
    cosine_distance = 1 - sim_matrix
    # cosine_distance = sim_matrix
    
    # Spatial distance
    coords = []
    for i in range(num_tokens):
        x, y = get_patch_coordinates(i, num_patches_per_row, patch_size)
        coords.append([x, y])
    coords = torch.tensor(coords, dtype=torch.float, device=device)
    
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    spatial_distance = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
    max_dist = torch.sqrt(torch.tensor(image_width ** 2 + image_height ** 2, dtype=torch.float, device=device))
    spatial_distance_norm = spatial_distance / max_dist
    print("spatial_distance_norm:", spatial_distance_norm.min().item(), spatial_distance_norm.max().item())
    print("cosine_distance:", cosine_distance.min().item(), cosine_distance.max().item())
    
    total_dist = cosine_distance + spatial_weight * spatial_distance_norm
    # total_dist = cosine_distance
    return total_dist.float().cpu().numpy()
    
    # num_tokens = hidden_states.size(0)
    # device = hidden_states.device
    
    # # MSE distance (Euclidean distance squared)
    # # Tính khoảng cách Euclidean bình phương giữa các hidden states
    # diff = hidden_states.unsqueeze(0) - hidden_states.unsqueeze(1)  # [num_tokens, num_tokens, hidden_dim]
    # mse_distance = (diff ** 2).mean(dim=-1)  # Mean squared error
    
    # # Hoặc nếu muốn dùng Euclidean distance thuần (không squared):
    # # euclidean_distance = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
    
    # # Normalize MSE distance về [0, 1] để cân bằng với spatial distance
    # mse_distance_norm = mse_distance / (mse_distance.max() + 1e-8)
    
    # # Spatial distance
    # coords = []
    # for i in range(num_tokens):
    #     x, y = get_patch_coordinates(i, num_patches_per_row, patch_size)
    #     coords.append([x, y])
    # coords = torch.tensor(coords, dtype=torch.float, device=device)
    
    # diff_spatial = coords.unsqueeze(0) - coords.unsqueeze(1)
    # spatial_distance = torch.sqrt((diff_spatial ** 2).sum(dim=-1) + 1e-8)
    # max_dist = torch.sqrt(torch.tensor(image_width ** 2 + image_height ** 2, dtype=torch.float, device=device))
    # spatial_distance_norm = spatial_distance / max_dist
    
    # print("spatial_distance_norm:", spatial_distance_norm.min().item(), spatial_distance_norm.max().item())
    # print("mse_distance_norm:", mse_distance_norm.min().item(), mse_distance_norm.max().item())
    
    # total_dist = mse_distance_norm + spatial_weight * spatial_distance_norm
    # # total_dist = mse_distance_norm  # Nếu chỉ dùng MSE
    # return total_dist.float().cpu().numpy()

def cluster_vision_tokens(hidden_states: torch.Tensor,
                         num_patches_per_row: int,
                         patch_size: int,
                         image_width: int,
                         image_height: int,
                         min_cluster_size: int = 8) -> np.ndarray:
    """Cluster vision tokens using HDBSCAN"""
    if hidden_states.size(0) < min_cluster_size:
        return np.zeros(hidden_states.size(0), dtype=np.int32)
    
    distance_matrix = compute_vision_distance_matrix(
        hidden_states, num_patches_per_row, patch_size,
        image_width, image_height, spatial_weight=0.1
    )
    
    # Ensure symmetry and non-negativity
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    distance_matrix = np.maximum(distance_matrix, 0)
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = distance_matrix.astype(np.float64)
    D = distance_matrix.copy()
    D = D[np.triu_indices_from(D, k=1)]
    eps = np.percentile(D, 2)
    
    print(f"Using eps = {eps:.4f}")

    clusterer = DBSCAN(
        eps=eps,
        min_samples=4,
        metric="precomputed"
    )
    
    # clusterer = hdbscan.HDBSCAN(
    #     min_cluster_size=min_cluster_size,
    #     metric='precomputed',
    #     allow_single_cluster=True,
    #     approx_min_span_tree=True,
    # )
    cluster_labels = clusterer.fit_predict(distance_matrix)
    # print(f"Number of clusters found: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")
    # print(f"Cluster labels: {cluster_labels}")
    
    if np.all(cluster_labels == -1):
        cluster_labels = np.zeros(hidden_states.size(0), dtype=np.int32)
    
    return cluster_labels

def prepare_vision_cluster_info(cluster_labels: np.ndarray, device: torch.device) -> Optional[ClusterInfo]:
    """Prepare cluster information for vision tokens"""
    cluster_labels = np.array(cluster_labels)
    valid_mask = cluster_labels >= 0
    
    if not np.any(valid_mask):
        return None
    
    valid_indices = np.where(valid_mask)[0]
    valid_clusters = cluster_labels[valid_mask]
    
    unique_clusters = np.unique(valid_clusters)
    cluster_mapping = {old: new for new, old in enumerate(unique_clusters)}
    remapped_clusters = np.array([cluster_mapping[c] for c in valid_clusters])
    
    return ClusterInfo(
        token_indices=torch.tensor(valid_indices, dtype=torch.long, device=device),
        cluster_ids=torch.tensor(remapped_clusters, dtype=torch.long, device=device),
        num_clusters=len(unique_clusters),
        cluster_mapping=cluster_mapping,
        original_labels=cluster_labels
    )


# ===================== Text Processing Functions =====================

def filter_overlapping_spans(spans):
    """Filter overlapping spans"""
    sorted_spans = sorted(spans, key=lambda s: (s[0], -s[1]))
    filtered = []
    words = []
    
    if not sorted_spans:
        return filtered, words
    
    current_span = sorted_spans[0]
    for next_span in sorted_spans[1:]:
        _, current_end, p = current_span
        _, next_end, _ = next_span
        if next_end <= current_end:
            continue
        filtered.append((current_span[0], current_span[1], current_span[2]))
        
        n_token = len(p)
        words.extend([(p[idx - 1].idx, p[idx].idx) for idx in range(1, n_token)])
        words.append((p[n_token - 1].idx, p[n_token - 1].idx + len(p[n_token - 1])))
        
        current_span = next_span
    
    filtered.append((current_span[0], current_span[1], current_span[2]))
    p = current_span[2]
    n_token = len(p)
    words.extend([(p[idx - 1].idx, p[idx].idx) for idx in range(1, n_token)])
    words.append((p[n_token - 1].idx, p[n_token - 1].idx + len(p[n_token - 1])))
    
    return filtered, words


def get_spans_offsets_with_text(text, nlp, matcher) -> Tuple[List[Tuple[int, int, str]], List[Tuple[int, int, str]]]:
    """Extract spans and words with their text content"""
    disabled_components = ["ner", "lemmatizer"]
    print(f"Processing text: {text}")  # Debug print
    text = text[0]
    doc = nlp(text)
    spans_with_offsets = []
    
    # Verb phrases
    vps = matcher(doc)
    for _, start, end in vps:
        vp = doc[start:end]
        spans_with_offsets.append((vp.start_char, vp.end_char, vp))
    
    # Noun chunks
    ncs = doc.noun_chunks
    spans_with_offsets.extend([(nc.start_char, nc.end_char, nc) for nc in ncs])
    
    unique_spans, unique_words = filter_overlapping_spans(spans_with_offsets)
    
    # Convert to tuples with text
    spans_result = [(s[0], s[1], text[s[0]:s[1]]) for s in unique_spans]
    words_result = [(w[0], w[1], text[w[0]:w[1]]) for w in unique_words]
    print(f"Span result: {spans_result}")
    print(f"Word result: {words_result}")
    
    return spans_result, words_result


def prepare_text_span_info(offset_mapping: torch.Tensor, 
                          span_offsets: List[Tuple[int, int, str]],
                          original_text: str,
                          device: torch.device) -> Optional[TextSpanInfo]:
    """Prepare text span information for visualization"""
    num_spans = len(span_offsets)
    if num_spans == 0:
        return None
    
    offset_mapping = offset_mapping[0]
    
    span_starts = torch.tensor([s[0] for s in span_offsets], dtype=torch.long, device=device)
    span_ends = torch.tensor([s[1] for s in span_offsets], dtype=torch.long, device=device)
    
    offsets_start = offset_mapping[:, 0].unsqueeze(1)
    offsets_end = offset_mapping[:, 1].unsqueeze(1)
    
    span_starts_exp = span_starts.unsqueeze(0)
    span_ends_exp = span_ends.unsqueeze(0)
    # print(f"offset mapping: {offset_mapping}")
    
    # print(f"offsets_start: {offsets_start}")
    # print(f"offsets_end: {offsets_end}")
    # print(f"span_starts_exp: {span_starts_exp}")
    # print(f"span_ends_exp: {span_ends_exp}")
    
    token_in_span_map = (offsets_start + 1 >= span_starts_exp) & (offsets_end <= span_ends_exp)
    
    if not token_in_span_map.any():
        return None
    
    nonzero_indices = token_in_span_map.nonzero(as_tuple=False)
    token_indices = nonzero_indices[:, 0]
    span_ids = nonzero_indices[:, 1]
    
    span_texts = [s[2] for s in span_offsets]
    span_char_offsets = [(s[0], s[1]) for s in span_offsets]
    
    return TextSpanInfo(
        token_indices=token_indices,
        span_ids=span_ids,
        num_spans=num_spans,
        token_to_span_map=token_in_span_map,
        span_texts=span_texts,
        span_char_offsets=span_char_offsets
    )


# ===================== Cross-Modal Attention =====================

def compute_cross_modal_attention(text_to_vision_attn: torch.Tensor,
                                 text_span_info: TextSpanInfo,
                                 vision_cluster_info: ClusterInfo) -> Optional[CrossModalAttention]:
    """Compute cross-modal attention weights between text spans and vision clusters"""
    if text_to_vision_attn is None or text_span_info is None or vision_cluster_info is None:
        return None
    
    device = text_to_vision_attn.device
    num_text_spans = text_span_info.num_spans
    num_vision_clusters = vision_cluster_info.num_clusters
    
    token_to_span_map = text_span_info.token_to_span_map.float()
    
    vision_cluster_labels = vision_cluster_info.original_labels
    cluster_mapping = vision_cluster_info.cluster_mapping
    
    num_vision_tokens = text_to_vision_attn.size(1)
    
    vision_to_cluster_map = torch.zeros((num_vision_tokens, num_vision_clusters), device=device, dtype=text_to_vision_attn.dtype)
    for v_idx in range(num_vision_tokens):
        orig_cluster = vision_cluster_labels[v_idx]
        if orig_cluster >= 0 and orig_cluster in cluster_mapping:
            new_cluster = cluster_mapping[orig_cluster]
            vision_to_cluster_map[v_idx, new_cluster] = 1.0
    
    # Aggregate attention
    attn_to_clusters = text_to_vision_attn @ vision_to_cluster_map
    token_to_span_map = token_to_span_map.to(device=device, dtype=text_to_vision_attn.dtype)
    span_to_cluster_attn = token_to_span_map.T @ attn_to_clusters
    
    # Normalize
    total_attn = span_to_cluster_attn.sum()
    if total_attn > 1e-8:
        span_to_cluster_attn = span_to_cluster_attn / total_attn
    
    return CrossModalAttention(
        span_to_cluster_attn=span_to_cluster_attn,
        text_span_info=text_span_info,
        vision_cluster_info=vision_cluster_info
    )
    
# ===================== Visualization Functions =====================

class ClusterVisualizer:
    """Visualizer for vision-text clustering and cross-modal attention"""
    
    def __init__(self, figsize: Tuple[int, int] = (20, 16)):
        self.figsize = figsize
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        
        # Add verb phrase pattern
        VERB_PHRASE_PATTERN = [
            {"POS": "AUX", "OP": "*"},
            {"POS": "ADV", "OP": "*"},
            {"POS": "VERB", "OP": "+"},
            {"POS": "ADV", "OP": "*"},
        ]
        self.matcher.add("VERB_PHRASE", [VERB_PHRASE_PATTERN])
    
    def visualize_vision_clusters(self,
                                  ax: plt.Axes,
                                  image: Image.Image,
                                  cluster_labels: np.ndarray,
                                  num_patches_per_row: int,
                                  patch_size: int,
                                  title: str = "Vision Clusters"):
        """Visualize vision clusters on the original image"""
        ax.imshow(image)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        colormap = get_cluster_colormap(num_clusters + 1)
        
        # Create mapping for remapped cluster IDs
        unique_clusters = sorted(set(cluster_labels))
        cluster_remap = {c: i for i, c in enumerate(unique_clusters) if c >= 0}
        
        img_width, img_height = image.size
        scale_x = img_width / (num_patches_per_row * patch_size)
        scale_y = img_height / (num_patches_per_row * patch_size)
        
        for patch_idx, cluster_id in enumerate(cluster_labels):
            row = patch_idx // num_patches_per_row
            col = patch_idx % num_patches_per_row
            
            x = col * patch_size * scale_x
            y = row * patch_size * scale_y
            w = patch_size * scale_x
            h = patch_size * scale_y
            
            if cluster_id >= 0:
                color = colormap.get(cluster_remap.get(cluster_id, 0), colormap[-1])
            else:
                color = colormap[-1]
            
            rect = mpatches.Rectangle(
                (x, y), w, h,
                linewidth=1,
                edgecolor='white',
                facecolor=color
            )
            ax.add_patch(rect)
            
            # Add cluster label in center
            if cluster_id >= 0:
                ax.text(
                    x + w/2, y + h/2,
                    str(cluster_id),
                    ha='center', va='center',
                    fontsize=6, color='white',
                    fontweight='bold'
                )
        
        # Create legend
        legend_patches = []
        for cluster_id in sorted(set(cluster_labels)):
            if cluster_id >= 0:
                color = colormap.get(cluster_remap.get(cluster_id, 0), colormap[-1])
                legend_patches.append(
                    mpatches.Patch(color=color, label=f'Cluster {cluster_id}')
                )
        
        if legend_patches:
            ax.legend(handles=legend_patches, loc='upper left', fontsize=8)
        
        ax.axis('off')
    
    def visualize_text_spans(self,
                            ax: plt.Axes,
                            text: str,
                            span_info: TextSpanInfo,
                            span_type: str = "spans",
                            title: str = "Text Spans"):
        """Visualize text spans with colored backgrounds"""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if span_info is None:
            ax.text(0.5, 0.5, "No spans detected", ha='center', va='center', fontsize=12)
            return
        
        num_spans = span_info.num_spans
        colormap = get_cluster_colormap(num_spans)
        
        # Create annotated text
        char_to_span = {}
        for i, (start, end) in enumerate(span_info.span_char_offsets):
            for c in range(start, end):
                char_to_span[c] = i
        
        # Render text with colored backgrounds
        y_pos = 0.9
        x_pos = 0.05
        max_width = 0.9
        line_height = 0.08
        char_width = 0.012
        
        current_x = x_pos
        current_y = y_pos
        
        i = 0
        while i < len(text):
            # Check if we need a new line
            if current_x > max_width:
                current_x = x_pos
                current_y -= line_height
            
            char = text[i]
            
            if i in char_to_span:
                span_id = char_to_span[i]
                color = colormap[span_id]
                
                # Find the end of this span's contiguous characters
                j = i
                while j < len(text) and j in char_to_span and char_to_span[j] == span_id:
                    j += 1
                
                span_text = text[i:j]
                text_len = len(span_text) * char_width
                
                # Check if we need a new line
                if current_x + text_len > max_width:
                    current_x = x_pos
                    current_y -= line_height
                
                # Draw background
                rect = FancyBboxPatch(
                    (current_x - 0.005, current_y - 0.02),
                    text_len + 0.01, 0.05,
                    boxstyle="round,pad=0.01",
                    facecolor=color,
                    edgecolor='none'
                )
                ax.add_patch(rect)
                
                # Draw text
                ax.text(current_x, current_y, span_text, fontsize=10, va='center')
                
                # Add span label below
                ax.text(
                    current_x + text_len/2, current_y - 0.035,
                    f"[{span_id}]",
                    fontsize=7, ha='center', va='top',
                    color=color[:3]
                )
                
                current_x += text_len + 0.01
                i = j
            else:
                # Regular character
                ax.text(current_x, current_y, char, fontsize=10, va='center')
                current_x += char_width
                i += 1
        
        # Legend
        legend_y = 0.15
        ax.text(0.05, legend_y, "Legend:", fontsize=10, fontweight='bold')
        
        for span_id, span_text in enumerate(span_info.span_texts[:10]):  # Limit to 10
            legend_y -= 0.05
            color = colormap[span_id]
            rect = FancyBboxPatch(
                (0.05, legend_y - 0.015), 0.03, 0.03,
                boxstyle="round,pad=0.005",
                facecolor=color,
                edgecolor='none'
            )
            ax.add_patch(rect)
            ax.text(0.1, legend_y, f"[{span_id}]: {span_text[:30]}...", fontsize=8, va='center')
    
    def visualize_cross_modal_attention(self,
                                       ax: plt.Axes,
                                       cross_modal_attn: CrossModalAttention,
                                       image: Image.Image,
                                       text: str,
                                       num_patches_per_row: int,
                                       patch_size: int,
                                       top_k: int = 3,
                                       title: str = "Cross-Modal Attention"):
        """Visualize cross-modal attention between text spans and vision clusters"""
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if cross_modal_attn is None:
            ax.text(0.5, 0.5, "No cross-modal attention available", ha='center', va='center')
            ax.axis('off')
            return
        
        attn_matrix = cross_modal_attn.span_to_cluster_attn.float().cpu().numpy()
        text_span_info = cross_modal_attn.text_span_info
        vision_cluster_info = cross_modal_attn.vision_cluster_info
        
        num_text_spans = attn_matrix.shape[0]
        num_vision_clusters = attn_matrix.shape[1]
        
        # Create layout: text spans on top, image on bottom
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.axis('off')
        
        # Colormap
        text_colormap = get_cluster_colormap(num_text_spans)
        vision_colormap = get_cluster_colormap(num_vision_clusters)
        
        # Draw text spans at top
        text_span_positions = []
        span_width = 0.8 / max(num_text_spans, 1)
        for i in range(num_text_spans):
            x = 0.1 + i * span_width + span_width / 2
            y = 0.95
            text_span_positions.append((x, y))
            
            color = text_colormap[i]
            rect = FancyBboxPatch(
                (x - span_width/2 + 0.01, y - 0.03),
                span_width - 0.02, 0.06,
                boxstyle="round,pad=0.01",
                facecolor=color,
                edgecolor='black',
                linewidth=1
            )
            ax.add_patch(rect)
            
            # Truncate span text
            span_text = text_span_info.span_texts[i][:15] + "..." if len(text_span_info.span_texts[i]) > 15 else text_span_info.span_texts[i]
            ax.text(x, y, span_text, ha='center', va='center', fontsize=7, fontweight='bold')
        
        # Draw vision clusters at bottom (as circles)
        vision_cluster_positions = []
        cluster_width = 0.8 / max(num_vision_clusters, 1)
        for i in range(num_vision_clusters):
            x = 0.1 + i * cluster_width + cluster_width / 2
            y = 0.15
            vision_cluster_positions.append((x, y))
            
            color = vision_colormap[i]
            circle = mpatches.Circle((x, y), 0.04, facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(circle)
            ax.text(x, y, f"V{i}", ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Draw attention connections
        for text_idx in range(num_text_spans):
            # Get top-k vision clusters for this text span
            attn_row = attn_matrix[text_idx]
            top_indices = np.argsort(attn_row)[-top_k:][::-1]
            
            for rank, vision_idx in enumerate(top_indices):
                attn_weight = attn_row[vision_idx]
                if attn_weight < 1e-6:
                    continue
                
                tx, ty = text_span_positions[text_idx]
                vx, vy = vision_cluster_positions[vision_idx]
                
                # Line width based on attention weight
                linewidth = 1 + 4 * (attn_weight / attn_matrix.max())
                alpha = 0.3 + 0.7 * (attn_weight / attn_matrix.max())
                
                # Color gradient from text to vision
                text_color = text_colormap[text_idx][:3]
                
                ax.plot(
                    [tx, vx], [ty - 0.03, vy + 0.04],
                    color=text_color,
                    linewidth=linewidth,
                    alpha=alpha,
                    linestyle='-',
                    zorder=1
                )
                
                # Add attention weight label at midpoint
                mid_x = (tx + vx) / 2
                mid_y = (ty - 0.03 + vy + 0.04) / 2
                ax.text(mid_x, mid_y, f"{attn_weight:.3f}", fontsize=6, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))
        
        # Add labels
        ax.text(0.5, 1.02, "Text Spans", ha='center', fontsize=10, fontweight='bold')
        ax.text(0.5, 0.05, "Vision Clusters", ha='center', fontsize=10, fontweight='bold')
    
    def visualize_attention_heatmap(self,
                                   ax: plt.Axes,
                                   cross_modal_attn: CrossModalAttention,
                                   title: str = "Attention Heatmap"):
        """Visualize cross-modal attention as a heatmap"""
        if cross_modal_attn is None:
            ax.text(0.5, 0.5, "No attention data", ha='center', va='center')
            ax.axis('off')
            return
        
        attn_matrix = cross_modal_attn.span_to_cluster_attn.float().cpu().numpy()
        text_span_info = cross_modal_attn.text_span_info
        
        im = ax.imshow(attn_matrix, cmap='YlOrRd', aspect='auto')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Labels
        ax.set_xlabel("Vision Cluster", fontsize=10)
        ax.set_ylabel("Text Span", fontsize=10)
        
        # Y-axis labels (text spans)
        y_labels = [f"{i}: {s[:10]}..." if len(s) > 10 else f"{i}: {s}" 
                   for i, s in enumerate(text_span_info.span_texts)]
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=8)
        
        # X-axis labels (vision clusters)
        ax.set_xticks(range(attn_matrix.shape[1]))
        ax.set_xticklabels([f"V{i}" for i in range(attn_matrix.shape[1])], fontsize=8)
        
        # Colorbar
        plt.colorbar(im, ax=ax, label='Attention Weight')
        
        # Add text annotations
        for i in range(attn_matrix.shape[0]):
            for j in range(attn_matrix.shape[1]):
                val = attn_matrix[i, j]
                if val > 0.01:
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=7,
                           color='white' if val > attn_matrix.max()/2 else 'black')
    
    def create_full_visualization(self,
                                 image: Image.Image,
                                 text: str,
                                 vis_data: VisualizationData,
                                 save_path: Optional[str] = None,
                                 show: bool = True):
        """Create a complete visualization for a single layer"""
        
        fig = plt.figure(figsize=(24, 20))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Extract data
        cluster_labels = vis_data.vision_cluster_info.original_labels if vis_data.vision_cluster_info else np.zeros(1)
        num_patches = int(np.sqrt(len(cluster_labels)))
        
        # Estimate patch size based on image dimensions
        img_width, img_height = image.size
        patch_size = img_width // num_patches
        
        # Top row: Vision clusters
        ax1 = fig.add_subplot(gs[0, 0])
        self.visualize_vision_clusters(
            ax1, image, cluster_labels, num_patches, patch_size,
            title=f"Vision Clusters (Layer {vis_data.layer_idx})"
        )
        
        # Top middle: Text spans (words)
        ax2 = fig.add_subplot(gs[0, 1])
        self.visualize_text_spans(
            ax2, text, vis_data.text_span_info_words, "words",
            title=f"Word-level Spans (Layer {vis_data.layer_idx})"
        )
        
        # Top right: Text spans (phrases)
        ax3 = fig.add_subplot(gs[0, 2])
        self.visualize_text_spans(
            ax3, text, vis_data.text_span_info_spans, "spans",
            title=f"Phrase-level Spans (Layer {vis_data.layer_idx})"
        )
        
        # Middle row: Cross-modal attention visualizations
        ax4 = fig.add_subplot(gs[1, :2])
        self.visualize_cross_modal_attention(
            ax4, vis_data.cross_modal_attn_words, image, text,
            num_patches, patch_size, top_k=3,
            title=f"Word-to-Vision Cross-Modal Attention (Layer {vis_data.layer_idx})"
        )
        
        ax5 = fig.add_subplot(gs[1, 2])
        self.visualize_attention_heatmap(
            ax5, vis_data.cross_modal_attn_words,
            title="Word-Vision Attention Heatmap"
        )
        
        # Bottom row: Phrase-level cross-modal attention
        ax6 = fig.add_subplot(gs[2, :2])
        self.visualize_cross_modal_attention(
            ax6, vis_data.cross_modal_attn_spans, image, text,
            num_patches, patch_size, top_k=3,
            title=f"Phrase-to-Vision Cross-Modal Attention (Layer {vis_data.layer_idx})"
        )
        
        ax7 = fig.add_subplot(gs[2, 2])
        self.visualize_attention_heatmap(
            ax7, vis_data.cross_modal_attn_spans,
            title="Phrase-Vision Attention Heatmap"
        )
        
        plt.suptitle(f"Vision-Text Clustering Visualization - Layer {vis_data.layer_idx}", 
                    fontsize=16, fontweight='bold', y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Saved visualization to {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
        
# ===================== Main Visualization Pipeline =====================

def extract_vision_hidden_states(hidden_states, num_vision_tokens, num_text_tokens):
    vision_hidden_list = []
    for layer_hidden in hidden_states:
        start_idx = -(num_vision_tokens + num_text_tokens)
        end_idx = -num_text_tokens if num_text_tokens > 0 else None
        vision_hidden = layer_hidden[0][start_idx:end_idx, :].detach()
        vision_hidden_list.append(vision_hidden)
    return vision_hidden_list

def extract_text_hidden_states(hidden_states, num_text_tokens, num_vision_tokens):
    text_hidden_list = []
    for layer_hidden in hidden_states:
        text_hidden = layer_hidden[0][-num_text_tokens:, :].detach()
        text_hidden_list.append(text_hidden)
        
    return text_hidden_list
def extract_attention_weights(attention_states, sample_idx, num_vision_tokens, num_text_tokens):
    attention_list = []
    for layer_attn in attention_states:
        if layer_attn is None:
            attention_list.append(None)
            continue
        
        if len(layer_attn.size()) == 4:
            attn = layer_attn[sample_idx].mean(dim=0)
        else:
            attn = layer_attn[sample_idx]
            
        text_start = -num_text_tokens if num_text_tokens > 0 else attn.size(0)
        vision_start = -(num_vision_tokens + num_text_tokens)
        vision_end = -num_text_tokens if num_text_tokens > 0 else None
        
        if num_text_tokens > 0:
            text_to_vision_attn = attn[text_start:, vision_start:vision_end].detach() # (num_text_tokens, num_vision_tokens)
        else:
            text_to_vision_attn = None
            
        attention_list.append(text_to_vision_attn)
    return attention_list

class VisionTextClusteringVisualizationPipeline:
    """
    Main pipeline for visualizing vision-text clustering and cross-modal attention
    
    Usage:
        pipeline = VisionTextClusteringVisualizationPipeline(model, tokenizer)
        pipeline.visualize(image_path, text, layers=[4, 8, 12], output_dir="./visualizations")
    """
    
    def __init__(self, 
                 model,
                 processor,
                 tokenizer,
                 patch_size: int = 28,
                 device: str = "cuda"):
        """
        Args:
            model: The vision-language model
            tokenizer: Tokenizer for the model
            patch_size: Patch size used by the vision encoder
            device: Device to run inference on
        """
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.patch_size = patch_size
        self.device = device
        
        self.visualizer = ClusterVisualizer()
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        
        VERB_PHRASE_PATTERN = [
            {"POS": "AUX", "OP": "*"},
            {"POS": "ADV", "OP": "*"},
            {"POS": "VERB", "OP": "+"},
            {"POS": "ADV", "OP": "*"},
        ]
        self.matcher.add("VERB_PHRASE", [VERB_PHRASE_PATTERN])
    
    def prepare_input(self, image, text: str) -> Dict[str, Any]:
        """Prepare model input from image path and text"""
        
        # This is a placeholder - you'll need to adapt this to your specific model
        # The actual implementation depends on your model's input format
        inputs = {
            'images': [image],
            'text': [text],
            # Add your model-specific preprocessing here
        }
        
        return inputs
    
    def extract_hidden_states_and_attention(self, inputs):
        """
        Extract hidden states and attention from model output
        
        Returns:
            vision_hidden_states: List of (num_vision_tokens, D) tensors per layer
            text_hidden_states: List of (num_text_tokens, D) tensors per layer
            attention_weights: List of attention tensors per layer
        """
        # Placeholder - implement based on your model's output format
        _, image_features, attention_matrix, output_hidden_states = self.model.encode_input(inputs)
        num_vision_tokens = image_features[0].size(0)
        # print(f"Number of vision tokens: {num_vision_tokens}")
        num_text_tokens = ((inputs['input_ids'] < 151643) | (inputs['input_ids'] > 151656)).sum(dim=1)
        # print(f"Number of text tokens: {num_text_tokens}")
        # print(f"Length of output_hidden_states: {len(output_hidden_states)}")
        # print(f"Shape of first layer hidden state: {output_hidden_states[0].shape}")
        
        text_hidden_list = extract_text_hidden_states(output_hidden_states, num_text_tokens=num_text_tokens[0], num_vision_tokens=num_vision_tokens)
        vision_hidden_list = extract_vision_hidden_states(output_hidden_states, num_vision_tokens=num_vision_tokens, num_text_tokens=num_text_tokens[0])
        attention_list = extract_attention_weights(attention_matrix, sample_idx=0, num_vision_tokens=num_vision_tokens, num_text_tokens=num_text_tokens[0])
        return vision_hidden_list, text_hidden_list, attention_list
        
    def process_layer(self,
                     layer_idx: int,
                     vision_hidden: torch.Tensor,
                     text_hidden: torch.Tensor,
                     text_to_vision_attn: torch.Tensor,
                     text: str,
                     offset_mapping: torch.Tensor,
                     image_width: int,
                     image_height: int,
                     min_cluster_size_vision: int = 8) -> VisualizationData:
        """Process a single layer to get visualization data"""
        
        device = vision_hidden.device
        num_vision_tokens = vision_hidden.size(0)
        num_patches_per_row = int(np.sqrt(num_vision_tokens))
        
        # Cluster vision tokens
        cluster_labels = cluster_vision_tokens(
            vision_hidden, num_patches_per_row, self.patch_size,
            image_width, image_height, min_cluster_size=min_cluster_size_vision
        )
        vision_cluster_info = prepare_vision_cluster_info(cluster_labels, device)
        
        # Get text spans
        spans_offsets, words_offsets = get_spans_offsets_with_text(text, self.nlp, self.matcher)
        
        # Prepare text span info
        text_span_info_words = prepare_text_span_info(offset_mapping, words_offsets, text, device)
        text_span_info_spans = prepare_text_span_info(offset_mapping, spans_offsets, text, device)
        
        # print(f"text_span_info_words: {text_span_info_words if text_span_info_words else 0} spans")
        # print(f"text_span_info_spans: {text_span_info_spans if text_span_info_spans else 0} spans")
        
        # Compute cross-modal attention
        cross_modal_attn_words = None
        cross_modal_attn_spans = None
        
        if text_to_vision_attn is not None and vision_cluster_info is not None:
            if text_span_info_words is not None:
                cross_modal_attn_words = compute_cross_modal_attention(
                    text_to_vision_attn, text_span_info_words, vision_cluster_info
                )
            
            if text_span_info_spans is not None:
                cross_modal_attn_spans = compute_cross_modal_attention(
                    text_to_vision_attn, text_span_info_spans, vision_cluster_info
                )
        
        return VisualizationData(
            layer_idx=layer_idx,
            vision_cluster_info=vision_cluster_info,
            text_span_info_words=text_span_info_words,
            text_span_info_spans=text_span_info_spans,
            cross_modal_attn_words=cross_modal_attn_words,
            cross_modal_attn_spans=cross_modal_attn_spans,
            hidden_states_vision=vision_hidden,
            hidden_states_text=text_hidden
        )
    
    def visualize(self,
                 image_path: str,
                 text: str,
                 layers: List[int],
                 output_dir: str = "./visualizations",
                 show: bool = False):
        """
        Run the full visualization pipeline
        
        Args:
            image_path: Path to the input image
            text: Input text
            layers: List of layer indices to visualize
            output_dir: Directory to save visualizations
            show: Whether to display visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((336, 336))
        img_width, img_height = image.size
        
        # Prepare input and run model
        inputs = self.prepare_input(image, text)

        inputs = Qwen2_VL_process_fn(inputs, self.processor)
        
        inputs = batch_to_device(inputs, self.device) 
        # Get model outputs (implement based on your model)
        vision_hidden_list, text_hidden_list, attention_list = self.extract_hidden_states_and_attention(inputs)
        # print("Extracted hidden states and attention from model.")
        # print(f"Number of layers with hidden states: {len(vision_hidden_list)}")
        # print(f"Number of layers with attention: {len(attention_list) if attention_list else 0}")
        # print(f"Number of vision tokens: {vision_hidden_list[0].size(0)}")
        # print(f"Number of text tokens: {text_hidden_list[0].size(0)}")
        # print(f"shape of attention (if available): {attention_list[0].shape if attention_list else 'N/A'}")
        # print(f"Shape of vision hidden (if available): {vision_hidden_list[0].shape if vision_hidden_list else 'N/A'}")
        # print(f"Shape of text hidden (if available): {text_hidden_list[0].shape if text_hidden_list else 'N/A'}")
        
        # Get offset mapping for text
        text = self.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
        
        offset_mapping = self.tokenizer(
            text, 
            return_offsets_mapping=True, 
            add_special_tokens=False, 
            return_tensors="pt",
            padding=True
        )["offset_mapping"].to(self.device)
        
        # Process each layer
        for layer_idx in layers:
            print(f"Processing layer {layer_idx}...")
            
            # Get hidden states and attention for this layer
            # This is a placeholder - implement based on your model
            vision_hidden = vision_hidden_list[layer_idx]
            text_hidden = text_hidden_list[layer_idx]
            attention = attention_list[layer_idx] if attention_list else None
            
            # For demonstration, create dummy data
            # Replace with actual model outputs
            num_patches = 16  # Example: 16x16 patches
            # vision_hidden = torch.randn(num_patches * num_patches, 768, device=self.device)
            # text_hidden = torch.randn(offset_mapping.size(0), 768, device=self.device)
            # attention = torch.rand(offset_mapping.size(0), num_patches * num_patches, device=self.device)
            attention = attention / attention.sum(dim=-1, keepdim=True)
            
            vis_data = self.process_layer(
                layer_idx=layer_idx,
                vision_hidden=vision_hidden,
                text_hidden=text_hidden,
                text_to_vision_attn=attention,
                text=text,
                offset_mapping=offset_mapping,
                image_width=img_width,
                image_height=img_height
            )
            
            save_path = os.path.join(output_dir, f"layer_{layer_idx}_visualization.png")
            self.visualizer.create_full_visualization(
                image=image,
                text=text,
                vis_data=vis_data,
                save_path=save_path,
                show=show
            )


# ===================== Standalone Visualization Functions =====================

def visualize_from_precomputed(
    image_path: str,
    text: str,
    vision_hidden_states: torch.Tensor,
    text_to_vision_attention: torch.Tensor,
    offset_mapping: torch.Tensor,
    layer_idx: int,
    patch_size: int = 28,
    min_cluster_size: int = 3,
    output_path: str = "visualization.png",
    show: bool = True
):
    """
    Create visualization from precomputed hidden states and attention
    
    Args:
        image_path: Path to the original image
        text: Original text input
        vision_hidden_states: (num_vision_tokens, D) tensor
        text_to_vision_attention: (num_text_tokens, num_vision_tokens) tensor
        offset_mapping: (num_text_tokens, 2) tensor with character offsets
        layer_idx: Layer index for labeling
        patch_size: Vision patch size
        min_cluster_size: Minimum cluster size for HDBSCAN
        output_path: Path to save visualization
        show: Whether to display the visualization
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    img_width, img_height = image.size
    
    # Initialize spacy
    nlp = spacy.load("en_core_web_sm")
    matcher = Matcher(nlp.vocab)
    VERB_PHRASE_PATTERN = [
        {"POS": "AUX", "OP": "*"},
        {"POS": "ADV", "OP": "*"},
        {"POS": "VERB", "OP": "+"},
        {"POS": "ADV", "OP": "*"},
    ]
    matcher.add("VERB_PHRASE", [VERB_PHRASE_PATTERN])
    
    device = vision_hidden_states.device
    num_vision_tokens = vision_hidden_states.size(0)
    num_patches_per_row = int(np.sqrt(num_vision_tokens))
    
    # Cluster vision tokens
    cluster_labels = cluster_vision_tokens(
        vision_hidden_states, num_patches_per_row, patch_size,
        img_width, img_height, min_cluster_size=min_cluster_size
    )
    vision_cluster_info = prepare_vision_cluster_info(cluster_labels, device)
    
    # Get text spans
    spans_offsets, words_offsets = get_spans_offsets_with_text(text, nlp, matcher)
    
    # Prepare text span info
    text_span_info_words = prepare_text_span_info(offset_mapping, words_offsets, text, device)
    text_span_info_spans = prepare_text_span_info(offset_mapping, spans_offsets, text, device)
    
    # Compute cross-modal attention
    cross_modal_attn_words = None
    cross_modal_attn_spans = None
    
    if text_to_vision_attention is not None and vision_cluster_info is not None:
        if text_span_info_words is not None:
            cross_modal_attn_words = compute_cross_modal_attention(
                text_to_vision_attention, text_span_info_words, vision_cluster_info
            )
        
        if text_span_info_spans is not None:
            cross_modal_attn_spans = compute_cross_modal_attention(
                text_to_vision_attention, text_span_info_spans, vision_cluster_info
            )
    
    vis_data = VisualizationData(
        layer_idx=layer_idx,
        vision_cluster_info=vision_cluster_info,
        text_span_info_words=text_span_info_words,
        text_span_info_spans=text_span_info_spans,
        cross_modal_attn_words=cross_modal_attn_words,
        cross_modal_attn_spans=cross_modal_attn_spans,
        hidden_states_vision=vision_hidden_states,
        hidden_states_text=None
    )
    
    visualizer = ClusterVisualizer()
    visualizer.create_full_visualization(
        image=image,
        text=text,
        vis_data=vis_data,
        save_path=output_path,
        show=show
    )


# ===================== Demo Function =====================

def demo_visualization():
    """
    Demo function showing how to use the visualization tools with dummy data
    """
    import tempfile
    
    # Create a dummy image
    dummy_image = Image.new('RGB', (448, 448), color=(100, 150, 200))
    
    # Add some patterns to make it interesting
    from PIL import ImageDraw
    draw = ImageDraw.Draw(dummy_image)
    for i in range(0, 448, 56):
        for j in range(0, 448, 56):
            color = ((i * 2) % 256, (j * 2) % 256, ((i + j)) % 256)
            draw.rectangle([i, j, i + 50, j + 50], fill=color)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        image_path = f.name
        dummy_image.save(image_path)
    
    # Sample text
    text = "A beautiful sunset over the mountain range with colorful clouds in the sky"
    
    # Create dummy hidden states and attention
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_patches = 16 * 16  # 16x16 patches
    hidden_dim = 768
    
    vision_hidden = torch.randn(num_patches, hidden_dim, device=device)
    
    # Create tokenizer-like offset mapping
    words = text.split()
    offset_mapping = []
    pos = 0
    for word in words:
        start = text.find(word, pos)
        end = start + len(word)
        offset_mapping.append([start, end])
        pos = end
    offset_mapping = torch.tensor(offset_mapping, device=device)
    
    num_text_tokens = len(offset_mapping)
    text_to_vision_attention = torch.rand(num_text_tokens, num_patches, device=device)
    text_to_vision_attention = text_to_vision_attention / text_to_vision_attention.sum(dim=-1, keepdim=True)
    
    # Run visualization
    output_path = "demo_visualization.png"
    visualize_from_precomputed(
        image_path=image_path,
        text=text,
        vision_hidden_states=vision_hidden,
        text_to_vision_attention=text_to_vision_attention,
        offset_mapping=offset_mapping,
        layer_idx=8,
        patch_size=28,
        min_cluster_size=3,
        output_path=output_path,
        show=False
    )
    
    print(f"Demo visualization saved to {output_path}")
    
    # Cleanup
    os.remove(image_path)
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Vision-Text Clustering")
    parser.add_argument("--demo", action="store_true", help="Run demo with dummy data")
    parser.add_argument("--image_path", type=str, help="Path to input image")
    parser.add_argument("--text", type=str, help="Input text")
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12], help="Layers to visualize")
    parser.add_argument("--output_dir", type=str, default="./visualizations", help="Output directory")
    parser.add_argument("--patch_size", type=int, default=28, help="Vision patch size")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_visualization()
    else:
        print("Please provide model and input data, or use --demo for demonstration")