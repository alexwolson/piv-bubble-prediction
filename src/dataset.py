"""
PyTorch Dataset class for PIV → bubble count sequences.
"""

from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class PIVBubbleDataset(Dataset):
    """
    Dataset for PIV velocity field sequences and bubble count targets.
    
    Args:
        sequences: PIV velocity sequences (n_sequences, sequence_length, height, width, channels)
        targets: Bubble count targets (n_sequences, 2) - [primary, secondary]
        device: Device to move tensors to
        augment: Whether to apply data augmentation
        temporal_shift_max: Maximum temporal shift in frames (±)
        noise_std: Standard deviation for Gaussian noise injection (0 to disable)
    """
    
    def __init__(
        self,
        experiments: list,
        sequence_length: int = 20,
        stride: int = 1,
        device: str = "cpu",
        augment: bool = False,
        temporal_shift_max: int = 2,
        noise_std: float = 0.0,
    ):
        """
        Args:
            experiments: List of experiment dictionaries, each containing:
                - frames: (n_frames, height, width, channels)
                - targets: (n_sensor_rows, 2)
                - alignment_indices: (n_frames,)
            sequence_length: Number of frames per sequence
            stride: Stride for sliding window
            device: Device to move tensors to
            augment: Whether to apply data augmentation
            temporal_shift_max: Maximum temporal shift in frames (±)
            noise_std: Standard deviation for Gaussian noise injection (0 to disable)
        """
        self.experiments = experiments
        self.sequence_length = sequence_length
        self.stride = stride
        self.device = device
        self.augment = augment
        self.temporal_shift_max = temporal_shift_max
        self.noise_std = noise_std
        
        # Build index mapping
        self.indices = []  # List of (experiment_idx, start_frame_idx)
        
        for exp_idx, exp in enumerate(experiments):
            frames = exp["frames"]
            targets = exp["targets"]
            alignment_indices = exp["alignment_indices"]
            
            n_frames = len(frames)
            n_sensor_rows = len(targets)
            
            # Identify valid sequences
            for start_idx in range(0, n_frames - sequence_length + 1, stride):
                end_idx = start_idx + sequence_length
                last_frame_idx = end_idx - 1
                
                # Check alignment validity
                if last_frame_idx >= len(alignment_indices):
                    continue
                    
                sensor_row_idx = alignment_indices[last_frame_idx]
                if sensor_row_idx < 0 or sensor_row_idx >= n_sensor_rows:
                    continue
                
                self.indices.append((exp_idx, start_idx))
        
        if not self.indices:
            raise ValueError("No valid sequences found in any experiment")
            
    def __len__(self) -> int:
        return len(self.indices)
    
    def _apply_temporal_shift(
        self, sequence: np.ndarray, shift: int
    ) -> np.ndarray:
        """Apply temporal shift to sequence."""
        # Note: In lazy loading, we might want to slice differently instead of zero-padding
        # but to keep behavior identical to original implementation, we'll shift the extracted slice.
        # Ideally, we would slice [start+shift : end+shift] but that requires bounds checking
        # against the raw experiment data.
        # For now, let's keep the original image-based shift logic which zero-pads.
        if shift == 0:
            return sequence
        
        if shift > 0:
            # Shift forward: pad with zeros at start, remove from end
            shifted = np.zeros_like(sequence)
            shifted[shift:] = sequence[:-shift]
        else:
            # Shift backward: pad with zeros at end, remove from start
            shifted = np.zeros_like(sequence)
            shifted[:shift] = sequence[-shift:]
        
        return shifted
    
    def _apply_noise(self, sequence: np.ndarray) -> np.ndarray:
        """Apply Gaussian noise to sequence."""
        if self.noise_std <= 0:
            return sequence
        
        noise = np.random.normal(0, self.noise_std, sequence.shape).astype(sequence.dtype)
        return sequence + noise
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            sequence: (sequence_length, channels, height, width) [Torch Tensor]
            target: (2,) - [bubble_count_primary, bubble_count_secondary] [Torch Tensor]
        """
        exp_idx, start_idx = self.indices[idx]
        exp = self.experiments[exp_idx]
        
        # Extract sequence
        end_idx = start_idx + self.sequence_length
        # Returns (sequence_length, height, width, channels)
        # Using .copy() to ensure we have a writable copy and not just a view
        sequence = exp["frames"][start_idx:end_idx].copy()
        
        # Get target
        last_frame_idx = end_idx - 1
        sensor_row_idx = exp["alignment_indices"][last_frame_idx]
        target = exp["targets"][sensor_row_idx].copy()
        
        # Apply data augmentation if enabled
        if self.augment:
            # Temporal shift
            if self.temporal_shift_max > 0:
                shift = np.random.randint(-self.temporal_shift_max, self.temporal_shift_max + 1)
                sequence = self._apply_temporal_shift(sequence, shift)
            
            # Noise injection
            if self.noise_std > 0:
                sequence = self._apply_noise(sequence)
        
        # Convert to torch tensors
        # Convert from (seq_len, height, width, channels) to (seq_len, channels, height, width)
        sequence = torch.from_numpy(sequence).float()
        sequence = sequence.permute(0, 3, 1, 2)  # (sequence_length, channels, height, width)
        
        target = torch.from_numpy(target).float()
        
        return sequence, target
