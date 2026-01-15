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
        sequences: np.ndarray,
        targets: np.ndarray,
        device: str = "cpu",
        augment: bool = False,
        temporal_shift_max: int = 2,
        noise_std: float = 0.0,
    ):
        """
        Args:
            sequences: (n_sequences, sequence_length, height, width, channels)
            targets: (n_sequences, 2)
            device: Device to move tensors to
            augment: Whether to apply data augmentation
            temporal_shift_max: Maximum temporal shift in frames (±)
            noise_std: Standard deviation for Gaussian noise injection (0 to disable)
        """
        self.sequences = sequences
        self.targets = targets
        self.device = device
        self.augment = augment
        self.temporal_shift_max = temporal_shift_max
        self.noise_std = noise_std
        
        # Validate shapes
        assert len(sequences) == len(targets), "Sequences and targets must have same length"
        assert targets.shape[1] == 2, "Targets must have 2 columns (primary, secondary)"
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def _apply_temporal_shift(
        self, sequence: np.ndarray, shift: int
    ) -> np.ndarray:
        """Apply temporal shift to sequence."""
        seq_len = sequence.shape[0]
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
            sequence: (sequence_length, channels, height, width)
            target: (2,) - [bubble_count_primary, bubble_count_secondary]
        """
        sequence = self.sequences[idx].copy()  # (sequence_length, height, width, channels)
        target = self.targets[idx]  # (2,)
        
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
