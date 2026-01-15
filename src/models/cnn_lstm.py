"""
CNN-LSTM model for predicting bubble counts from PIV velocity fields.

Architecture:
- CNN encoder: Extract spatial features from each frame
- LSTM: Model temporal evolution of features
- Output: Predict bubble_count_primary and bubble_count_secondary
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """
    CNN encoder for extracting spatial features from PIV frames.
    
    Input: (batch, channels, height, width)
    Output: (batch, feature_dim)
    """
    
    def __init__(
        self,
        input_channels: int = 2,  # u and v
        feature_dim: int = 128,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Final projection to feature_dim
        self.fc = nn.Linear(128, feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            features: (batch, feature_dim)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)  # (batch, 128, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 128)
        x = self.fc(x)  # (batch, feature_dim)
        
        return x


class CNNLSTM(nn.Module):
    """
    CNN-LSTM model for spatiotemporal bubble count prediction.
    
    Architecture:
    1. CNN encoder extracts spatial features from each frame
    2. LSTM processes sequence of spatial features
    3. Dense layers predict bubble counts
    """
    
    def __init__(
        self,
        input_channels: int = 2,  # u and v
        sequence_length: int = 20,
        height: int = 22,
        width: int = 30,
        cnn_feature_dim: int = 128,
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
        lstm_bidirectional: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.cnn_feature_dim = cnn_feature_dim
        
        # CNN encoder for spatial feature extraction
        self.cnn_encoder = CNNEncoder(
            input_channels=input_channels,
            feature_dim=cnn_feature_dim,
        )
        
        # LSTM for temporal modeling
        lstm_input_dim = cnn_feature_dim
        lstm_output_dim = lstm_hidden_dim * (2 if lstm_bidirectional else 1)
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            bidirectional=lstm_bidirectional,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
        )
        
        # Output head for bubble count prediction
        self.fc1 = nn.Linear(lstm_output_dim, 256)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout)
        
        # Final output: bubble_count_primary and bubble_count_secondary
        self.fc_out = nn.Linear(128, 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, sequence_length, height, width, channels)
               or (batch, sequence_length, channels, height, width)
        Returns:
            predictions: (batch, 2) - [bubble_count_primary, bubble_count_secondary]
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Handle input format: convert to (batch * seq_len, channels, height, width)
        if x.dim() == 5 and x.shape[-1] == 2:
            # Input is (batch, seq_len, height, width, channels)
            x = x.permute(0, 1, 4, 2, 3)  # (batch, seq_len, channels, height, width)
        
        x = x.contiguous().view(batch_size * seq_len, *x.shape[2:])
        # Now x is (batch * seq_len, channels, height, width)
        
        # Extract spatial features for each frame
        cnn_features = self.cnn_encoder(x)  # (batch * seq_len, cnn_feature_dim)
        
        # Reshape back to sequence
        cnn_features = cnn_features.view(batch_size, seq_len, self.cnn_feature_dim)
        # Now cnn_features is (batch, seq_len, cnn_feature_dim)
        
        # Process sequence with LSTM
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)
        # lstm_out: (batch, seq_len, lstm_output_dim)
        # Use the last output
        lstm_features = lstm_out[:, -1, :]  # (batch, lstm_output_dim)
        
        # Predict bubble counts
        x = F.relu(self.fc1(lstm_features))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        predictions = self.fc_out(x)  # (batch, 2)
        
        return predictions


def create_model(
    sequence_length: int = 20,
    height: int = 22,
    width: int = 30,
    **kwargs,
) -> CNNLSTM:
    """
    Factory function to create CNN-LSTM model.
    
    Args:
        sequence_length: Number of frames per sequence
        height: PIV grid height
        width: PIV grid width
        **kwargs: Additional model parameters
    
    Returns:
        CNNLSTM model
    """
    model = CNNLSTM(
        sequence_length=sequence_length,
        height=height,
        width=width,
        **kwargs,
    )
    return model
