import os
import pandas as pd
import numpy as np
import pickle
import torch
from torch_geometric.data import Data, Dataset
from typing import Optional, Tuple

class PEMSBayDataset(Dataset):
    def __init__(self, root_dir: str, window_size: int = 12, pred_horizon: int = 12, 
                 transform=None, pre_transform=None):
        """
        PEMS-BAY dataset loader
        
        Args:
            root_dir (str): Directory containing the dataset files
            window_size (int): Number of time steps to use as input
            pred_horizon (int): Number of time steps to predict
            transform: Transform to be applied to the data
            pre_transform: Transform to be applied to the data before saving
        """
        self.root_dir = root_dir
        self.window_size = window_size
        self.pred_horizon = pred_horizon
        self.data = None
        self.adj_mx = None
        self.meta_data = None
        super().__init__(root_dir, transform, pre_transform)
        
        # Load the data
        self._load_data()
        
    def _load_data(self):
        """Load all dataset files"""
        # Load traffic data
        data_path = os.path.join(self.root_dir, 'PEMS-BAY.csv')
        self.data = pd.read_csv(data_path)
        
        # If there's a datetime column, use it as index and drop it from features
        if 'timestamp' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data.set_index('timestamp', inplace=True)
        
        # Convert data to float values only
        self.data = self.data.select_dtypes(include=[np.number])
        
        # Load adjacency matrix
        adj_path = os.path.join(self.root_dir, 'adj_mx_bay.pkl')
        with open(adj_path, 'rb') as f:
            loaded_data = pickle.load(f, encoding='latin1')
            print("Loaded data type:", type(loaded_data))
            print("Loaded data length:", len(loaded_data))
            print("First element type:", type(loaded_data[0]))
            print("First element shape:", np.array(loaded_data[0]).shape)
            
            # Extract the adjacency matrix from the loaded data
            self.adj_mx = np.array(loaded_data[2])  # Try the third element
            print("Adjacency matrix shape:", self.adj_mx.shape)
            
        # Try to load metadata
        try:
            meta_path = os.path.join(self.root_dir, 'PEMS-BAY-META.csv')
            self.meta_data = pd.read_csv(meta_path)
        except FileNotFoundError:
            print("Metadata file not found. Continuing without metadata.")
            
    def _create_samples(self) -> list:
        """Create sliding window samples from the time series data"""
        samples = []
        values = self.data.values  # Convert to numpy for faster operations
        
        for i in range(len(values) - self.window_size - self.pred_horizon + 1):
            x = values[i:i+self.window_size]
            y = values[i+self.window_size:i+self.window_size+self.pred_horizon]
            samples.append((x, y))
            
        return samples
    
    def len(self) -> int:
        """Return the number of samples"""
        return len(self.data) - self.window_size - self.pred_horizon + 1
    
    def get(self, idx: int) -> Data:
        """Get a single sample as a PyG Data object"""
        # Get the sample
        samples = self._create_samples()
        x, y = samples[idx]
        
        # Convert inputs to numpy arrays first
        x = np.array(x).astype(np.float32)  # [window_size, num_nodes]
        y = np.array(y).astype(np.float32)  # [pred_horizon, num_nodes]
        
        # Transpose to get [num_nodes, window_size]
        x = x.transpose()
        y = y.transpose()
        
        # Create a fully connected graph for now
        num_nodes = x.shape[0]
        rows, cols = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                rows.append(i)
                cols.append(j)
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        
        # Convert to tensors
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=torch.ones(len(rows), dtype=torch.float),
            y=y
        )
        
        return data
    
    def simulate_sensor_failures(self, data: Data, failure_rate: float = 0.1) -> Data:
        """Simulate random sensor failures by masking values"""
        mask = torch.rand_like(data.x) > failure_rate
        data.x = data.x * mask
        return data
    
    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the graph"""
        return self.adj_mx.shape[0]
    
    @property
    def num_features(self) -> int:
        """Return the number of features per node"""
        return self.window_size
    
    @property
    def num_edge_features(self) -> int:
        """Return the number of edge features"""
        return 1  # Basic adjacency only