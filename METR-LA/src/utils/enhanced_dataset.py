"""
Enhanced Dataset Handler with Missing Data Detection and Advanced Preprocessing
"""

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
import pickle
from typing import List, Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')


class MissingDataAnalyzer:
    """Analyze and handle missing data in traffic sensor data"""
    
    def __init__(self):
        self.missing_stats = {}
        self.sensor_reliability = {}
        
    def analyze_missing_data(self, data: np.ndarray, sensor_names: Optional[List[str]] = None) -> Dict:
        """
        Analyze missing data patterns in the dataset
        
        Args:
            data: Traffic data array [time_steps, sensors]
            sensor_names: Optional sensor names
            
        Returns:
            Dictionary with missing data statistics
        """
        if sensor_names is None:
            sensor_names = [f"Sensor_{i}" for i in range(data.shape[1])]
        
        # Detect missing data (NaN, zeros, or very small values)
        missing_mask = np.isnan(data) | (np.abs(data) < 1e-3)
        
        # Calculate missing data statistics
        total_missing = np.sum(missing_mask)
        total_values = data.size
        missing_percentage = (total_missing / total_values) * 100
        
        # Per-sensor missing data
        sensor_missing = np.sum(missing_mask, axis=0)
        sensor_missing_pct = (sensor_missing / data.shape[0]) * 100
        
        # Per-time missing data
        time_missing = np.sum(missing_mask, axis=1)
        time_missing_pct = (time_missing / data.shape[1]) * 100
        
        # Sensor reliability (percentage of valid data)
        sensor_reliability = 100 - sensor_missing_pct
        
        self.missing_stats = {
            'total_missing_count': int(total_missing),
            'total_missing_percentage': float(missing_percentage),
            'sensor_missing_counts': sensor_missing.tolist(),
            'sensor_missing_percentages': sensor_missing_pct.tolist(),
            'sensor_reliability_scores': sensor_reliability.tolist(),
            'time_missing_counts': time_missing.tolist(),
            'time_missing_percentages': time_missing_pct.tolist(),
            'most_reliable_sensors': np.argsort(sensor_reliability)[-10:].tolist(),
            'least_reliable_sensors': np.argsort(sensor_reliability)[:10].tolist()
        }
        
        self.sensor_reliability = dict(zip(sensor_names, sensor_reliability))
        
        return self.missing_stats
    
    def get_sensor_quality_groups(self) -> Dict[str, List[str]]:
        """Group sensors by data quality"""
        if not self.sensor_reliability:
            return {}
        
        high_quality = []
        medium_quality = []
        low_quality = []
        
        for sensor, reliability in self.sensor_reliability.items():
            if reliability >= 90:
                high_quality.append(sensor)
            elif reliability >= 70:
                medium_quality.append(sensor)
            else:
                low_quality.append(sensor)
        
        return {
            'high_quality': high_quality,
            'medium_quality': medium_quality,
            'low_quality': low_quality
        }


class AdvancedDataPreprocessor:
    """Advanced preprocessing with multiple imputation strategies"""
    
    def __init__(self, use_knn: bool = True):
        self.scalers = {}
        self.imputers = {}
        self.is_fitted = False
        self.use_knn = use_knn
        
    def fit_transform(self, data: np.ndarray, method: str = 'robust') -> np.ndarray:
        """
        Fit preprocessor and transform data
        
        Args:
            data: Input data [time_steps, sensors]
            method: Scaling method ('standard', 'robust', 'minmax')
            
        Returns:
            Preprocessed data
        """
        # Handle missing data first
        data_imputed = self._impute_missing_data(data, use_knn=self.use_knn)
        
        # Scale data
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        
        data_scaled = scaler.fit_transform(data_imputed)
        
        self.scalers['main'] = scaler
        self.is_fitted = True
        
        return data_scaled
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform new data using fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        data_imputed = self._impute_missing_data(data, use_knn=self.use_knn)
        return self.scalers['main'].transform(data_imputed)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        return self.scalers['main'].inverse_transform(data)
    
    def _impute_missing_data(self, data: np.ndarray, use_knn: bool = False) -> np.ndarray:
        """Impute missing data using multiple strategies"""
        # Create a copy to avoid modifying original data
        data_imputed = data.copy()
        
        # Replace NaN and very small values with NaN for consistent handling
        missing_mask = np.isnan(data_imputed) | (np.abs(data_imputed) < 1e-3)
        data_imputed[missing_mask] = np.nan
        
        # Strategy 1: Forward fill for short gaps
        df = pd.DataFrame(data_imputed)
        data_imputed = df.ffill(limit=3).values
        
        # Strategy 2: Backward fill
        df = pd.DataFrame(data_imputed)
        data_imputed = df.bfill(limit=3).values
        
        # Strategy 3: KNN imputation for remaining missing values (optional, slower)
        if use_knn and np.isnan(data_imputed).any():
            print("[IMPUTE] Using KNN imputation (this may take a few minutes)...")
            imputer = KNNImputer(n_neighbors=5, weights='distance')
            data_imputed = imputer.fit_transform(data_imputed)
            self.imputers['knn'] = imputer
        
        # Strategy 4: Use median for any remaining NaN values
        if np.isnan(data_imputed).any():
            col_medians = np.nanmedian(data_imputed, axis=0)
            for i in range(data_imputed.shape[1]):
                col_mask = np.isnan(data_imputed[:, i])
                data_imputed[col_mask, i] = col_medians[i]
        
        return data_imputed


class EnhancedPEMSBayDataset(Dataset):
    """
    Enhanced PEMS-BAY dataset with advanced preprocessing and missing data handling
    """
    
    def __init__(
        self,
        root_dir: str = 'data',
        sequence_length: int = 12,
        prediction_length: int = 12,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        preprocessing_method: str = 'robust',
        analyze_missing_data: bool = True
    ):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.preprocessing_method = preprocessing_method
        
        # Initialize components
        self.missing_analyzer = MissingDataAnalyzer()
        self.preprocessor = AdvancedDataPreprocessor()
        
        # Load and process data
        self.raw_data, self.adjacency_matrix, self.sensor_metadata = self._load_raw_data()
        
        if analyze_missing_data:
            print("[ANALYSIS] Analyzing missing data patterns...")
            self.missing_stats = self.missing_analyzer.analyze_missing_data(self.raw_data)
            self._print_missing_data_report()
        
        # Preprocess data
        print("[PREPROCESS] Preprocessing data...")
        self.processed_data = self.preprocessor.fit_transform(self.raw_data, self.preprocessing_method)
        
        # Create sequences
        print("[SEQUENCES] Creating sequences...")
        self.sequences = self._create_sequences()
        
        # Split data
        self.train_indices, self.val_indices, self.test_indices = self._split_data()
        
        print(f"[READY] Dataset ready: {len(self.sequences)} sequences")
        print(f"   Train: {len(self.train_indices)}, Val: {len(self.val_indices)}, Test: {len(self.test_indices)}")
    
    def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
        """Load raw traffic data, adjacency matrix, and metadata"""
        # Load traffic data
        data_path = os.path.join(self.root_dir, 'PEMS-BAY.csv')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        print(f"[LOAD] Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Check if first column contains timestamps or non-numeric data
        first_col_sample = str(df.iloc[0, 0])
        if any(char in first_col_sample for char in ['-', ':', ' ']) or df.columns[0].lower() in ['time', 'timestamp', 'date']:
            # First column is timestamp, skip it
            data = df.iloc[:, 1:].values
            print(f"   Detected timestamp column, using columns 1-{df.shape[1]-1}")
        else:
            # All columns are numeric
            data = df.values
        
        print(f"   Data shape: {data.shape}")
        
        # Load adjacency matrix
        adj_path = os.path.join(self.root_dir, 'adj_mx_bay.pkl')
        if os.path.exists(adj_path):
            try:
                with open(adj_path, 'rb') as f:
                    adj_data = pickle.load(f, encoding='latin1')
                    if isinstance(adj_data, list):
                        adjacency_matrix = adj_data[2]  # Assuming third element is the matrix
                    else:
                        adjacency_matrix = adj_data
                print(f"   Adjacency matrix shape: {adjacency_matrix.shape}")
            except Exception as e:
                print(f"   [WARN] Error loading adjacency matrix: {e}")
                print("   Creating identity matrix instead")
                adjacency_matrix = np.eye(data.shape[1])
        else:
            print("   [WARN] Adjacency matrix not found, creating identity matrix")
            adjacency_matrix = np.eye(data.shape[1])
        
        # Load metadata if available
        meta_path = os.path.join(self.root_dir, 'PEMS-BAY-META.csv')
        metadata = None
        if os.path.exists(meta_path):
            metadata = pd.read_csv(meta_path)
            print(f"   Metadata loaded: {len(metadata)} sensors")
        
        return data.astype(np.float32), adjacency_matrix.astype(np.float32), metadata
    
    def _print_missing_data_report(self):
        """Print detailed missing data analysis report"""
        stats = self.missing_stats
        
        print("\n[MISSING] Missing Data Analysis Report")
        print("=" * 50)
        print(f"Total missing values: {stats['total_missing_count']:,} ({stats['total_missing_percentage']:.2f}%)")
        
        # Sensor reliability
        reliability_scores = np.array(stats['sensor_reliability_scores'])
        print(f"\nSensor Reliability Statistics:")
        print(f"  Mean reliability: {np.mean(reliability_scores):.2f}%")
        print(f"  Median reliability: {np.median(reliability_scores):.2f}%")
        print(f"  Min reliability: {np.min(reliability_scores):.2f}%")
        print(f"  Max reliability: {np.max(reliability_scores):.2f}%")
        
        # Quality groups
        quality_groups = self.missing_analyzer.get_sensor_quality_groups()
        print(f"\nSensor Quality Groups:")
        print(f"  High quality (>=90%): {len(quality_groups.get('high_quality', []))} sensors")
        print(f"  Medium quality (70-90%): {len(quality_groups.get('medium_quality', []))} sensors")
        print(f"  Low quality (<70%): {len(quality_groups.get('low_quality', []))} sensors")
        
        # Most/least reliable sensors
        most_reliable = stats['most_reliable_sensors'][-5:]
        least_reliable = stats['least_reliable_sensors'][:5]
        
        print(f"\nTop 5 Most Reliable Sensors:")
        for idx in most_reliable:
            print(f"  Sensor {idx}: {reliability_scores[idx]:.2f}% reliability")
        
        print(f"\nTop 5 Least Reliable Sensors:")
        for idx in least_reliable:
            print(f"  Sensor {idx}: {reliability_scores[idx]:.2f}% reliability")
    
    def _create_sequences(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Create input-output sequences with missing data masks"""
        sequences = []
        
        for i in range(len(self.processed_data) - self.sequence_length - self.prediction_length + 1):
            # Input sequence
            x = self.processed_data[i:i + self.sequence_length].T  # [sensors, time_steps]
            
            # Output sequence
            y = self.processed_data[i + self.sequence_length:i + self.sequence_length + self.prediction_length].T
            
            # Missing data mask for input
            raw_x = self.raw_data[i:i + self.sequence_length].T
            missing_mask = np.isnan(raw_x) | (np.abs(raw_x) < 1e-3)
            
            sequences.append((x, y, missing_mask))
        
        return sequences
    
    def _split_data(self) -> Tuple[List[int], List[int], List[int]]:
        """Split data into train/validation/test sets"""
        n_sequences = len(self.sequences)
        
        train_size = int(n_sequences * self.train_ratio)
        val_size = int(n_sequences * self.val_ratio)
        
        indices = list(range(n_sequences))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        return train_indices, val_indices, test_indices
    
    def get_train_data(self) -> List[Data]:
        """Get training data as PyTorch Geometric Data objects"""
        return [self._create_pyg_data(i) for i in self.train_indices]
    
    def get_val_data(self) -> List[Data]:
        """Get validation data as PyTorch Geometric Data objects"""
        return [self._create_pyg_data(i) for i in self.val_indices]
    
    def get_test_data(self) -> List[Data]:
        """Get test data as PyTorch Geometric Data objects"""
        return [self._create_pyg_data(i) for i in self.test_indices]
    
    def _create_pyg_data(self, idx: int) -> Data:
        """Create PyTorch Geometric Data object"""
        x, y, missing_mask = self.sequences[idx]
        
        # Create edge index from adjacency matrix
        edge_index = self._adjacency_to_edge_index(self.adjacency_matrix)
        
        return Data(
            x=torch.tensor(x, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            missing_mask=torch.tensor(missing_mask, dtype=torch.bool)
        )
    
    def _adjacency_to_edge_index(self, adj_matrix: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """Convert adjacency matrix to edge index format"""
        # Use threshold to create edges (only keep strong connections)
        adj_binary = (adj_matrix > threshold).astype(int)
        
        # Get edge indices
        edge_indices = np.where(adj_binary)
        edge_index = np.vstack([edge_indices[0], edge_indices[1]])
        
        return edge_index
    
    def get_data_statistics(self) -> Dict:
        """Get comprehensive data statistics"""
        return {
            'num_sensors': self.raw_data.shape[1],
            'num_timesteps': self.raw_data.shape[0],
            'sequence_length': self.sequence_length,
            'prediction_length': self.prediction_length,
            'num_sequences': len(self.sequences),
            'train_sequences': len(self.train_indices),
            'val_sequences': len(self.val_indices),
            'test_sequences': len(self.test_indices),
            'missing_data_stats': self.missing_stats,
            'data_range': {
                'min': float(np.nanmin(self.raw_data)),
                'max': float(np.nanmax(self.raw_data)),
                'mean': float(np.nanmean(self.raw_data)),
                'std': float(np.nanstd(self.raw_data))
            }
        }
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Data:
        return self._create_pyg_data(idx)


class EnhancedMETRLADataset(Dataset):
    """
    Enhanced METR-LA dataset with advanced preprocessing and missing data handling
    Handles HDF5 format data
    """
    
    def __init__(
        self,
        root_dir: str = 'data',
        sequence_length: int = 12,
        prediction_length: int = 12,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        preprocessing_method: str = 'robust',
        analyze_missing_data: bool = True
    ):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.preprocessing_method = preprocessing_method
        
        # Initialize components
        self.missing_analyzer = MissingDataAnalyzer()
        # Skip KNN for METR-LA (too slow for large datasets)
        self.preprocessor = AdvancedDataPreprocessor(use_knn=False)
        
        # Load and process data
        self.raw_data, self.adjacency_matrix, self.sensor_metadata = self._load_raw_data()
        
        if analyze_missing_data:
            print("[ANALYSIS] Analyzing missing data patterns...")
            self.missing_stats = self.missing_analyzer.analyze_missing_data(self.raw_data)
            self._print_missing_data_report()
        
        # Preprocess data
        print("[PREPROCESS] Preprocessing data...")
        self.processed_data = self.preprocessor.fit_transform(self.raw_data, self.preprocessing_method)
        
        # Create sequences
        print("[SEQUENCES] Creating sequences...")
        self.sequences = self._create_sequences()
        
        # Split data
        self.train_indices, self.val_indices, self.test_indices = self._split_data()
        
        print(f"[READY] Dataset ready: {len(self.sequences)} sequences")
        print(f"   Train: {len(self.train_indices)}, Val: {len(self.val_indices)}, Test: {len(self.test_indices)}")
    
    def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
        """Load raw METR-LA data from CSV or HDF5, adjacency matrix, and metadata"""
        # Load traffic data
        csv_candidates = [
            os.path.join(self.root_dir, 'METR-LA.csv'),
            os.path.join(self.root_dir, 'metr-la', 'METR-LA.csv')
        ]
        h5_candidates = [
            os.path.join(self.root_dir, 'metr-la.h5'),
            os.path.join(self.root_dir, 'METR-LA.h5'),
            os.path.join(self.root_dir, 'metr-la', 'metr-la.h5')
        ]
        
        data_path = None
        is_csv = False
        for p in csv_candidates:
            if os.path.exists(p):
                data_path = p
                is_csv = True
                break
        if not data_path:
            for p in h5_candidates:
                if os.path.exists(p):
                    data_path = p
                    is_csv = False
                    break
        
        if not data_path or not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found in {self.root_dir} (looked for METR-LA.csv, metr-la.h5, METR-LA.h5)")
        
        print(f"[LOAD] Loading data from {data_path}")
        if is_csv:
            df = pd.read_csv(data_path)
            first_col_sample = str(df.iloc[0, 0])
            if any(char in first_col_sample for char in ['-', ':', ' ']) or str(df.columns[0]).lower() in ['time', 'timestamp', 'date', 'unnamed: 0']:
                data = df.iloc[:, 1:].values
                print(f"   Detected timestamp column, using columns 1-{df.shape[1]-1}")
            else:
                data = df.values
        else:
            df = pd.read_hdf(data_path)
            data = df.values
        
        print(f"   Data shape: {data.shape}")
        print(f"   Number of sensors: {data.shape[1]}")
        
        # Load adjacency matrix
        adj_candidates = [
            os.path.join(self.root_dir, 'adj_mx_METR-LA.pkl'),
            os.path.join(self.root_dir, 'adj_mx.pkl'),
            os.path.join(self.root_dir, 'metr-la', 'adj_mx_METR-LA.pkl'),
            os.path.join(self.root_dir, 'metr-la', 'adj_mx.pkl')
        ]
        adjacency_matrix = None
        for adj_path in adj_candidates:
            if os.path.exists(adj_path):
                try:
                    with open(adj_path, 'rb') as f:
                        raw = f.read().replace(b'\r\n', b'\n')
                        adj_data = pickle.loads(raw, encoding='latin1')
                        if isinstance(adj_data, list):
                            adjacency_matrix = adj_data[2]
                        else:
                            adjacency_matrix = adj_data
                    print(f"   Adjacency matrix loaded from {adj_path} (shape: {adjacency_matrix.shape})")
                    break
                except Exception as e:
                    print(f"   [WARN] Error loading adjacency matrix from {adj_path}: {e}")
        
        if adjacency_matrix is None:
            print("   [WARN] Adjacency matrix not found, creating identity matrix")
            adjacency_matrix = np.eye(data.shape[1], dtype=np.float32)
        
        # Load metadata if available
        meta_candidates = [
            os.path.join(self.root_dir, 'METR-LA-META.csv'),
            os.path.join(self.root_dir, 'metr-la', 'METR-LA-META.csv')
        ]
        metadata = None
        for meta_path in meta_candidates:
            if os.path.exists(meta_path):
                try:
                    metadata = pd.read_csv(meta_path)
                    print(f"   Metadata loaded: {len(metadata)} sensors")
                    break
                except Exception as e:
                    print(f"   [WARN] Error loading metadata from {meta_path}: {e}")
        
        return data.astype(np.float32), adjacency_matrix.astype(np.float32), metadata
    
    def _print_missing_data_report(self):
        """Print detailed missing data analysis report"""
        stats = self.missing_stats
        
        print("\n[MISSING] Missing Data Analysis Report")
        print("=" * 50)
        print(f"Total missing values: {stats['total_missing_count']:,} ({stats['total_missing_percentage']:.2f}%)")
        
        # Sensor reliability
        reliability_scores = np.array(stats['sensor_reliability_scores'])
        print(f"\nSensor Reliability Statistics:")
        print(f"  Mean reliability: {np.mean(reliability_scores):.2f}%")
        print(f"  Median reliability: {np.median(reliability_scores):.2f}%")
        print(f"  Min reliability: {np.min(reliability_scores):.2f}%")
        print(f"  Max reliability: {np.max(reliability_scores):.2f}%")
        
        # Quality groups
        quality_groups = self.missing_analyzer.get_sensor_quality_groups()
        print(f"\nSensor Quality Groups:")
        print(f"  High quality (>=90%): {len(quality_groups.get('high_quality', []))} sensors")
        print(f"  Medium quality (70-90%): {len(quality_groups.get('medium_quality', []))} sensors")
        print(f"  Low quality (<70%): {len(quality_groups.get('low_quality', []))} sensors")
        
        # Most/least reliable sensors
        most_reliable = stats['most_reliable_sensors'][-5:]
        least_reliable = stats['least_reliable_sensors'][:5]
        
        print(f"\nTop 5 Most Reliable Sensors:")
        for idx in most_reliable:
            print(f"  Sensor {idx}: {reliability_scores[idx]:.2f}% reliability")
        
        print(f"\nTop 5 Least Reliable Sensors:")
        for idx in least_reliable:
            print(f"  Sensor {idx}: {reliability_scores[idx]:.2f}% reliability")
    
    def _create_sequences(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Create input-output sequences with missing data masks"""
        sequences = []
        
        for i in range(len(self.processed_data) - self.sequence_length - self.prediction_length + 1):
            # Input sequence
            x = self.processed_data[i:i + self.sequence_length].T  # [sensors, time_steps]
            
            # Output sequence
            y = self.processed_data[i + self.sequence_length:i + self.sequence_length + self.prediction_length].T
            
            # Missing data mask for input
            raw_x = self.raw_data[i:i + self.sequence_length].T
            missing_mask = np.isnan(raw_x) | (np.abs(raw_x) < 1e-3)
            
            sequences.append((x, y, missing_mask))
        
        return sequences
    
    def _split_data(self) -> Tuple[List[int], List[int], List[int]]:
        """Split data into train/validation/test sets"""
        n_sequences = len(self.sequences)
        
        train_size = int(n_sequences * self.train_ratio)
        val_size = int(n_sequences * self.val_ratio)
        
        indices = list(range(n_sequences))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        return train_indices, val_indices, test_indices
    
    def get_train_data(self) -> List[Data]:
        """Get training data as PyTorch Geometric Data objects"""
        return [self._create_pyg_data(i) for i in self.train_indices]
    
    def get_val_data(self) -> List[Data]:
        """Get validation data as PyTorch Geometric Data objects"""
        return [self._create_pyg_data(i) for i in self.val_indices]
    
    def get_test_data(self) -> List[Data]:
        """Get test data as PyTorch Geometric Data objects"""
        return [self._create_pyg_data(i) for i in self.test_indices]
    
    def _create_pyg_data(self, idx: int) -> Data:
        """Create PyTorch Geometric Data object"""
        x, y, missing_mask = self.sequences[idx]
        
        # Create edge index from adjacency matrix
        edge_index = self._adjacency_to_edge_index(self.adjacency_matrix)
        
        return Data(
            x=torch.tensor(x, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            missing_mask=torch.tensor(missing_mask, dtype=torch.bool)
        )
    
    def _adjacency_to_edge_index(self, adj_matrix: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """Convert adjacency matrix to edge index format"""
        # Use threshold to create edges (only keep strong connections)
        adj_binary = (adj_matrix > threshold).astype(int)
        
        # Get edge indices
        edge_indices = np.where(adj_binary)
        edge_index = np.vstack([edge_indices[0], edge_indices[1]])
        
        return edge_index
    
    def get_data_statistics(self) -> Dict:
        """Get comprehensive data statistics"""
        return {
            'num_sensors': self.raw_data.shape[1],
            'num_timesteps': self.raw_data.shape[0],
            'sequence_length': self.sequence_length,
            'prediction_length': self.prediction_length,
            'num_sequences': len(self.sequences),
            'train_sequences': len(self.train_indices),
            'val_sequences': len(self.val_indices),
            'test_sequences': len(self.test_indices),
            'missing_data_stats': self.missing_stats,
            'data_range': {
                'min': float(np.nanmin(self.raw_data)),
                'max': float(np.nanmax(self.raw_data)),
                'mean': float(np.nanmean(self.raw_data)),
                'std': float(np.nanstd(self.raw_data))
            }
        }
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Data:
        return self._create_pyg_data(idx)


def create_enhanced_dataset(
    root_dir: str = 'data',
    sequence_length: int = 12,
    prediction_length: int = 12,
    preprocessing_method: str = 'robust',
    dataset_name: str = 'PEMS-BAY'
):
    """
    Factory function to create enhanced dataset
    
    Args:
        root_dir: Data directory path
        sequence_length: Input sequence length
        prediction_length: Output sequence length
        preprocessing_method: Preprocessing method
        dataset_name: Dataset name ('PEMS-BAY' or 'METR-LA')
        
    Returns:
        Enhanced dataset (PEMS-BAY or METR-LA)
    """
    if dataset_name.upper() in ['METR-LA', 'METRLA'] or 'metr-la' in str(root_dir).lower() or 'metrla' in str(root_dir).lower():
        return EnhancedMETRLADataset(
            root_dir=root_dir,
            sequence_length=sequence_length,
            prediction_length=prediction_length,
            preprocessing_method=preprocessing_method,
            analyze_missing_data=True
        )
    else:
        return EnhancedPEMSBayDataset(
            root_dir=root_dir,
            sequence_length=sequence_length,
            prediction_length=prediction_length,
            preprocessing_method=preprocessing_method,
            analyze_missing_data=True
        )