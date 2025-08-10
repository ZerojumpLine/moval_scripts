"""
Metrics utilities for MOVAL experiments.

This module provides common evaluation metrics used across different
classification and segmentation tasks.
"""

import numpy as np
from typing import Union, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_prob: Optional[np.ndarray] = None,
                                   average: str = 'weighted') -> dict:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for AUC calculation)
        average: Averaging method for multi-class metrics
        
    Returns:
        Dictionary containing calculated metrics
    """
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # AUC calculation if probabilities are provided
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
            else:
                # Multi-class classification
                metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average=average)
        except ValueError:
            metrics['auc'] = np.nan
    
    return metrics


def calculate_segmentation_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                 threshold: float = 0.5) -> dict:
    """
    Calculate segmentation metrics (DSC, IoU, etc.).
    
    Args:
        y_true: True segmentation masks
        y_pred: Predicted segmentation masks
        threshold: Threshold for binary segmentation
        
    Returns:
        Dictionary containing calculated metrics
    """
    # Ensure binary masks
    y_true_bin = (y_true > threshold).astype(np.uint8)
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    
    metrics = {}
    
    # Calculate metrics for each sample
    dsc_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []
    
    for i in range(len(y_true_bin)):
        true_mask = y_true_bin[i].flatten()
        pred_mask = y_pred_bin[i].flatten()
        
        # Intersection and union
        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()
        
        # Dice Similarity Coefficient (DSC)
        if (true_mask.sum() + pred_mask.sum()) > 0:
            dsc = (2 * intersection) / (true_mask.sum() + pred_mask.sum())
        else:
            dsc = 1.0 if intersection == 0 else 0.0
        dsc_scores.append(dsc)
        
        # IoU (Jaccard Index)
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0 if intersection == 0 else 0.0
        iou_scores.append(iou)
        
        # Precision and Recall
        if pred_mask.sum() > 0:
            precision = intersection / pred_mask.sum()
        else:
            precision = 1.0 if intersection == 0 else 0.0
        
        if true_mask.sum() > 0:
            recall = intersection / true_mask.sum()
        else:
            recall = 1.0 if intersection == 0 else 0.0
            
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    # Aggregate metrics
    metrics['dsc'] = np.mean(dsc_scores)
    metrics['dsc_std'] = np.std(dsc_scores)
    metrics['iou'] = np.mean(iou_scores)
    metrics['iou_std'] = np.std(iou_scores)
    metrics['precision'] = np.mean(precision_scores)
    metrics['precision_std'] = np.std(precision_scores)
    metrics['recall'] = np.mean(recall_scores)
    metrics['recall_std'] = np.std(recall_scores)
    
    return metrics


def calculate_confidence_metrics(confidence_scores: np.ndarray, 
                                true_metrics: np.ndarray,
                                confidence_bins: int = 10) -> dict:
    """
    Calculate confidence calibration metrics.
    
    Args:
        confidence_scores: Model confidence scores
        true_metrics: True performance metrics
        confidence_bins: Number of confidence bins
        
    Returns:
        Dictionary containing confidence calibration metrics
    """
    metrics = {}
    
    # Create confidence bins
    bin_edges = np.linspace(0, 1, confidence_bins + 1)
    bin_indices = np.digitize(confidence_scores, bin_edges) - 1
    
    # Calculate calibration metrics
    bin_confidence = []
    bin_performance = []
    bin_counts = []
    
    for i in range(confidence_bins):
        mask = (bin_indices == i)
        if mask.sum() > 0:
            bin_conf = confidence_scores[mask].mean()
            bin_perf = true_metrics[mask].mean()
            bin_count = mask.sum()
            
            bin_confidence.append(bin_conf)
            bin_performance.append(bin_perf)
            bin_counts.append(bin_count)
    
    # Expected Calibration Error (ECE)
    if len(bin_confidence) > 0:
        ece = np.sum(np.array(bin_counts) * np.abs(np.array(bin_confidence) - np.array(bin_performance))) / np.sum(bin_counts)
        metrics['ece'] = ece
    else:
        metrics['ece'] = np.nan
    
    # Reliability diagram data
    metrics['bin_confidence'] = bin_confidence
    metrics['bin_performance'] = bin_performance
    metrics['bin_counts'] = bin_counts
    
    return metrics


def aggregate_metrics(metrics_list: List[dict], method: str = 'mean') -> dict:
    """
    Aggregate metrics from multiple experiments or folds.
    
    Args:
        metrics_list: List of metric dictionaries
        method: Aggregation method ('mean', 'median', 'std')
        
    Returns:
        Aggregated metrics dictionary
    """
    if not metrics_list:
        return {}
    
    aggregated = {}
    
    # Get all unique metric names
    all_metrics = set()
    for metrics in metrics_list:
        all_metrics.update(metrics.keys())
    
    for metric in all_metrics:
        values = [m.get(metric, np.nan) for m in metrics_list]
        values = [v for v in values if not np.isnan(v)]
        
        if values:
            if method == 'mean':
                aggregated[metric] = np.mean(values)
            elif method == 'median':
                aggregated[metric] = np.median(values)
            elif method == 'std':
                aggregated[metric] = np.std(values)
            else:
                aggregated[metric] = values
        else:
            aggregated[metric] = np.nan
    
    return aggregated


def format_metrics_for_output(metrics: dict, precision: int = 4) -> dict:
    """
    Format metrics for output display and saving.
    
    Args:
        metrics: Raw metrics dictionary
        precision: Number of decimal places
        
    Returns:
        Formatted metrics dictionary
    """
    formatted = {}
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not np.isnan(value):
            if isinstance(value, int):
                formatted[key] = value
            else:
                formatted[key] = round(value, precision)
        else:
            formatted[key] = value
    
    return formatted 