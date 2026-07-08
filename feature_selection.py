"""
Feature Selection Module for Intrusion Detection System

Implements four feature selection methods from Lightweight IDS:
1. Forward Selection (FS)
2. Backward Elimination (BE)
3. Correlation-based Selection (CFS)
4. Feature Importance-based Selection (FIS)

These methods are designed to work with lightweight classifiers 
and can significantly reduce feature dimensionality while maintaining 
or improving detection accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from typing import List, Tuple, Union, Callable
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector:
    """Base class for feature selection methods."""
    
    def __init__(self, estimator, scoring='accuracy', cv=3, verbose=False):
        """
        Initialize feature selector.
        
        Args:
            estimator: Sklearn-compatible classifier
            scoring: Scoring metric ('accuracy', 'f1', 'precision', 'recall')
            cv: Number of cross-validation folds
            verbose: Whether to print progress information
        """
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose
        self.selected_features_ = None
        self.scores_ = []
        
    def fit(self, X, y):
        """Fit the feature selector. To be implemented by subclasses."""
        raise NotImplementedError
        
    def transform(self, X):
        """Transform the dataset to selected features."""
        if self.selected_features_ is None:
            raise ValueError("Feature selector not fitted yet.")
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_features_]
        else:
            return X[:, self.selected_features_]
    
    def fit_transform(self, X, y):
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)
    
    def _evaluate_features(self, X, y, features):
        """Evaluate a subset of features using cross-validation."""
        if isinstance(X, pd.DataFrame):
            X_subset = X.iloc[:, features]
        else:
            X_subset = X[:, features]
        
        try:
            scores = cross_val_score(
                clone(self.estimator), 
                X_subset, 
                y, 
                cv=self.cv, 
                scoring=self.scoring,
                n_jobs=-1
            )
            return np.mean(scores)
        except Exception as e:
            if self.verbose:
                print(f"Error evaluating features: {e}")
            return 0.0


class ForwardSelection(FeatureSelector):
    """
    Forward Feature Selection (FS).
    
    Iteratively adds features that improve the model performance the most.
    Stops when no improvement is achieved or max_features is reached.
    """
    
    def __init__(self, estimator, max_features=None, min_improvement=0.001, 
                 scoring='accuracy', cv=3, verbose=False):
        """
        Initialize Forward Selection.
        
        Args:
            estimator: Sklearn-compatible classifier
            max_features: Maximum number of features to select (None for auto)
            min_improvement: Minimum improvement required to add a feature
            scoring: Scoring metric
            cv: Number of cross-validation folds
            verbose: Whether to print progress
        """
        super().__init__(estimator, scoring, cv, verbose)
        self.max_features = max_features
        self.min_improvement = min_improvement
    
    def fit(self, X, y):
        """Perform forward feature selection."""
        n_features = X.shape[1]
        if self.max_features is None:
            self.max_features = n_features
        
        selected = []
        remaining = list(range(n_features))
        current_score = 0.0
        
        if self.verbose:
            print(f"Forward Selection: Starting with {n_features} features")
        
        while len(selected) < self.max_features and remaining:
            best_feature = None
            best_score = current_score
            
            for feature in remaining:
                candidate = selected + [feature]
                score = self._evaluate_features(X, y, candidate)
                
                if score > best_score:
                    best_score = score
                    best_feature = feature
            
            if best_feature is not None and (best_score - current_score) >= self.min_improvement:
                selected.append(best_feature)
                remaining.remove(best_feature)
                current_score = best_score
                self.scores_.append(current_score)
                
                if self.verbose:
                    print(f"  Added feature {best_feature}: score = {current_score:.4f}")
            else:
                break
        
        self.selected_features_ = selected
        
        if self.verbose:
            print(f"Forward Selection: Selected {len(selected)} features")
            print(f"Final score: {current_score:.4f}")
        
        return self


class BackwardElimination(FeatureSelector):
    """
    Backward Feature Elimination (BE).
    
    Starts with all features and iteratively removes the least important ones.
    Stops when performance drops significantly or min_features is reached.
    """
    
    def __init__(self, estimator, min_features=5, max_drop=0.01, 
                 scoring='accuracy', cv=3, verbose=False):
        """
        Initialize Backward Elimination.
        
        Args:
            estimator: Sklearn-compatible classifier
            min_features: Minimum number of features to keep
            max_drop: Maximum allowed performance drop
            scoring: Scoring metric
            cv: Number of cross-validation folds
            verbose: Whether to print progress
        """
        super().__init__(estimator, scoring, cv, verbose)
        self.min_features = min_features
        self.max_drop = max_drop
    
    def fit(self, X, y):
        """Perform backward feature elimination."""
        n_features = X.shape[1]
        selected = list(range(n_features))
        
        # Evaluate with all features
        current_score = self._evaluate_features(X, y, selected)
        initial_score = current_score
        self.scores_.append(current_score)
        
        if self.verbose:
            print(f"Backward Elimination: Starting with {n_features} features")
            print(f"Initial score: {initial_score:.4f}")
        
        while len(selected) > self.min_features:
            worst_feature = None
            best_score = 0.0
            
            for feature in selected:
                candidate = [f for f in selected if f != feature]
                score = self._evaluate_features(X, y, candidate)
                
                if score > best_score:
                    best_score = score
                    worst_feature = feature
            
            # Check if removal is acceptable
            if worst_feature is not None and (initial_score - best_score) <= self.max_drop:
                selected.remove(worst_feature)
                current_score = best_score
                self.scores_.append(current_score)
                
                if self.verbose:
                    print(f"  Removed feature {worst_feature}: score = {current_score:.4f}")
            else:
                break
        
        self.selected_features_ = selected
        
        if self.verbose:
            print(f"Backward Elimination: Selected {len(selected)} features")
            print(f"Final score: {current_score:.4f}")
        
        return self


class CorrelationBasedSelection(FeatureSelector):
    """
    Correlation-based Feature Selection (CFS).
    
    Selects features with high correlation to the target 
    and low correlation with each other to reduce redundancy.
    """
    
    def __init__(self, estimator=None, threshold=0.7, max_features=None, 
                 scoring='accuracy', cv=3, verbose=False):
        """
        Initialize Correlation-based Selection.
        
        Args:
            estimator: Sklearn-compatible classifier (optional, for scoring)
            threshold: Maximum correlation threshold between features
            max_features: Maximum number of features to select
            scoring: Scoring metric
            cv: Number of cross-validation folds
            verbose: Whether to print progress
        """
        super().__init__(estimator, scoring, cv, verbose)
        self.threshold = threshold
        self.max_features = max_features
        self.feature_target_corr_ = None
        self.feature_corr_matrix_ = None
    
    def fit(self, X, y):
        """Perform correlation-based feature selection."""
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()
        
        y_series = pd.Series(y) if isinstance(y, np.ndarray) else y
        
        n_features = X_df.shape[1]
        
        # Calculate correlation with target
        feature_target_corr = []
        for col in X_df.columns:
            try:
                corr = abs(X_df[col].corr(y_series))
                feature_target_corr.append(corr if not np.isnan(corr) else 0.0)
            except:
                feature_target_corr.append(0.0)
        
        self.feature_target_corr_ = np.array(feature_target_corr)
        
        # Calculate feature-feature correlation matrix
        self.feature_corr_matrix_ = X_df.corr().abs().values
        
        # Sort features by target correlation (descending)
        sorted_indices = np.argsort(self.feature_target_corr_)[::-1]
        
        selected = []
        
        if self.verbose:
            print(f"Correlation-based Selection: Starting with {n_features} features")
        
        for idx in sorted_indices:
            # Check correlation with already selected features
            is_redundant = False
            for selected_idx in selected:
                if self.feature_corr_matrix_[idx, selected_idx] > self.threshold:
                    is_redundant = True
                    break
            
            if not is_redundant:
                selected.append(idx)
                
                if self.verbose:
                    print(f"  Added feature {idx}: target_corr = {self.feature_target_corr_[idx]:.4f}")
                
                if self.max_features and len(selected) >= self.max_features:
                    break
        
        self.selected_features_ = sorted(selected)
        
        if self.verbose:
            print(f"Correlation-based Selection: Selected {len(selected)} features")
        
        # Evaluate if estimator provided
        if self.estimator is not None:
            score = self._evaluate_features(X, y, self.selected_features_)
            self.scores_.append(score)
            if self.verbose:
                print(f"Final score: {score:.4f}")
        
        return self


class ImportanceBasedSelection(FeatureSelector):
    """
    Feature Importance-based Selection (FIS).
    
    Uses tree-based models or models with feature_importances_ 
    to select the most important features.
    """
    
    def __init__(self, estimator, top_k=None, threshold=None, 
                 scoring='accuracy', cv=3, verbose=False):
        """
        Initialize Importance-based Selection.
        
        Args:
            estimator: Sklearn-compatible classifier with feature_importances_
            top_k: Number of top features to select (mutually exclusive with threshold)
            threshold: Minimum importance threshold (mutually exclusive with top_k)
            scoring: Scoring metric
            cv: Number of cross-validation folds
            verbose: Whether to print progress
        """
        super().__init__(estimator, scoring, cv, verbose)
        self.top_k = top_k
        self.threshold = threshold
        self.feature_importances_ = None
        
        if top_k is None and threshold is None:
            raise ValueError("Either top_k or threshold must be specified")
        if top_k is not None and threshold is not None:
            raise ValueError("Only one of top_k or threshold can be specified")
    
    def fit(self, X, y):
        """Perform importance-based feature selection."""
        # Train estimator to get feature importances
        self.estimator.fit(X, y)
        
        if not hasattr(self.estimator, 'feature_importances_'):
            raise ValueError("Estimator must have feature_importances_ attribute")
        
        self.feature_importances_ = self.estimator.feature_importances_
        
        n_features = X.shape[1]
        
        if self.verbose:
            print(f"Importance-based Selection: Starting with {n_features} features")
        
        # Select features based on criterion
        if self.top_k is not None:
            # Select top-k features
            indices = np.argsort(self.feature_importances_)[::-1][:self.top_k]
            self.selected_features_ = sorted(indices.tolist())
        else:
            # Select features above threshold
            indices = np.where(self.feature_importances_ >= self.threshold)[0]
            self.selected_features_ = sorted(indices.tolist())
        
        if self.verbose:
            print(f"Importance-based Selection: Selected {len(self.selected_features_)} features")
            print(f"Selected features: {self.selected_features_}")
        
        # Evaluate selected features
        score = self._evaluate_features(X, y, self.selected_features_)
        self.scores_.append(score)
        
        if self.verbose:
            print(f"Final score: {score:.4f}")
        
        return self


def compare_feature_selection_methods(X, y, estimator, methods=None, cv=3, verbose=True):
    """
    Compare different feature selection methods.
    
    Args:
        X: Feature matrix
        y: Target labels
        estimator: Sklearn-compatible classifier
        methods: List of method names to compare (default: all)
        cv: Number of cross-validation folds
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary with method names as keys and results as values
    """
    if methods is None:
        methods = ['forward', 'backward', 'correlation', 'importance']
    
    results = {}
    
    if verbose:
        print("=" * 80)
        print("Feature Selection Comparison")
        print("=" * 80)
    
    # Forward Selection
    if 'forward' in methods:
        if verbose:
            print("\n[1/4] Forward Selection")
            print("-" * 80)
        fs = ForwardSelection(estimator, max_features=20, cv=cv, verbose=verbose)
        fs.fit(X, y)
        results['forward'] = {
            'selector': fs,
            'n_features': len(fs.selected_features_),
            'features': fs.selected_features_,
            'score': fs.scores_[-1] if fs.scores_ else 0.0
        }
    
    # Backward Elimination
    if 'backward' in methods:
        if verbose:
            print("\n[2/4] Backward Elimination")
            print("-" * 80)
        be = BackwardElimination(estimator, min_features=10, cv=cv, verbose=verbose)
        be.fit(X, y)
        results['backward'] = {
            'selector': be,
            'n_features': len(be.selected_features_),
            'features': be.selected_features_,
            'score': be.scores_[-1] if be.scores_ else 0.0
        }
    
    # Correlation-based Selection
    if 'correlation' in methods:
        if verbose:
            print("\n[3/4] Correlation-based Selection")
            print("-" * 80)
        cfs = CorrelationBasedSelection(estimator, threshold=0.7, max_features=20, cv=cv, verbose=verbose)
        cfs.fit(X, y)
        results['correlation'] = {
            'selector': cfs,
            'n_features': len(cfs.selected_features_),
            'features': cfs.selected_features_,
            'score': cfs.scores_[-1] if cfs.scores_ else 0.0
        }
    
    # Importance-based Selection
    if 'importance' in methods:
        if verbose:
            print("\n[4/4] Importance-based Selection")
            print("-" * 80)
        # Use a tree-based estimator for importance
        from sklearn.ensemble import RandomForestClassifier
        importance_estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        fis = ImportanceBasedSelection(importance_estimator, top_k=20, cv=cv, verbose=verbose)
        fis.fit(X, y)
        results['importance'] = {
            'selector': fis,
            'n_features': len(fis.selected_features_),
            'features': fis.selected_features_,
            'score': fis.scores_[-1] if fis.scores_ else 0.0
        }
    
    if verbose:
        print("\n" + "=" * 80)
        print("Comparison Summary")
        print("=" * 80)
        print(f"{'Method':<20} {'# Features':<15} {'Score':<10}")
        print("-" * 80)
        for method, result in results.items():
            print(f"{method:<20} {result['n_features']:<15} {result['score']:<10.4f}")
        print("=" * 80)
    
    return results


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.linear_model import SGDClassifier
    
    print("Feature Selection Module - Example Usage")
    print("=" * 80)
    
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000, 
        n_features=50, 
        n_informative=20, 
        n_redundant=15,
        n_classes=2, 
        random_state=42
    )
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Create lightweight classifier
    clf = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
    
    # Compare all methods
    results = compare_feature_selection_methods(X, y, clf, cv=3, verbose=True)
    
    # Use the best method
    best_method = max(results.items(), key=lambda x: x[1]['score'])
    print(f"\nBest method: {best_method[0]} (score: {best_method[1]['score']:.4f})")
    print(f"Selected features: {best_method[1]['features']}")
