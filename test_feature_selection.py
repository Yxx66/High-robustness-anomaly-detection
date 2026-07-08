"""
Quick test script for feature selection module.

This script performs a quick sanity check on all feature selection methods.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Testing Feature Selection Module...")
print("=" * 80)

# Import feature selection classes
try:
    from feature_selection import (
        ForwardSelection,
        BackwardElimination,
        CorrelationBasedSelection,
        ImportanceBasedSelection,
        compare_feature_selection_methods,
    )
    print("✓ Successfully imported feature selection module")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    exit(1)

# Generate synthetic dataset
print("\n1. Generating synthetic dataset...")
X, y = make_classification(
    n_samples=1000,
    n_features=50,
    n_informative=20,
    n_redundant=15,
    n_classes=2,
    random_state=42
)
print(f"   Dataset: {X.shape[0]} samples, {X.shape[1]} features")

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create classifier
clf = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)

# Test 1: Forward Selection
print("\n2. Testing Forward Selection...")
try:
    fs = ForwardSelection(clf, max_features=10, cv=2, verbose=False)
    fs.fit(X_train, y_train)
    print(f"   ✓ Selected {len(fs.selected_features_)} features")
    print(f"   Features: {fs.selected_features_[:5]}...")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 2: Backward Elimination
print("\n3. Testing Backward Elimination...")
try:
    be = BackwardElimination(clf, min_features=10, cv=2, verbose=False)
    be.fit(X_train, y_train)
    print(f"   ✓ Selected {len(be.selected_features_)} features")
    print(f"   Features: {be.selected_features_[:5]}...")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 3: Correlation-based Selection
print("\n4. Testing Correlation-based Selection...")
try:
    cfs = CorrelationBasedSelection(
        clf, threshold=0.7, max_features=15, cv=2, verbose=False
    )
    cfs.fit(X_train, y_train)
    print(f"   ✓ Selected {len(cfs.selected_features_)} features")
    print(f"   Features: {cfs.selected_features_[:5]}...")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 4: Importance-based Selection
print("\n5. Testing Importance-based Selection...")
try:
    from sklearn.ensemble import RandomForestClassifier
    importance_clf = RandomForestClassifier(
        n_estimators=50, random_state=42, n_jobs=-1
    )
    fis = ImportanceBasedSelection(
        importance_clf, top_k=15, cv=2, verbose=False
    )
    fis.fit(X_train, y_train)
    print(f"   ✓ Selected {len(fis.selected_features_)} features")
    print(f"   Features: {fis.selected_features_[:5]}...")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 5: Compare all methods
print("\n6. Testing compare_feature_selection_methods...")
try:
    results = compare_feature_selection_methods(
        X_train, y_train, clf, cv=2, verbose=False
    )
    print(f"   ✓ Compared {len(results)} methods")
    for method, result in results.items():
        print(f"   - {method}: {result['n_features']} features, "
              f"score={result['score']:.4f}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 6: Transform functionality
print("\n7. Testing transform functionality...")
try:
    fs = ForwardSelection(clf, max_features=10, cv=2, verbose=False)
    X_train_selected = fs.fit_transform(X_train, y_train)
    X_test_selected = fs.transform(X_test)
    print(f"   ✓ Train shape: {X_train.shape} → {X_train_selected.shape}")
    print(f"   ✓ Test shape: {X_test.shape} → {X_test_selected.shape}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 7: Classification with selected features
print("\n8. Testing classification with selected features...")
try:
    fs = ForwardSelection(clf, max_features=15, cv=2, verbose=False)
    X_train_selected = fs.fit_transform(X_train, y_train)
    X_test_selected = fs.transform(X_test)
    
    # Train classifier on selected features
    clf_selected = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
    clf_selected.fit(X_train_selected, y_train)
    accuracy = clf_selected.score(X_test_selected, y_test)
    print(f"   ✓ Accuracy with selected features: {accuracy:.4f}")
    
    # Compare with all features
    clf_all = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
    clf_all.fit(X_train, y_train)
    accuracy_all = clf_all.score(X_test, y_test)
    print(f"   ✓ Accuracy with all features: {accuracy_all:.4f}")
    
    improvement = accuracy - accuracy_all
    print(f"   {'✓' if improvement >= 0 else '!'} Improvement: "
          f"{improvement:+.4f} ({improvement*100:+.2f}%)")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n" + "=" * 80)
print("All tests completed!")
print("=" * 80)
print("\nNext steps:")
print("1. Run: python evaluate_feature_selection.py --dataset <your_dataset>")
print("2. Run: python pipeline.py --feature-selection compare")
print("3. Compare results with and without feature selection")
