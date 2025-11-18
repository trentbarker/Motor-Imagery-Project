"""
BCI Motor Imagery Classifier Training Script

This script trains a classic BCI classifier using:
- Common Spatial Patterns (CSP) for feature extraction
- Linear Discriminant Analysis (LDA) for classification

The script processes BCI Competition IV Dataset 2a for motor imagery classification.
"""

import os
import numpy as np
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
import joblib

# Set MNE to download datasets if needed
mne.set_log_level('WARNING')  # Reduce verbosity

# ============================================================================
# STEP 1: Download and Load BCI Competition IV Dataset 2a
# ============================================================================

print("=" * 60)
print("BCI Motor Imagery Classifier Training")
print("=" * 60)

# BCI Competition IV Dataset 2a Information:
# - 9 subjects performing motor imagery (left hand, right hand, foot, tongue)
# - Download from: http://www.bbci.de/competition/iv/#dataset2a
# - Files are in GDF format: A01T.gdf (training), A01E.gdf (evaluation)
# - Subject IDs: A01 through A09
# - Event codes: 1=left hand, 2=right hand, 3=foot, 4=tongue

# Set subject ID (1-9)
subject_id = 1

# Define data path
data_path = os.path.join('data', 'BCICIV_2a')
file_path = os.path.join(data_path, f'A0{subject_id}T.gdf')

# Check if dataset file exists
if not os.path.exists(file_path):
    print("\n" + "!" * 60)
    print("BCI Competition IV Dataset 2a not found!")
    print("!" * 60)
    print("\nTo download the dataset:")
    print("1. Visit: http://www.bbci.de/competition/iv/#dataset2a")
    print("2. Download the dataset files")
    print("3. Extract to: data/BCICIV_2a/")
    print("4. Files should be named: A01T.gdf, A01E.gdf, A02T.gdf, etc.")
    print("\nFor now, using MNE sample dataset for demonstration...")
    print("(Replace this section with actual Dataset 2a loading code)")
    
    # Load MNE sample dataset for demonstration
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                        'sample_audvis_raw.fif')
    raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)
    
    # Note: The rest of the script will work with this sample data structure
    # but results won't be meaningful. Replace with actual Dataset 2a loading.
    print("\n⚠️  Using sample dataset - results are for demonstration only!")
else:
    # Load actual BCI Competition IV Dataset 2a data
    print(f"\nLoading BCI Competition IV Dataset 2a")
    print(f"Subject: {subject_id}")
    print(f"File: {file_path}")
    
    # Load GDF file (BCI Competition format)
    # Note: You may need to specify montage or channel info
    raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=False)
    
    # Set montage if not already set (BCI Competition uses standard 10-20 system)
    # This helps with channel locations for spatial filtering
    try:
        # Try to set standard montage
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='warn', verbose=False)
    except:
        print("Note: Could not set standard montage. Continuing without it.")
    
    print(f"✓ Loaded subject {subject_id} training data")
    print(f"  Channels: {len(raw.ch_names)}")
    print(f"  Sampling rate: {raw.info['sfreq']} Hz")
    print(f"  Duration: {raw.times[-1]:.1f} seconds")

# ============================================================================
# STEP 2: Preprocessing - Band-pass Filtering
# ============================================================================

print("\n" + "-" * 60)
print("STEP 2: Applying Band-pass Filter (8-30 Hz)")
print("-" * 60)

# Band-pass filter: 8-30 Hz is optimal for motor imagery (mu and beta bands)
# This frequency range captures the event-related desynchronization (ERD)
# that occurs during motor imagery
raw_filtered = raw.copy().filter(l_freq=8.0, h_freq=30.0, 
                                  method='iir', iir_params=None, 
                                  picks='eeg', verbose=False)

print(f"Applied band-pass filter: 8-30 Hz")
print(f"Data shape: {raw_filtered.get_data().shape}")

# ============================================================================
# STEP 3: Artifact Removal using ICA
# ============================================================================

print("\n" + "-" * 60)
print("STEP 3: Artifact Removal using Independent Component Analysis (ICA)")
print("-" * 60)

# ICA separates signals into independent components, allowing us to identify
# and remove artifacts (eye blinks, muscle activity, etc.)
# We'll fit ICA and automatically detect/remove bad components

# Check data quality before ICA
n_channels = len(raw_filtered.ch_names)
print(f"Number of channels: {n_channels}")

# Try ICA with error handling
try:
    # Use a fixed number of components - ensure it's reasonable
    # For BCI data, typically use 10-20 components
    # Make sure we have at least 2 components and at most n_channels - 1
    n_ica_components = max(2, min(20, n_channels - 1))
    
    print(f"Attempting ICA with {n_ica_components} components...")
    
    # Create ICA object with fixed number of components
    ica = mne.preprocessing.ICA(n_components=n_ica_components, random_state=97, 
                                max_iter='auto', verbose=False)
    
    # Fit ICA on filtered data
    ica.fit(raw_filtered, verbose=False)
    print("✓ ICA fitted successfully")
    
    # Automatically detect artifacts (eye blinks, muscle artifacts)
    print("Detecting artifacts...")
    
    # Remove components that are likely artifacts
    # In practice, you might want to manually inspect and select components
    # For automation, we'll use correlation-based detection
    try:
        eog_indices, eog_scores = ica.find_bads_eog(raw_filtered, ch_name=None, 
                                                   threshold=3.0, verbose=False)
        if len(eog_indices) > 0:
            print(f"Found {len(eog_indices)} EOG-related components to remove")
            ica.exclude = eog_indices
        else:
            # If no EOG detected, exclude components with high variance (likely artifacts)
            # This is a simplified approach - manual inspection is recommended
            var_ratios = np.var(ica.get_sources(raw_filtered).get_data(), axis=1)
            # Exclude top 2 components with highest variance (often artifacts)
            exclude_indices = np.argsort(var_ratios)[-2:].tolist()
            ica.exclude = exclude_indices
            print(f"Excluding {len(exclude_indices)} high-variance components")
    except Exception as e:
        print(f"Warning: Could not detect artifacts automatically: {e}")
        print("Proceeding without artifact removal")
        ica.exclude = []
    
    # Apply ICA to remove artifacts
    raw_clean = raw_filtered.copy()
    ica.apply(raw_clean, verbose=False)
    print("✓ ICA applied - artifacts removed")
    
except RuntimeError as e:
    if "PCA component captures most" in str(e) or "threshold results in 1 component" in str(e):
        print("⚠️  ICA failed due to data characteristics (one component dominates variance)")
        print("   This can happen with certain data types or reference channels.")
        print("   Skipping ICA and using filtered data directly.")
        print("   Note: For production use, consider manual artifact removal or different preprocessing.")
        raw_clean = raw_filtered.copy()
    else:
        raise
except Exception as e:
    print(f"⚠️  ICA encountered an error: {e}")
    print("   Skipping ICA and using filtered data directly.")
    raw_clean = raw_filtered.copy()

# ============================================================================
# STEP 4: Epoching - Extract Trials for Left and Right Hand Imagery
# ============================================================================

print("\n" + "-" * 60)
print("STEP 4: Epoching Data into Trials")
print("-" * 60)

# For BCI Competition IV Dataset 2a:
# Event codes: 769 = left hand, 770 = right hand, 771 = foot, 772 = tongue
# We'll focus on left hand vs right hand classification

# Extract events from the dataset
# BCI Competition datasets have events embedded in the annotations
try:
    # Try to find events in the data
    events, event_dict = mne.events_from_annotations(raw_clean, verbose=False)
    
    print(f"Found {len(events)} total events")
    print(f"Event dictionary (annotation -> event code): {event_dict}")
    
    # BCI Competition IV Dataset 2a event codes:
    # 769 = left hand, 770 = right hand, 771 = foot, 772 = tongue
    # The event_dict maps annotation descriptions (strings) to event IDs (integers)
    
    # Check what event codes are actually in the events array
    unique_event_codes = np.unique(events[:, 2])
    print(f"Unique event codes in events array: {unique_event_codes}")
    
    # Diagnostic: Count events by code BEFORE filtering
    print("\nEvent counts by code (before filtering):")
    for code in unique_event_codes:
        count = np.sum(events[:, 2] == code)
        # Try to find description in event_dict
        desc = "unknown"
        for key, val in event_dict.items():
            if val == code:
                desc = str(key)
                break
        print(f"  Code {code} ({desc}): {count} events")
    
    # Identify left and right hand events by checking event_dict
    # MNE maps annotation strings to sequential integers (1, 2, 3...)
    # So we need to find which MNE code corresponds to annotation '769' (left) and '770' (right)
    
    left_code = None
    right_code = None
    
    # Method 1: Look for annotation keys '769' and '770' in event_dict
    # These are the standard BCI Competition IV Dataset 2a codes
    if '769' in event_dict and '770' in event_dict:
        left_code = event_dict['769']   # MNE-assigned code for annotation '769'
        right_code = event_dict['770']  # MNE-assigned code for annotation '770'
        print(f"✓ Identified using BCI Competition codes: annotation '769'→code {left_code} (left), '770'→code {right_code} (right)")
    
    # Method 2: Check if annotation keys are integers instead of strings
    elif 769 in event_dict and 770 in event_dict:
        left_code = event_dict[769]
        right_code = event_dict[770]
        print(f"✓ Identified using BCI Competition codes: annotation 769→code {left_code} (left), 770→code {right_code} (right)")
    
    # Method 3: Try to find by checking event_dict values for codes that have 72 events (expected count)
    # BCI Competition IV Dataset 2a should have ~72 trials per class
    else:
        # Look for codes with ~72 events (typical for BCI Competition)
        code_counts = {code: np.sum(events[:, 2] == code) for code in unique_event_codes}
        codes_with_72 = [code for code, count in code_counts.items() if 65 <= count <= 80]  # Allow some variance
        
        if len(codes_with_72) >= 2:
            # Try to match with event_dict annotations
            for key, val in event_dict.items():
                key_str = str(key)
                if ('769' in key_str or 'left' in key_str.lower()) and val in codes_with_72:
                    left_code = val
                elif ('770' in key_str or 'right' in key_str.lower()) and val in codes_with_72:
                    right_code = val
            
            if left_code is None or right_code is None:
                # Use first two codes with ~72 events
                left_code = codes_with_72[0]
                right_code = codes_with_72[1]
                print(f"⚠️  Using codes with ~72 events: {left_code}=left, {right_code}=right (verify these are correct!)")
            else:
                print(f"✓ Identified from event counts: code {left_code} (left), code {right_code} (right)")
        else:
            raise ValueError(f"Could not identify left/right hand events. "
                           f"Expected ~72 events per class. Found: {code_counts}. "
                           f"Event dict: {event_dict}")
    
    # Create event_id dictionary
    event_id = {'left_hand': left_code, 'right_hand': right_code}
    
    # Filter events to only include our classes of interest
    target_codes = [left_code, right_code]
    event_mask = np.isin(events[:, 2], target_codes)
    events_filtered = events[event_mask]
    
    print(f"\nUsing {len(events_filtered)} events for left/right hand classification")
    
    # Check if we have events for both classes
    left_count = np.sum(events_filtered[:, 2] == left_code)
    right_count = np.sum(events_filtered[:, 2] == right_code)
    print(f"  - Left hand events (code {left_code}): {left_count}")
    print(f"  - Right hand events (code {right_code}): {right_count}")
    
    if len(events_filtered) == 0:
        raise ValueError("No events found for left/right hand classification!")
    
    if left_count == 0 or right_count == 0:
        raise ValueError(f"Missing events for one class! Left: {left_count}, Right: {right_count}")
    
    events = events_filtered
    
except Exception as e:
    # If events can't be found, create synthetic events for demonstration
    print(f"⚠️  Could not extract events from annotations: {e}")
    print("Creating synthetic events for demonstration...")
    # Create events every 2 seconds
    sfreq = raw_clean.info['sfreq']
    events = np.array([[int(i * sfreq * 2), 0, 1 if i % 2 == 0 else 2] 
                       for i in range(20)])
    event_id = {'left_hand': 1, 'right_hand': 2}
    print("⚠️  Using synthetic events - replace with actual event extraction")

# Create epochs: time window from -0.5 to 4 seconds relative to cue
# This captures the motor imagery period
tmin, tmax = -0.5, 4.0  # seconds
epochs = mne.Epochs(raw_clean, events, event_id, tmin, tmax, 
                    baseline=(None, 0), preload=True, verbose=False)

print(f"Created {len(epochs)} epochs")
print(f"Epoch shape: {epochs.get_data().shape}")
print(f"Time range: {epochs.tmin:.2f}s to {epochs.tmax:.2f}s")

# ============================================================================
# STEP 5: Feature Extraction using Common Spatial Patterns (CSP)
# ============================================================================

print("\n" + "-" * 60)
print("STEP 5: Feature Extraction using Common Spatial Patterns (CSP)")
print("-" * 60)

"""
Common Spatial Patterns (CSP) Explanation:
-------------------------------------------
CSP is a spatial filtering technique that finds optimal linear combinations
of EEG channels that maximize the variance for one class while minimizing
it for the other class. 

How it works:
1. CSP computes spatial filters that project the multi-channel EEG data
   onto a lower-dimensional space
2. These filters are found by solving a generalized eigenvalue problem
   using the covariance matrices of the two classes
3. The resulting CSP features (log-variance of filtered signals) capture
   the discriminative spatial patterns between left and right hand imagery
4. CSP is particularly effective for motor imagery because it captures
   the contralateral activation patterns (left hand imagery activates
   right motor cortex, and vice versa)

The CSP algorithm outputs features that are highly discriminative for
binary classification tasks like left vs right hand motor imagery.
"""

# Prepare data for CSP
# Get data in shape (n_epochs, n_channels, n_times)
X = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
y = epochs.events[:, 2]  # Event labels

# Convert labels to 0 and 1 for sklearn compatibility
# Use the actual event_id values from the event_id dictionary
left_code = event_id['left_hand']
y_binary = (y == left_code).astype(int)  # 1 for left hand, 0 for right hand

print(f"Data shape: {X.shape}")
print(f"Labels: {np.sum(y_binary)} left hand, {np.sum(1-y_binary)} right hand")

# Check for class imbalance (informational only - no balancing)
left_count = np.sum(y_binary)
right_count = np.sum(1 - y_binary)
min_class_count = min(left_count, right_count)

if min_class_count < 2:
    print("\n" + "!" * 60)
    print("ERROR: Insufficient samples for one class!")
    print("!" * 60)
    print(f"Left: {left_count}, Right: {right_count}")
    print(f"\nCSP requires at least 2 samples per class.")
    print("\nPossible solutions:")
    print("1. Check if event extraction is correct - maybe some events are being missed")
    print("2. Try a different subject (change subject_id at top of script)")
    print("3. Check if you need to combine training + test data")
    print("4. Verify the dataset file is complete and not corrupted")
    print("\nBCI Competition IV Dataset 2a should have ~72 trials per class per subject.")
    print("If you're seeing much fewer, there may be an event extraction issue.")
    raise ValueError(f"Insufficient samples for one class! Left: {left_count}, Right: {right_count}. "
                     f"CSP requires at least 2 samples per class.")

# Report class distribution (no balancing applied)
if abs(left_count - right_count) > 0:
    imbalance_ratio = max(left_count, right_count) / min(left_count, right_count)
    print(f"ℹ️  Class distribution: Left={left_count}, Right={right_count} (ratio: {imbalance_ratio:.2f}:1)")
    if imbalance_ratio > 2.0:
        print("   Note: Significant class imbalance detected. Model will train on unbalanced data.")
elif min_class_count < 5:
    print(f"ℹ️  Small sample size: Left={left_count}, Right={right_count}")
    print("   Results may be less reliable with such small sample sizes.")

# Create CSP object
# n_components: number of CSP components to use (typically 4-6 for binary classification)
# reg: regularization parameter to prevent overfitting
# transform_into='average_power' gives log-variance features (2D array for LDA)
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False, 
          transform_into='average_power')

# Fit CSP on the data
print("Fitting CSP...")
csp.fit(X, y_binary)

# Transform data to CSP features
X_csp = csp.transform(X)
print(f"CSP features shape: {X_csp.shape}")

# Ensure X_csp is 2D (n_samples, n_features) for LDA
if X_csp.ndim > 2:
    # Reshape if somehow 3D
    n_samples = X_csp.shape[0]
    X_csp = X_csp.reshape(n_samples, -1)
    print(f"Reshaped CSP features to: {X_csp.shape}")

print("CSP fitted and features extracted")

# ============================================================================
# STEP 6: Train Linear Discriminant Analysis (LDA) Classifier
# ============================================================================

print("\n" + "-" * 60)
print("STEP 6: Training Linear Discriminant Analysis (LDA) Classifier")
print("-" * 60)

"""
Linear Discriminant Analysis (LDA) Explanation:
-----------------------------------------------
LDA is a linear classification algorithm that finds the optimal linear
boundary (hyperplane) to separate two classes.

How it works:
1. LDA assumes that data from each class follows a Gaussian distribution
   with different means but the same covariance matrix
2. It finds the linear combination of features that maximizes the ratio
   of between-class variance to within-class variance
3. The decision boundary is a linear function of the input features
4. For CSP features, LDA works well because CSP already extracts
   discriminative features, and LDA provides a simple, robust classifier

Why LDA for BCI:
- Fast training and prediction
- Works well with small datasets
- Robust to overfitting
- Interpretable (linear decision boundary)
- Often achieves good performance with CSP features
"""

# Create and train LDA classifier
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

print("Training LDA classifier...")
lda.fit(X_csp, y_binary)

# Evaluate on training data (for demonstration)
train_score = lda.score(X_csp, y_binary)
print(f"Training accuracy: {train_score:.2%}")

# ============================================================================
# STEP 7: Save Models
# ============================================================================

print("\n" + "-" * 60)
print("STEP 7: Saving Models")
print("-" * 60)

# Create saved_models directory if it doesn't exist
os.makedirs('saved_models', exist_ok=True)

# Save CSP model
csp_path = os.path.join('saved_models', 'csp_model.pkl')
joblib.dump(csp, csp_path)
print(f"CSP model saved to: {csp_path}")

# Save LDA model
lda_path = os.path.join('saved_models', 'lda_model.pkl')
joblib.dump(lda, lda_path)
print(f"LDA model saved to: {lda_path}")

# Optionally, save both together in a pipeline
pipeline = Pipeline([('csp', csp), ('lda', lda)])
pipeline_path = os.path.join('saved_models', 'csp_lda_pipeline.pkl')
joblib.dump(pipeline, pipeline_path)
print(f"Complete pipeline saved to: {pipeline_path}")

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print("\nModels saved in saved_models/ folder:")
print("  - csp_model.pkl: CSP feature extractor")
print("  - lda_model.pkl: LDA classifier")
print("  - csp_lda_pipeline.pkl: Complete pipeline (CSP + LDA)")
print("\nYou can now use these models for prediction in your app.py")
