"""
BCI Motor Imagery Classifier - Streamlit Application

Interactive demo for testing the trained CSP+LDA classifier on motor imagery data.
"""

import os
import time
import numpy as np
import streamlit as st
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib
import matplotlib.pyplot as plt
from io import BytesIO

# Set MNE log level to reduce verbosity
mne.set_log_level('WARNING')

# Page configuration
st.set_page_config(
    page_title="BCI Motor Imagery Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Header Section - Welcome and Attribution
# ============================================================================

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .welcome-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .welcome-subtitle {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    .creator-credit {
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.95rem;
        text-align: center;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
    }
    .info-box {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .info-box h4 {
        color: #a78bfa !important;
        margin-top: 0;
    }
    .info-box p {
        color: #e2e8f0 !important;
        margin-bottom: 0.5rem;
    }
    .info-box strong {
        color: #cbd5e0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header with gradient background
st.markdown("""
<div class="main-header">
    <div class="welcome-title">🧠 BCI Motor Imagery Classifier</div>
    <div class="welcome-subtitle">
        <strong>What is this?</strong> Imagine you're thinking about moving your left hand, but you don't actually move it. 
        Your brain still sends signals to your muscles—we can detect these signals using sensors on your head (EEG). 
        This app uses artificial intelligence to "read your mind" and figure out whether you're thinking about moving 
        your left hand or right hand, just by looking at your brain signals!
        <br><br>
        <strong>Why does this matter?</strong> This technology helps people who can't move their bodies control computers, 
        wheelchairs, or robotic arms using only their thoughts. It's like having a direct connection between your brain and a computer.
    </div>
    <div class="creator-credit">
        Created by <strong>Trent Barker</strong>
    </div>
</div>
""", unsafe_allow_html=True)

# Information box
st.markdown("""
<div class="info-box">
    <h4>📖 How It Works (Simple Version)</h4>
    <p>
        <strong>Step 1: Reading Brain Signals</strong> - We place sensors on your head that measure tiny electrical signals 
        from your brain. When you think about moving your left hand, different parts of your brain become active compared 
        to when you think about moving your right hand.
    </p>
    <p>
        <strong>Step 2: Finding Patterns</strong> - The computer looks at signals from all the sensors and finds patterns that 
        are different between "left hand thinking" and "right hand thinking." It's like learning to recognize the difference 
        between two songs by listening to them many times.
    </p>
    <p style="margin-bottom: 0;">
        <strong>Step 3: Making a Guess</strong> - When you think about moving a hand, the computer looks at the patterns it learned 
        and makes its best guess: "This looks like left hand thinking" or "This looks like right hand thinking." The more confident 
        it is, the higher the percentage you'll see.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# Load Models
# ============================================================================

@st.cache_resource
def load_models():
    """Load the trained CSP and LDA models."""
    try:
        # Load CSP model
        csp_path = os.path.join('saved_models', 'csp_model.pkl')
        if not os.path.exists(csp_path):
            st.error(f"❌ CSP model not found at {csp_path}")
            st.info("Please run `2_train_model.py` first to train and save the models.")
            return None, None
        
        csp = joblib.load(csp_path)
        
        # Load LDA model
        lda_path = os.path.join('saved_models', 'lda_model.pkl')
        if not os.path.exists(lda_path):
            st.error(f"❌ LDA model not found at {lda_path}")
            st.info("Please run `2_train_model.py` first to train and save the models.")
            return None, None
        
        lda = joblib.load(lda_path)
        
        return csp, lda
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load models
csp, lda = load_models()

if csp is None or lda is None:
    st.stop()

st.success("✅ Models loaded successfully!")

# ============================================================================
# Load Test Data
# ============================================================================

@st.cache_data
def load_test_data():
    """
    Load test data for demonstration.
    Uses training file (A01T.gdf) which has more trials (144 total: 72 left + 72 right).
    For demo, we'll create synthetic data or use MNE sample data.
    """
    # Try to load actual BCI Competition data
    # Using training file (T) instead of test file (E) for more trials
    data_path = os.path.join('data', 'BCICIV_2a')
    subject_id = 1
    data_file = os.path.join(data_path, f'A0{subject_id}T.gdf')  # T = training (has 144 trials)
    
    if os.path.exists(data_file):
        # Load actual data
        raw = mne.io.read_raw_gdf(data_file, preload=True, verbose=False)
        
        # Set montage if not already set (BCI Competition uses standard 10-20 system)
        # This is needed for topomap visualization
        # BCI Competition IV Dataset 2a uses 22 EEG channels + 3 EOG channels
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            # Try to set montage, but don't warn if channels don't match exactly
            # The on_missing='ignore' will skip channels that don't match
            raw.set_montage(montage, on_missing='ignore', verbose=False)
        except Exception as e:
            # If montage setting fails, that's okay - we'll use bar chart fallback
            pass
        
        # Apply same preprocessing as training
        raw_filtered = raw.copy().filter(l_freq=8.0, h_freq=30.0, 
                                         method='iir', picks='eeg', verbose=False)
        
        # Extract events
        events, event_dict = mne.events_from_annotations(raw_filtered, verbose=False)
        
        # Use the same event identification logic as training script
        unique_event_codes = np.unique(events[:, 2])
        
        # Look for annotation keys '769' and '770' in event_dict (BCI Competition codes)
        left_code = None
        right_code = None
        
        if '769' in event_dict and '770' in event_dict:
            left_code = event_dict['769']   # MNE-assigned code for annotation '769'
            right_code = event_dict['770']  # MNE-assigned code for annotation '770'
        elif 769 in event_dict and 770 in event_dict:
            left_code = event_dict[769]
            right_code = event_dict[770]
        else:
            # Fallback: find codes with ~72 events (typical for BCI Competition)
            code_counts = {code: np.sum(events[:, 2] == code) for code in unique_event_codes}
            motor_codes = [code for code, count in code_counts.items() if 65 <= count <= 80]
            if len(motor_codes) >= 2:
                left_code = motor_codes[0]
                right_code = motor_codes[1]
        
        if left_code is None or right_code is None:
            st.error("Could not identify left/right hand event codes!")
            st.error(f"Event dict: {event_dict}")
            st.error(f"Unique codes: {unique_event_codes}")
            st.stop()
        
        event_id = {'left_hand': left_code, 'right_hand': right_code}
        
        # Create epochs
        tmin, tmax = -0.5, 4.0
        epochs = mne.Epochs(raw_filtered, events, event_id, tmin, tmax,
                           baseline=(None, 0), preload=True, verbose=False)
        
        return epochs, True
    else:
        # Use MNE sample data for demonstration
        st.info("ℹ️ Using MNE sample dataset for demonstration. For actual BCI data, place test files in data/BCICIV_2a/")
        
        # Load sample data
        sample_data_folder = mne.datasets.sample.data_path()
        sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                           'sample_audvis_raw.fif')
        raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)
        
        # Apply filter
        raw_filtered = raw.copy().filter(l_freq=8.0, h_freq=30.0, 
                                        method='iir', picks='eeg', verbose=False)
        
        # Create synthetic events for demonstration
        sfreq = raw_filtered.info['sfreq']
        # Create alternating left/right events
        events = np.array([[int(i * sfreq * 2), 0, 1 if i % 2 == 0 else 2] 
                          for i in range(20)])
        event_id = {'left_hand': 1, 'right_hand': 2}
        
        # Create epochs
        tmin, tmax = -0.5, 4.0
        epochs = mne.Epochs(raw_filtered, events, event_id, tmin, tmax,
                           baseline=(None, 0), preload=True, verbose=False)
        
        return epochs, False

# Load test epochs
test_epochs, is_real_data = load_test_data()

# ============================================================================
# Helper Functions
# ============================================================================

def get_trial_by_index(epoch_idx, epochs):
    """
    Get a specific trial by its index in the epochs object.
    
    Parameters:
    -----------
    epoch_idx : int
        Index of the trial in the epochs object
    epochs : mne.Epochs
        Epochs object containing the test data
    
    Returns:
    --------
    trial_data : ndarray
        Single trial data (n_channels, n_times)
    true_label : str
        Ground truth label
    trial_number : int
        Trial number (1-indexed for display)
    """
    if epoch_idx < 0 or epoch_idx >= len(epochs):
        return None, None, None
    
    # Get the trial data (n_channels, n_times)
    trial_data = epochs[epoch_idx].get_data()[0]  # [0] to remove epoch dimension
    
    # Get true label from event code using epochs.event_id
    event_code = epochs.events[epoch_idx, 2]
    
    # Use epochs.event_id to determine left/right
    if 'left_hand' in epochs.event_id and 'right_hand' in epochs.event_id:
        left_code = epochs.event_id['left_hand']
        right_code = epochs.event_id['right_hand']
        
        if event_code == left_code:
            true_label = "LEFT"
        elif event_code == right_code:
            true_label = "RIGHT"
        else:
            # Fallback: use lower code as left
            true_label = "LEFT" if event_code < right_code else "RIGHT"
    else:
        # Fallback: try to determine from event codes
        unique_codes = np.unique(epochs.events[:, 2])
        code_counts = {code: np.sum(epochs.events[:, 2] == code) for code in unique_codes}
        motor_codes = sorted([code for code, count in code_counts.items() if 65 <= count <= 80])
        
        if len(motor_codes) >= 2:
            left_code = motor_codes[0]
            right_code = motor_codes[1]
            true_label = "LEFT" if event_code == left_code else "RIGHT"
        else:
            # Last resort: use lower codes as left
            true_label = "LEFT" if event_code < 3 else "RIGHT"
    
    trial_number = epoch_idx + 1  # 1-indexed for display
    
    return trial_data, true_label, trial_number

def get_random_trial(class_type, epochs, seed=None):
    """
    Get a random trial of the specified class.
    
    Parameters:
    -----------
    class_type : str
        'left' or 'right' to specify which class
    epochs : mne.Epochs
        Epochs object containing the test data
    seed : int, optional
        Random seed for reproducibility (if None, uses current time)
    
    Returns:
    --------
    trial_data : ndarray
        Single trial data (n_channels, n_times)
    true_label : str
        Ground truth label
    trial_number : int
        Trial number (1-indexed for display)
    epoch_idx : int
        Index in epochs object (0-indexed)
    """
    # Get the event codes from epochs.event_id
    if 'left_hand' in epochs.event_id and 'right_hand' in epochs.event_id:
        left_code = epochs.event_id['left_hand']
        right_code = epochs.event_id['right_hand']
    else:
        # Fallback: try to determine from event codes
        unique_codes = np.unique(epochs.events[:, 2])
        code_counts = {code: np.sum(epochs.events[:, 2] == code) for code in unique_codes}
        motor_codes = sorted([code for code, count in code_counts.items() if 65 <= count <= 80])
        if len(motor_codes) >= 2:
            left_code = motor_codes[0]
            right_code = motor_codes[1]
        else:
            st.warning("Could not determine left/right event codes")
            return None, None, None, None
    
    # Select the appropriate event code based on class_type
    target_code = left_code if class_type.lower() == 'left' else right_code
    
    # Get indices of trials with the specified label
    trial_indices = np.where(epochs.events[:, 2] == target_code)[0]
    
    if len(trial_indices) == 0:
        st.warning(f"No trials found for {class_type} hand")
        return None, None, None, None
    
    # Use seed if provided, otherwise use current time for true randomness
    if seed is None:
        np.random.seed()  # Reset to use current time
    else:
        np.random.seed(seed)
    
    # Select a random trial
    random_epoch_idx = np.random.choice(trial_indices)
    
    # Get the trial data (n_channels, n_times)
    trial_data = epochs[random_epoch_idx].get_data()[0]  # [0] to remove epoch dimension
    
    # Get true label
    true_label = "LEFT" if class_type.lower() == 'left' else "RIGHT"
    
    trial_number = random_epoch_idx + 1  # 1-indexed for display
    
    return trial_data, true_label, trial_number, random_epoch_idx

def predict_trial(trial_data, csp, lda):
    """
    Apply CSP transform and LDA prediction to a trial.
    
    Parameters:
    -----------
    trial_data : ndarray
        Single trial data (n_channels, n_times)
    csp : CSP object
        Trained CSP transformer
    lda : LDA object
        Trained LDA classifier
    
    Returns:
    --------
    prediction : str
        Predicted label ("LEFT" or "RIGHT")
    csp_features : ndarray
        CSP-transformed features
    """
    # Reshape trial_data to (1, n_channels, n_times) for CSP
    trial_reshaped = trial_data[np.newaxis, :, :]
    
    # Apply CSP transform
    csp_features = csp.transform(trial_reshaped)
    
    # Get prediction
    prediction_proba = lda.predict_proba(csp_features)[0]
    prediction_class = lda.predict(csp_features)[0]
    
    # Convert to label
    # Note: LDA was trained with y_binary where 1=left, 0=right
    # But we need to check the actual mapping
    prediction = "LEFT" if prediction_class == 1 else "RIGHT"
    
    return prediction, csp_features, prediction_proba

def plot_topomap_trial(trial_data, info, times=None):
    """
    Plot topomap visualization of the trial.
    
    Parameters:
    -----------
    trial_data : ndarray
        Single trial data (n_channels, n_times)
    info : mne.Info
        MNE info object with channel information
    times : array-like, optional
        Time points to plot (default: middle of trial)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # If no times specified, plot at the middle of the trial
    if times is None:
        n_times = trial_data.shape[1]
        # Plot at 1 second into the trial (motor imagery period)
        time_idx = int(n_times * 0.3)  # 30% into trial
        times = [trial_data.shape[1] * 0.3 / info['sfreq']]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot topomap
    # Use average power in a time window around the specified time
    time_window = int(0.5 * info['sfreq'])  # 0.5 second window
    time_idx = int(times[0] * info['sfreq'])
    start_idx = max(0, time_idx - time_window // 2)
    end_idx = min(trial_data.shape[1], time_idx + time_window // 2)
    
    # Average power in the time window
    avg_power = np.mean(trial_data[:, start_idx:end_idx] ** 2, axis=1)
    
    # Try to plot topomap, fallback to simple plot if channel locations unavailable
    try:
        # Check if montage/channel locations are available
        montage = info.get_montage()
        if montage is None:
            raise ValueError("No montage set - channel locations unavailable")
        
        # Check if we have enough channel positions
        ch_pos = mne.channels.layout._find_topomap_coords(info, picks=None)
        if ch_pos is None or len(ch_pos) == 0:
            raise ValueError("No channel positions available")
        
        # Plot topomap
        im, _ = mne.viz.plot_topomap(avg_power, info, axes=ax, show=False,
                                     cmap='RdBu_r', vlim=(None, None))
        ax.set_title(f'EEG Topography at {times[0]:.2f}s', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Power (μV²)')
    except Exception as e:
        # Fallback: plot channel power as bar chart if topomap fails
        # This is a perfectly valid visualization when topomap isn't available
        channel_names = info.ch_names[:len(avg_power)] if len(info.ch_names) >= len(avg_power) else [f'Ch{i}' for i in range(len(avg_power))]
        ax.bar(range(len(avg_power)), avg_power, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(avg_power)))
        ax.set_xticklabels(channel_names, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Channel', fontsize=12)
        ax.set_ylabel('Average Power (μV²)', fontsize=12)
        ax.set_title(f'Channel Power Distribution at {times[0]:.2f}s', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        # Don't show warning - bar chart is a valid alternative visualization
    
    plt.tight_layout()
    return fig

# ============================================================================
# Streamlit UI
# ============================================================================

st.markdown("### 🎯 Try It Out!")
st.markdown("**What to do:** Click a button below to load a real brain signal recording. The computer will try to guess whether the person was thinking about moving their left hand or right hand. See if it gets it right!")

# Initialize session state
if 'trial_data' not in st.session_state:
    st.session_state.trial_data = None
    st.session_state.true_label = None
    st.session_state.prediction = None
    st.session_state.trial_number = None
    st.session_state.total_trials = len(test_epochs)

# Display total number of trials
st.info(f"📊 Total trials available: {st.session_state.total_trials}")

# Create columns for buttons and trial number input
col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    load_left_button = st.button("🫲 Load random 'LEFT' trial", 
                                 use_container_width=True, 
                                 type="primary",
                                 key="load_left")

with col2:
    load_right_button = st.button("🫱 Load random 'RIGHT' trial", 
                                  use_container_width=True, 
                                  type="primary",
                                  key="load_right")

with col3:
    st.markdown("**Or load by trial number:**")
    trial_number_input = st.number_input(
        "Trial #",
        min_value=1,
        max_value=st.session_state.total_trials,
        value=1,
        step=1,
        key="trial_number_input"
    )
    load_specific_button = st.button("📌 Load Trial", 
                                     use_container_width=True,
                                     key="load_specific")

# Handle button clicks
if load_left_button:
    # Use current time as seed to ensure different trials on each click
    seed = int(time.time() * 1000) % 1000000
    trial_data, true_label, trial_number, epoch_idx = get_random_trial('left', test_epochs, seed=seed)
    if trial_data is not None:
        st.session_state.trial_data = trial_data
        st.session_state.true_label = true_label
        st.session_state.trial_number = trial_number
        st.session_state.epoch_idx = epoch_idx
        # Get prediction
        prediction, csp_features, proba = predict_trial(trial_data, csp, lda)
        st.session_state.prediction = prediction
        st.session_state.csp_features = csp_features
        st.session_state.proba = proba
        st.rerun()  # Force rerun to show new trial

if load_right_button:
    # Use current time as seed to ensure different trials on each click
    seed = int(time.time() * 1000) % 1000000
    trial_data, true_label, trial_number, epoch_idx = get_random_trial('right', test_epochs, seed=seed)
    if trial_data is not None:
        st.session_state.trial_data = trial_data
        st.session_state.true_label = true_label
        st.session_state.trial_number = trial_number
        st.session_state.epoch_idx = epoch_idx
        # Get prediction
        prediction, csp_features, proba = predict_trial(trial_data, csp, lda)
        st.session_state.prediction = prediction
        st.session_state.csp_features = csp_features
        st.session_state.proba = proba
        st.rerun()  # Force rerun to show new trial

if load_specific_button:
    epoch_idx = trial_number_input - 1  # Convert to 0-indexed
    trial_data, true_label, trial_number = get_trial_by_index(epoch_idx, test_epochs)
    if trial_data is not None:
        st.session_state.trial_data = trial_data
        st.session_state.true_label = true_label
        st.session_state.trial_number = trial_number
        st.session_state.epoch_idx = epoch_idx
        # Get prediction
        prediction, csp_features, proba = predict_trial(trial_data, csp, lda)
        st.session_state.prediction = prediction
        st.session_state.csp_features = csp_features
        st.session_state.proba = proba
        st.rerun()  # Force rerun to show new trial

# Display results if a trial has been loaded
if st.session_state.trial_data is not None:
    st.markdown("---")
    
    # Display trial number prominently
    st.markdown(f"### 🔢 Trial #{st.session_state.trial_number} of {st.session_state.total_trials}")
    
    # Create columns for results
    result_col1, result_col2 = st.columns(2)
    
    with result_col1:
        st.markdown("### 🤖 Computer's Guess")
        prediction_color = "🟢" if st.session_state.prediction == st.session_state.true_label else "🔴"
        st.markdown(f"**{prediction_color} The computer thinks:** `{st.session_state.prediction}`")
        
        # Show prediction confidence
        if hasattr(st.session_state, 'proba'):
            confidence = max(st.session_state.proba) * 100
            st.metric("How Sure?", f"{confidence:.1f}%")
            if confidence > 80:
                st.caption("Very confident! 🎯")
            elif confidence > 60:
                st.caption("Pretty sure 👍")
            else:
                st.caption("Not very sure 🤔")
    
    with result_col2:
        st.markdown("### 📝 What Actually Happened")
        st.markdown(f"**The person was thinking about:** `{st.session_state.true_label}`")
        
        # Show if prediction is correct
        is_correct = st.session_state.prediction == st.session_state.true_label
        if is_correct:
            st.success("🎉 The computer got it right!")
        else:
            st.error("❌ The computer guessed wrong this time")
    
    # Display visualization
    st.markdown("---")
    st.markdown("### 🗺️ Brain Activity Map")
    st.markdown("**What you're seeing:** This map shows which parts of the brain were most active when the person was thinking about moving their hand. Red areas = more active, Blue areas = less active. The computer uses patterns like this to make its guess!")
    
    # Get info from test epochs
    info = test_epochs.info
    
    # Plot topomap
    fig = plot_topomap_trial(st.session_state.trial_data, info)
    st.pyplot(fig)
    
    # Additional information
    with st.expander("📈 Additional Information"):
        st.markdown(f"**Trial Number:** {st.session_state.trial_number} (epoch index: {st.session_state.epoch_idx})")
        st.markdown(f"**Trial Shape:** {st.session_state.trial_data.shape}")
        st.markdown(f"**Number of Channels:** {st.session_state.trial_data.shape[0]}")
        st.markdown(f"**Number of Time Points:** {st.session_state.trial_data.shape[1]}")
        st.markdown(f"**Sampling Rate:** {info['sfreq']} Hz")
        
        if hasattr(st.session_state, 'csp_features'):
            st.markdown(f"**CSP Features Shape:** {st.session_state.csp_features.shape}")
            st.markdown(f"**CSP Features:** {st.session_state.csp_features[0]}")

# Footer
st.markdown("---")
st.markdown("### ℹ️ Technical Details")
with st.expander("Click to see technical information"):
    st.markdown("""
    This application uses:
    - **CSP (Common Spatial Patterns)** - Finds the best way to combine signals from different brain sensors
    - **LDA (Linear Discriminant Analysis)** - Makes the final decision about left vs. right
    
    The data comes from the BCI Competition IV Dataset 2a, which contains real EEG recordings from people 
    performing motor imagery tasks.
    """)

# Instructions
with st.sidebar:
    st.markdown("### 📋 Quick Guide")
    st.markdown("""
    **How to use:**
    1. Click a button to load a brain signal recording
    2. See what the computer guessed
    3. Check if it was right!
    4. Look at the brain activity map to see what the computer "saw"
    
    **Tips:**
    - Try loading multiple trials to see how accurate the computer is
    - Higher "How Sure?" percentage = computer is more confident
    - Green circle = correct guess, Red circle = wrong guess
    """)
    
    st.markdown("### 🔧 Model Information")
    if csp is not None:
        st.markdown(f"**CSP Components:** {csp.n_components}")
    if lda is not None:
        st.markdown(f"**LDA Solver:** {lda.solver}")
