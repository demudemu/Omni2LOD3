import os
import laspy
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.spatial import cKDTree
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Define the classes
CLASSES = ["Wall", "Window", "Door", "Building Installation"]

def get_file_path(title):
    """Opens a file dialog to select a single .las file."""
    root = tk.Tk()
    root.withdraw() # Hide the main tkinter window
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[("LAS files", "*.las"), ("All files", "*.*")]
    )
    return file_path

def get_save_dir(title):
    """Opens a dialog to select a directory for saving output."""
    root = tk.Tk()
    root.withdraw()
    dir_path = filedialog.askdirectory(title=title)
    return dir_path

def load_class_files(phase_name):
    """Prompts the user to select files for each class and loads their coordinates."""
    coords_list = []
    labels_list = []
    
    for i, cls_name in enumerate(CLASSES):
        print(f"Waiting for user to select {phase_name} file for class: {cls_name}...")
        file_path = get_file_path(f"Select {phase_name} .las file for {cls_name}")
        
        if not file_path:
            print("Selection cancelled. Exiting.")
            exit()
            
        print(f"Loading {file_path}...")
        las = laspy.read(file_path)
        
        # Extract X, Y, Z coordinates and stack them into an N x 3 array
        coords = np.vstack((las.x, las.y, las.z)).transpose()
        labels = np.full((coords.shape[0],), i) # Assign integer label (0 to 3)
        
        coords_list.append(coords)
        labels_list.append(labels)
        
    # Combine all classes into one large array
    all_coords = np.vstack(coords_list)
    all_labels = np.concatenate(labels_list)
    return all_coords, all_labels

def main():
    print("--- STEP 1: Select Reference (Ground Truth) Files ---")
    ref_coords, ref_labels = load_class_files("REFERENCE (Ground Truth)")
    
    print("\n--- STEP 2: Select Test (Output) Files ---")
    test_coords, test_labels = load_class_files("TEST (Output)")
    
    print("\n--- STEP 3: Matching Points Spatially ---")
    print("Building KD-Tree for test points (this may take a moment for large clouds)...")
    tree = cKDTree(test_coords)
    
    print("Querying nearest neighbors to match reference points to test points...")
    # distance_upper_bound ensures we only match points that actually share the same physical location
    # 0.05 meters (5cm) tolerance is used to account for minor floating-point shifts.
    distances, indices = tree.query(ref_coords, distance_upper_bound=0.05)
    
    # Filter out reference points that didn't find a matching test point within the tolerance
    valid_matches = distances != float('inf')
    matched_ref_labels = ref_labels[valid_matches]
    matched_test_labels = test_labels[indices[valid_matches]]
    
    dropped_points = len(ref_labels) - np.sum(valid_matches)
    if dropped_points > 0:
        print(f"Warning: {dropped_points} reference points could not be matched to any test point and will be excluded.")

    print("\n--- STEP 4: Computing Metrics ---")
    # 4x4 Confusion Matrix
    cm = confusion_matrix(matched_ref_labels, matched_test_labels, labels=[0, 1, 2, 3])
    
    # Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        matched_ref_labels, matched_test_labels, labels=[0, 1, 2, 3], zero_division=0
    )
    
    print("\n--- STEP 5: Saving Output ---")
    save_dir = get_save_dir("Select folder to save the output CSVs")
    if not save_dir:
        print("No save directory selected. Outputting to current working directory.")
        save_dir = os.getcwd()

    # Format Confusion Matrix as a DataFrame for readability
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in CLASSES], columns=[f"Pred_{c}" for c in CLASSES])
    
    # Format Metrics as a DataFrame
    metrics_df = pd.DataFrame({
        'Class': CLASSES,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'Support (Points)': support
    })
    
    # Save to CSV
    cm_path = os.path.join(save_dir, "confusion_matrix.csv")
    metrics_path = os.path.join(save_dir, "classification_metrics.csv")
    
    cm_df.to_csv(cm_path)
    metrics_df.to_csv(metrics_path, index=False)
    
    print(f"\nSuccess! Files saved to:")
    print(f"1. {cm_path}")
    print(f"2. {metrics_path}")

    # Display results in the console as well
    print("\n--- Confusion Matrix ---")
    print(cm_df)
    print("\n--- Classification Metrics ---")
    print(metrics_df)

if __name__ == "__main__":
    main()