import click
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from ..pointnet import create_pointnet_model

def generate_log_train_sizes(min_size=50, max_size=16000, num_points=20):
    """Generate logarithmically spaced training set sizes."""
    log_min = np.log10(min_size)
    log_max = np.log10(max_size)
    log_sizes = np.linspace(log_min, log_max, num_points)
    sizes = np.round(10**log_sizes).astype(int)
    # Remove duplicates and ensure we have the exact min and max
    sizes = np.unique(sizes)
    if min_size not in sizes:
        sizes = np.append([min_size], sizes)
    if max_size not in sizes:
        sizes = np.append(sizes, [max_size])
    return np.sort(sizes)

@click.command()
@click.option('--beam', default='O16', type=click.STRING, help='The beam to train on (e.g. O16, Mg22, C16)')
@click.option('--num-points', default=512, type=click.INT, help='Number of points per event')
@click.option('--num-classes', default=2, type=click.INT, help='Number of classes to predict')
@click.option('--test-size', default=0.2, type=click.FLOAT, help='Fraction of data to use for testing')
@click.option('--random-state', default=42, type=click.INT, help='Random state for reproducibility')
@click.option('--regularization', default=1.0, type=click.FLOAT, help='Regularization strength for logistic regression')
@click.option('--min-train-size', default=50, type=click.INT, help='Minimum training set size')
@click.option('--max-train-size', default=16000, type=click.INT, help='Maximum training set size')
@click.option('--num-size-points', default=20, type=click.INT, help='Number of training sizes to test')
@click.option('--cv-folds', default=3, type=click.INT, help='Number of cross-validation folds for each size')
@click.argument('model-folder')
@click.argument('data-file-stem-2track')
@click.argument('data-file-stem-3track')

def linear_probe_evaluation(beam, num_points, num_classes, test_size, random_state, 
                          regularization, min_train_size, max_train_size, num_size_points,
                          cv_folds, model_folder, data_file_stem_2track, data_file_stem_3track):
    """
    Perform linear probe evaluation with learning curve analysis.
    
    This function:
    1. Loads the trained PointNet model
    2. Extracts global features from the latent space
    3. Tests linear classifiers with logarithmically growing training set sizes
    4. Evaluates performance at each training size
    5. Creates learning curve visualizations
    """
    
    print("Loading PointNet model...")
    # load model
    model = tf.keras.models.load_model(model_folder)
    print("Model loaded successfully")
    
    # Define batch size
    BATCH_SIZE = 32
    
    print("Loading datasets...")
    # Load both datasets
    test_ds_2track = np.load(f'{data_file_stem_2track}.npy')
    test_ds_3track = np.load(f'{data_file_stem_3track}.npy')
    
    print(f"2-track dataset shape: {test_ds_2track.shape}")
    print(f"3-track dataset shape: {test_ds_3track.shape}")
    
    # Create track type labels (0 for 2-track, 1 for 3-track)
    track_labels_2 = np.zeros(len(test_ds_2track), dtype=int)
    track_labels_3 = np.ones(len(test_ds_3track), dtype=int)
    
    # Combine datasets (use first 4 features as in your original code)
    combined_features = np.concatenate([test_ds_2track[:, :, :4], test_ds_3track[:, :, :4]], axis=0)
    combined_track_labels = np.concatenate([track_labels_2, track_labels_3])
    
    print(f"Combined dataset shape: {combined_features.shape}")
    print(f"Combined labels shape: {combined_track_labels.shape}")
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(combined_features).batch(BATCH_SIZE)
    
    print("Extracting global features from PointNet latent space...")
    # Extract global features from the latent space
    try:
        global_feature_extractor = keras.Model(inputs=model.input, outputs=model.get_layer("latent_space").output)
    except ValueError:
        # If "latent_space" layer doesn't exist, try to find a suitable layer
        print("'latent_space' layer not found. Available layers:")
        for i, layer in enumerate(model.layers):
            print(f"  {i}: {layer.name} - {type(layer).__name__}")
        
        # Try to use the layer before the final classification layer
        global_feature_extractor = keras.Model(inputs=model.input, 
                                             outputs=model.layers[-3].output)
        print(f"Using layer '{model.layers[-3].name}' for feature extraction")
    
    # Extract features
    global_features = global_feature_extractor.predict(dataset, verbose=1)
    print(f"Extracted global features shape: {global_features.shape}")
    
    # Create results folder
    results_folder = f"./{beam}_linear_probe_learning_curve"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Split data into train and test sets (fixed test set for consistent evaluation)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        global_features, combined_track_labels, 
        test_size=test_size, random_state=random_state, 
        stratify=combined_track_labels
    )
    
    print(f"Full training set size: {X_train_full.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Check if we have enough data for the maximum training size
    max_possible_size = min(max_train_size, X_train_full.shape[0])
    if max_possible_size < max_train_size:
        print(f"Warning: Requested max training size ({max_train_size}) exceeds available data.")
        print(f"Using maximum available size: {max_possible_size}")
        max_train_size = max_possible_size
    
    # Generate logarithmic training sizes
    train_sizes = generate_log_train_sizes(min_train_size, max_train_size, num_size_points)
    print(f"Training sizes to test: {train_sizes}")
    
    # Standardize features (fit on full training set)
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize results storage
    learning_curve_results = {
        'train_sizes': train_sizes.tolist(),
        'train_accuracies_mean': [],
        'train_accuracies_std': [],
        'test_accuracies_mean': [],
        'test_accuracies_std': [],
        'detailed_results': []
    }
    
    print(f"\nStarting learning curve analysis with {len(train_sizes)} training sizes...")
    print("=" * 60)
    
    # For each training size
    for i, train_size in enumerate(train_sizes):
        print(f"\nProgress: {i+1}/{len(train_sizes)} - Training size: {train_size}")
        
        train_accs = []
        test_accs = []
        
        # Perform multiple runs with different random subsets for robustness
        for fold in range(cv_folds):
            # Randomly sample training data of the specified size
            if train_size >= len(X_train_full_scaled):
                # Use all available training data
                X_train_subset = X_train_full_scaled
                y_train_subset = y_train_full
            else:
                # Stratified sampling to maintain class balance
                indices = []
                for class_label in np.unique(y_train_full):
                    class_indices = np.where(y_train_full == class_label)[0]
                    n_samples_per_class = train_size // num_classes
                    if class_label == np.unique(y_train_full)[-1]:  # Last class gets remainder
                        n_samples_per_class += train_size % num_classes
                    
                    selected_indices = np.random.choice(
                        class_indices, 
                        size=min(n_samples_per_class, len(class_indices)), 
                        replace=False
                    )
                    indices.extend(selected_indices)
                
                np.random.shuffle(indices)
                X_train_subset = X_train_full_scaled[indices]
                y_train_subset = y_train_full[indices]
            
            # Train linear probe
            linear_probe = LogisticRegression(
                C=regularization, 
                random_state=random_state + fold,
                max_iter=1000
            )
            linear_probe.fit(X_train_subset, y_train_subset)
            
            # Evaluate
            train_pred = linear_probe.predict(X_train_subset)
            test_pred = linear_probe.predict(X_test_scaled)
            
            train_acc = accuracy_score(y_train_subset, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)
        
        # Calculate statistics
        train_acc_mean = np.mean(train_accs)
        train_acc_std = np.std(train_accs)
        test_acc_mean = np.mean(test_accs)
        test_acc_std = np.std(test_accs)
        
        # Store results
        learning_curve_results['train_accuracies_mean'].append(train_acc_mean)
        learning_curve_results['train_accuracies_std'].append(train_acc_std)
        learning_curve_results['test_accuracies_mean'].append(test_acc_mean)
        learning_curve_results['test_accuracies_std'].append(test_acc_std)
        
        # Store detailed results
        detailed_result = {
            'train_size': int(train_size),
            'train_accuracy_mean': float(train_acc_mean),
            'train_accuracy_std': float(train_acc_std),
            'test_accuracy_mean': float(test_acc_mean),
            'test_accuracy_std': float(test_acc_std),
            'train_accuracies': [float(acc) for acc in train_accs],
            'test_accuracies': [float(acc) for acc in test_accs]
        }
        learning_curve_results['detailed_results'].append(detailed_result)
        
        print(f"  Train Acc: {train_acc_mean:.4f} ± {train_acc_std:.4f}")
        print(f"  Test Acc:  {test_acc_mean:.4f} ± {test_acc_std:.4f}")
    
    print("\n" + "=" * 60)
    print("Learning curve analysis completed!")
    
    # Save detailed results
    full_results = {
        'model_folder': model_folder,
        'dataset_info': {
            '2track_file': data_file_stem_2track,
            '3track_file': data_file_stem_3track,
            'total_samples': len(combined_features),
            'feature_dim': global_features.shape[1]
        },
        'experiment_config': {
            'test_size': test_size,
            'regularization': regularization,
            'random_state': random_state,
            'cv_folds': cv_folds,
            'min_train_size': min_train_size,
            'max_train_size': max_train_size,
            'num_size_points': num_size_points
        },
        'learning_curve_results': learning_curve_results
    }
    
    with open(f'{results_folder}/learning_curve_results.json', 'w') as f:
        json.dump(full_results, f, indent=4)
    
    # Create visualizations
    create_learning_curve_visualizations(learning_curve_results, results_folder)
    
    # Train final model with full training data for additional analysis
    print("\nTraining final model with full training data...")
    final_linear_probe = LogisticRegression(
        C=regularization, 
        random_state=random_state,
        max_iter=1000
    )
    final_linear_probe.fit(X_train_full_scaled, y_train_full)
    
    # Final evaluation
    y_train_final_pred = final_linear_probe.predict(X_train_full_scaled)
    y_test_final_pred = final_linear_probe.predict(X_test_scaled)
    y_test_final_prob = final_linear_probe.predict_proba(X_test_scaled)
    
    final_train_acc = accuracy_score(y_train_full, y_train_final_pred)
    final_test_acc = accuracy_score(y_test, y_test_final_pred)
    
    print(f"Final model - Train Acc: {final_train_acc:.4f}, Test Acc: {final_test_acc:.4f}")
    
    # Create additional visualizations for final model
    class_names = ['2-track', '3-track']
    create_final_model_visualizations(
        X_test_scaled, y_test, y_test_final_pred, y_test_final_prob,
        class_names, results_folder, final_linear_probe
    )
    
    print(f"\nAll results saved to: {results_folder}")
    
    return {
        'learning_curve_results': learning_curve_results,
        'final_model': final_linear_probe,
        'scaler': scaler,
        'final_train_accuracy': final_train_acc,
        'final_test_accuracy': final_test_acc
    }

def create_performance_table(results, results_folder):
    """Create a detailed performance table."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['Training Size', 'Train Acc (Mean)', 'Train Acc (Std)', 
               'Test Acc (Mean)', 'Test Acc (Std)', 'Overfitting Gap']
    
    table_data = []
    for i, size in enumerate(results['train_sizes']):
        gap = results['train_accuracies_mean'][i] - results['test_accuracies_mean'][i]
        row = [
            f"{size:,}",
            f"{results['train_accuracies_mean'][i]:.4f}",
            f"{results['train_accuracies_std'][i]:.4f}",
            f"{results['test_accuracies_mean'][i]:.4f}",
            f"{results['test_accuracies_std'][i]:.4f}",
            f"{gap:.4f}"
        ]
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Detailed Learning Curve Results', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(f'{results_folder}/performance_table.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    linear_probe_evaluation()