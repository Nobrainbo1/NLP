import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import tensorflow as tf

# Create benchmark_results directory if it doesn't exist
if not os.path.exists('benchmark_results'):
    os.makedirs('benchmark_results')

def run_initial_approach():
    from initial_approach.train_model import main as nb_main
    from initial_approach.train_model_2 import main as lr_main
    
    # Run and time Naive Bayes
    start_time = time.time()
    nb_results = nb_main()
    nb_time = time.time() - start_time
    
    # Run and time Logistic Regression
    start_time = time.time()
    lr_results = lr_main()
    lr_time = time.time() - start_time
    
    return {
        'naive_bayes': {
            'time': nb_time,
            'results': nb_results
        },
        'logistic_regression': {
            'time': lr_time,
            'results': lr_results
        }
    }

def run_v2_approach():
    from v2.ML_train_model import main as nb_v2_main
    from v2.ML_train_model_2 import main as lr_v2_main
    from v2.LSTM_train_model import main as lstm_main
    
    # Run and time V2 Naive Bayes
    start_time = time.time()
    nb_v2_results = nb_v2_main()
    nb_v2_time = time.time() - start_time
    
    # Run and time V2 Logistic Regression
    start_time = time.time()
    lr_v2_results = lr_v2_main()
    lr_v2_time = time.time() - start_time
    
    # Run and time LSTM
    start_time = time.time()
    lstm_results = lstm_main()
    lstm_time = time.time() - start_time
    
    return {
        'naive_bayes_v2': {
            'time': nb_v2_time,
            'results': nb_v2_results
        },
        'logistic_regression_v2': {
            'time': lr_v2_time,
            'results': lr_v2_results
        },
        'lstm': {
            'time': lstm_time,
            'results': lstm_results
        }
    }

def plot_results(initial_results, v2_results):
    # Prepare data for plotting
    models = []
    accuracies = []
    times = []
    
    # Add initial approach results
    for model, data in initial_results.items():
        models.append(f"Initial {model}")
        accuracies.append(data['results']['accuracy'])
        times.append(data['time'])
    
    # Add V2 approach results
    for model, data in v2_results.items():
        models.append(f"V2 {model}")
        accuracies.append(data['results']['accuracy'])
        times.append(data['time'])
    
    # Plot accuracy comparison
    plt.figure(figsize=(12, 6))
    plt.bar(models, accuracies)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('benchmark_results/accuracy_comparison.png')
    plt.close()
    
    # Plot training time comparison
    plt.figure(figsize=(12, 6))
    plt.bar(models, times)
    plt.title('Training Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('benchmark_results/time_comparison.png')
    plt.close()

def save_results(initial_results, v2_results):
    results_df = pd.DataFrame(columns=['Model', 'Approach', 'Accuracy', 'Training Time'])
    
    # Add initial approach results
    for model, data in initial_results.items():
        results_df = results_df.append({
            'Model': model,
            'Approach': 'Initial',
            'Accuracy': data['results']['accuracy'],
            'Training Time': data['time']
        }, ignore_index=True)
    
    # Add V2 approach results
    for model, data in v2_results.items():
        results_df = results_df.append({
            'Model': model,
            'Approach': 'V2',
            'Accuracy': data['results']['accuracy'],
            'Training Time': data['time']
        }, ignore_index=True)
    
    results_df.to_csv('benchmark_results/benchmark_results.csv', index=False)

def main():
    print("Running Initial Approach Benchmark...")
    initial_results = run_initial_approach()
    
    print("\nRunning V2 Approach Benchmark...")
    v2_results = run_v2_approach()
    
    print("\nGenerating Plots...")
    plot_results(initial_results, v2_results)
    
    print("\nSaving Results...")
    save_results(initial_results, v2_results)
    
    print("\nBenchmark completed! Results saved in 'benchmark_results' directory.")

if __name__ == "__main__":
    main() 