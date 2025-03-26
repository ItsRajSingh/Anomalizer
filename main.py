import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import shap
import gtts
import pygame
import os
import time
import warnings
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import threading
from pathlib import Path
import joblib
try:
    import mplcursors
    MPLCURSORS_AVAILABLE = True
except ImportError:
    MPLCURSORS_AVAILABLE = False

# Optional imports for Excel handling
try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Create custom colormap for advanced visualizations
anomaly_cmap = LinearSegmentedColormap.from_list(
    "anomaly_cmap", ["#4575B4", "#FFFFBF", "#D73027"], N=256)

class ExplainableAnomalyDetector:
    """
    An explainable AI system that detects anomalies in datasets and provides
    visual and voice explanations of the detected anomalies.
    """
    
    def __init__(self, contamination=0.05, random_state=42):
        """
        Initialize the anomaly detector.
        
        Parameters:
        -----------
        contamination : float, default=0.05
            The expected proportion of outliers in the data.
        random_state : int, default=42
            Random state for reproducibility.
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self.pca = None
        self.tsne = None
        self.feature_names = None
        self.explainer = None
        self.df = None
        self.anomalies = None
        self.anomaly_indices = None
        self.temp_audio_file = "explanation.mp3"
        self.result_df = None
        self.X_tsne = None
        pygame.mixer.init()
    
    def fit(self, df):
        """
        Fit the anomaly detection model to the dataset.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The input dataset.
        
        Returns:
        --------
        self : object
            Returns self.
        """
        self.df = df.copy()
        self.feature_names = df.columns.tolist()
        
        # Scale the data
        X = self.scaler.fit_transform(df)
        
        # Fit the model
        self.model.fit(X)
        
        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        
        # Fit PCA for visualization if there are more than 2 features
        if df.shape[1] > 2:
            self.pca = PCA(n_components=2)
            self.pca.fit(X)
            
            # Also initialize t-SNE for alternative visualization
            self.tsne = TSNE(n_components=2, random_state=self.random_state, 
                           perplexity=min(30, max(5, df.shape[0]//10)))
        
        return self
    
    def detect_anomalies(self):
        """
        Detect anomalies in the fitted dataset.
        
        Returns:
        --------
        pandas.DataFrame
            A DataFrame containing the original data with additional columns
            indicating whether each instance is an anomaly.
        """
        if self.df is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Scale the data
        X = self.scaler.transform(self.df)
        
        # Predict anomalies
        # -1 for anomalies, 1 for normal points
        y_pred = self.model.predict(X)
        
        # Decision function gives the "anomaly score"
        anomaly_scores = self.model.decision_function(X)
        
        # Normalize anomaly scores to [0,1] for better visualization
        normalized_scores = self.min_max_scaler.fit_transform(-anomaly_scores.reshape(-1, 1)).ravel()
        
        # Create a DataFrame with the results
        result_df = self.df.copy()
        result_df['anomaly'] = y_pred == -1
        result_df['anomaly_score'] = anomaly_scores
        result_df['normalized_score'] = normalized_scores
        
        # Store anomaly indices for later use
        self.anomalies = result_df[result_df['anomaly']].copy()
        self.anomaly_indices = self.anomalies.index.tolist()
        self.result_df = result_df
        
        return result_df
    
    def explain_anomalies(self, max_anomalies=5):
        """
        Generate explanations for the detected anomalies.
        
        Parameters:
        -----------
        max_anomalies : int, default=5
            Maximum number of anomalies to explain.
        
        Returns:
        --------
        list
            A list of textual explanations for the anomalies.
        """
        if self.anomalies is None:
            raise ValueError("No anomalies detected yet. Call detect_anomalies() first.")
        
        explanations = []
        
        # Get the top anomalies based on anomaly score
        top_anomalies = self.anomalies.sort_values('anomaly_score').head(max_anomalies)
        
        for idx, row in top_anomalies.iterrows():
            # Get the original data point
            data_point = row[self.feature_names].values.reshape(1, -1)
            
            # Scale the data point
            scaled_point = self.scaler.transform(data_point)
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(scaled_point)[0]
            
            # Pair features with their SHAP values
            feature_impacts = list(zip(self.feature_names, shap_values))
            
            # Sort by absolute impact
            feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Generate explanation
            explanation = f"Anomaly detected at index {idx} with anomaly score {row['anomaly_score']:.4f}.\n"
            explanation += "This data point is unusual because:\n"
            
            for feature, impact in feature_impacts[:3]:  # Top 3 contributing features
                feature_value = row[feature]
                avg_value = self.df[feature].mean()
                std_value = self.df[feature].std()
                
                if feature_value > avg_value + 2 * std_value:
                    comparison = "much higher than"
                elif feature_value > avg_value + std_value:
                    comparison = "higher than"
                elif feature_value < avg_value - 2 * std_value:
                    comparison = "much lower than"
                elif feature_value < avg_value - std_value:
                    comparison = "lower than"
                else:
                    comparison = "unusual compared to"
                
                explanation += f"- {feature} is {comparison} normal with a value of {feature_value:.2f} "
                explanation += f"(average is {avg_value:.2f})\n"
            
            explanations.append(explanation)
        
        return explanations
    
    def visualize_anomalies(self, projection_method='pca', figsize=(10, 6)):
        """
        Visualize the detected anomalies.
        
        Parameters:
        -----------
        projection_method : str, default='pca'
            Method to use for dimensionality reduction. Options: 'pca', 'tsne', or None.
        figsize : tuple, default=(10, 6)
            Figure size for the plot.
            
        Returns:
        --------
        matplotlib.figure.Figure
            A figure containing the visualization.
        """
        if self.anomalies is None:
            raise ValueError("No anomalies detected yet. Call detect_anomalies() first.")
        
        # Create a copy of the dataframe with anomaly column
        df_with_anomalies = self.df.copy()
        df_with_anomalies['anomaly'] = False
        df_with_anomalies.loc[self.anomaly_indices, 'anomaly'] = True
        df_with_anomalies['normalized_score'] = self.result_df['normalized_score']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # If there are only 2 features, plot them directly
        if len(self.feature_names) == 2:
            scatter = ax.scatter(
                df_with_anomalies[self.feature_names[0]],
                df_with_anomalies[self.feature_names[1]],
                c=df_with_anomalies['normalized_score'],
                cmap=anomaly_cmap,
                s=50,
                alpha=0.8,
                edgecolors='k',
                linewidths=0.5
            )
            ax.set_xlabel(self.feature_names[0], fontsize=12)
            ax.set_ylabel(self.feature_names[1], fontsize=12)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Anomaly Score')
            
            # Highlight anomalies with a different marker
            ax.scatter(
                df_with_anomalies[df_with_anomalies['anomaly']][self.feature_names[0]],
                df_with_anomalies[df_with_anomalies['anomaly']][self.feature_names[1]],
                s=120,
                facecolors='none',
                edgecolors='red',
                linewidths=2,
                label='Anomaly'
            )
        
        # If there are more than 2 features, use chosen projection method
        else:
            X = self.scaler.transform(self.df)
            
            if projection_method == 'pca':
                # Use PCA for visualization
                X_2d = self.pca.transform(X)
                title_prefix = 'PCA'
                xlabel = 'Principal Component 1'
                ylabel = 'Principal Component 2'
            
            elif projection_method == 'tsne':
                # Use t-SNE for visualization
                if self.X_tsne is None:
                    self.X_tsne = self.tsne.fit_transform(X)
                X_2d = self.X_tsne
                title_prefix = 't-SNE'
                xlabel = 't-SNE Component 1'
                ylabel = 't-SNE Component 2'
            
            else:
                raise ValueError("projection_method must be 'pca', 'tsne', or None")
            
            # Create a DataFrame with projection results
            proj_df = pd.DataFrame(X_2d, columns=['Comp1', 'Comp2'])
            proj_df['anomaly'] = False
            proj_df.loc[self.anomaly_indices, 'anomaly'] = True
            proj_df['normalized_score'] = self.result_df['normalized_score']
            
            # Plot normal points with color gradient based on anomaly score
            scatter = ax.scatter(
                proj_df['Comp1'],
                proj_df['Comp2'],
                c=proj_df['normalized_score'],
                cmap=anomaly_cmap,
                s=50,
                alpha=0.8,
                edgecolors='k',
                linewidths=0.5
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Anomaly Score')
            
            # Highlight anomalies with a different marker
            ax.scatter(
                proj_df[proj_df['anomaly']]['Comp1'],
                proj_df[proj_df['anomaly']]['Comp2'],
                s=120,
                facecolors='none',
                edgecolors='red',
                linewidths=2,
                label='Anomaly'
            )
            
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            title_prefix = f"{title_prefix} Projection"
        
        ax.set_title(f'Anomaly Detection Visualization ({title_prefix})', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add annotations on hover using mplcursors if available
        if MPLCURSORS_AVAILABLE:
            cursor = mplcursors.cursor(scatter, hover=True)
            
            @cursor.connect("add")
            def on_add(sel):
                index = sel.target.index
                point_info = f"Index: {index}"
                if index in self.anomaly_indices:
                    point_info += " (ANOMALY)"
                for feature in self.feature_names:
                    point_info += f"\n{feature}: {self.df.iloc[index][feature]:.2f}"
                sel.annotation.set_text(point_info)
        
        plt.tight_layout()
        return fig
    
    def _text_to_speech(self, text):
        """
        Convert text to speech and save as an audio file.
        
        Parameters:
        -----------
        text : str
            The text to convert to speech.
        
        Returns:
        --------
        str
            The path to the saved audio file.
        """
        tts = gtts.gTTS(text=text, lang='en')
        tts.save(self.temp_audio_file)
        return self.temp_audio_file
    
    def explain_with_voice(self, max_anomalies=3):
        """
        Generate and play voice explanations for the detected anomalies.
        
        Parameters:
        -----------
        max_anomalies : int, default=3
            Maximum number of anomalies to explain.
        """
        explanations = self.explain_anomalies(max_anomalies)
        
        for i, explanation in enumerate(explanations):
            print(f"Explanation {i+1}:")
            print(explanation)
            
            try:
                print("\nTrying to play voice explanation...")
                
                # Convert explanation to speech
                audio_file = self._text_to_speech(explanation)
                
                # Play the audio using pygame
                pygame.mixer.music.load(audio_file)  # Load the audio file
                pygame.mixer.music.play()  # Play the audio
                while pygame.mixer.music.get_busy():  # Wait for the sound to finish playing
                    pygame.time.Clock().tick(10)  # Check every 100 ms
                time.sleep(1)  # Add a short pause between explanations
            except Exception as e:
                print(f"Could not play audio explanation due to error: {str(e)}")
                print("Continuing with text explanations only.")
        
        # Clean up the audio file
        try:
            if os.path.exists(self.temp_audio_file):
                os.remove(self.temp_audio_file)
        except Exception as e:
            print(f"Could not remove temporary audio file: {str(e)}")
    
    def plot_feature_importance(self, anomaly_idx=None, figsize=(10, 6)):
        """
        Plot the feature importance for a specific anomaly or for all anomalies.
        
        Parameters:
        -----------
        anomaly_idx : int, optional
            The index of the specific anomaly to explain. If None, plot average importance.
        figsize : tuple, default=(10, 6)
            Figure size for the plot.
            
        Returns:
        --------
        matplotlib.figure.Figure
            A figure containing the feature importance plot.
        """
        if self.anomalies is None:
            raise ValueError("No anomalies detected yet. Call detect_anomalies() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if anomaly_idx is not None:
            # Check if the specified index is an anomaly
            if anomaly_idx not in self.anomaly_indices:
                raise ValueError(f"Index {anomaly_idx} is not an anomaly.")
            
            # Get the data point
            data_point = self.df.loc[anomaly_idx, self.feature_names].values.reshape(1, -1)
            
            # Scale the data point
            scaled_point = self.scaler.transform(data_point)
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(scaled_point)[0]
            
            # Pair features with their SHAP values
            feature_impacts = list(zip(self.feature_names, shap_values))
            
            # Sort by absolute impact
            feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Extract feature names and values
            features = [f[0] for f in feature_impacts]
            values = [f[1] for f in feature_impacts]
            
            # Create color map based on positive/negative values
            colors = ['#D73027' if v < 0 else '#4575B4' for v in values]
            
            # Plot bar chart
            bars = ax.barh(features, values, color=colors)
            ax.set_xlabel('Feature Importance (SHAP Value)', fontsize=12)
            ax.set_title(f'Feature Importance for Anomaly at Index {anomaly_idx}', fontsize=14)
            
            # Add a line at x=0
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Add annotations
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width + 0.01 if width >= 0 else width - 0.01
                ha = 'left' if width >= 0 else 'right'
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha=ha, va='center', fontsize=10)
        
        else:
            # Get all anomalies
            anomaly_data = self.df.loc[self.anomaly_indices, self.feature_names].values
            
            # Scale the data
            scaled_data = self.scaler.transform(anomaly_data)
            
            # Get SHAP values for all anomalies
            shap_values = self.explainer.shap_values(scaled_data)
            
            # Calculate average absolute SHAP values
            avg_shap = np.mean(shap_values, axis=0)
            
            # Pair features with their average SHAP values
            feature_impacts = list(zip(self.feature_names, avg_shap))
            
            # Sort by average impact
            feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Extract feature names and values
            features = [f[0] for f in feature_impacts]
            values = [f[1] for f in feature_impacts]
            
            # Create color map based on positive/negative values
            colors = ['#D73027' if v < 0 else '#4575B4' for v in values]
            
            # Plot bar chart
            bars = ax.barh(features, values, color=colors)
            ax.set_xlabel('Average Feature Importance (SHAP Value)', fontsize=12)
            ax.set_title('Average Feature Importance for All Anomalies', fontsize=14)
            
            # Add a line at x=0
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Add annotations
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width + 0.01 if width >= 0 else width - 0.01
                ha = 'left' if width >= 0 else 'right'
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha=ha, va='center', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_distributions(self, figsize=(15, 10)):
        """
        Plot the distribution of each feature, highlighting anomalies.
        
        Parameters:
        -----------
        figsize : tuple, default=(15, 10)
            Figure size for the plot.
            
        Returns:
        --------
        matplotlib.figure.Figure
            A figure containing the feature distributions.
        """
        if self.anomalies is None:
            raise ValueError("No anomalies detected yet. Call detect_anomalies() first.")
        
        n_features = len(self.feature_names)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(n_rows, n_cols)
        
        for i, feature in enumerate(self.feature_names):
            ax = plt.subplot(gs[i])
            
            # Plot normal points
            sns.kdeplot(
                self.df.loc[~self.result_df['anomaly'], feature],
                ax=ax,
                color='blue',
                fill=True,
                alpha=0.3,
                label='Normal'
            )
            
            # Plot anomalies
            if not self.anomalies.empty:
                sns.kdeplot(
                    self.df.loc[self.anomaly_indices, feature],
                    ax=ax,
                    color='red',
                    fill=True,
                    alpha=0.3,
                    label='Anomaly'
                )
                
                # Add scatter plot for anomaly points
                ax.scatter(
                    self.df.loc[self.anomaly_indices, feature],
                    np.zeros_like(self.df.loc[self.anomaly_indices, feature]),
                    color='red',
                    alpha=0.6,
                    s=50,
                    marker='o'
                )
            
            ax.set_title(feature)
            if i == 0:
                ax.legend()
            
            # Add vertical lines for mean and std dev
            mean_val = self.df[feature].mean()
            std_val = self.df[feature].std()
            
            ax.axvline(x=mean_val, color='green', linestyle='--', alpha=0.7, label='Mean')
            ax.axvline(x=mean_val + 2*std_val, color='orange', linestyle='--', alpha=0.7, label='+2σ')
            ax.axvline(x=mean_val - 2*std_val, color='orange', linestyle='--', alpha=0.7, label='-2σ')
            
            # Only add x-label for bottom row
            if i >= n_features - n_cols:
                ax.set_xlabel(feature)
            
            # Remove y-labels for cleaner look
            ax.set_ylabel('')
            ax.set_yticks([])
        
        plt.suptitle('Feature Distributions with Anomalies Highlighted', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig
    
    def plot_correlation_heatmap(self, figsize=(10, 8)):
        """
        Plot a correlation heatmap of the features.
        
        Parameters:
        -----------
        figsize : tuple, default=(10, 8)
            Figure size for the plot.
            
        Returns:
        --------
        matplotlib.figure.Figure
            A figure containing the correlation heatmap.
        """
        corr = self.df[self.feature_names].corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Draw the heatmap
        sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            vmax=1,
            vmin=-1,
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .8},
            annot=True,
            fmt=".2f",
            ax=ax
        )
        
        ax.set_title('Feature Correlation Heatmap', fontsize=14)
        plt.tight_layout()
        return fig
    
    def plot_anomaly_scores(self, figsize=(12, 6)):
        """
        Plot the anomaly scores distribution and threshold.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 6)
            Figure size for the plot.
            
        Returns:
        --------
        matplotlib.figure.Figure
            A figure containing the anomaly scores plot.
        """
        if self.result_df is None:
            raise ValueError("No anomalies detected yet. Call detect_anomalies() first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot histogram of anomaly scores
        ax1.hist(
            self.result_df['anomaly_score'],
            bins=30,
            alpha=0.7,
            color='skyblue',
            edgecolor='black'
        )
        
        # Add vertical line for threshold
        threshold = self.result_df[self.result_df['anomaly']]['anomaly_score'].max()
        ax1.axvline(
            x=threshold,
            color='red',
            linestyle='--',
            label=f'Threshold: {threshold:.3f}'
        )
        
        ax1.set_title('Distribution of Anomaly Scores', fontsize=14)
        ax1.set_xlabel('Anomaly Score', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.legend()
        
        # Plot sorted anomaly scores
        sorted_scores = self.result_df['anomaly_score'].sort_values().values
        ax2.plot(
            range(len(sorted_scores)),
            sorted_scores,
            marker='.',
            markersize=3,
            linestyle='-',
            linewidth=1,
            color='blue'
        )
        
        # Highlight anomalies
        anomaly_indices = np.where(sorted_scores <= threshold)[0]
        if len(anomaly_indices) > 0:
            ax2.plot(
                anomaly_indices,
                sorted_scores[anomaly_indices],
                marker='o',
                markersize=5,
                linestyle='',
                color='red',
                alpha=0.7,
                label='Anomalies'
            )
        
        ax2.axhline(
            y=threshold,
            color='red',
            linestyle='--',
            label=f'Threshold: {threshold:.3f}'
        )
        
        ax2.set_title('Sorted Anomaly Scores', fontsize=14)
        ax2.set_xlabel('Index', fontsize=12)
        ax2.set_ylabel('Anomaly Score', fontsize=12)
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def save_model(self, filepath):
        """
        Save the trained model and related components to disk.
        
        Parameters:
        -----------
        filepath : str
            Path where to save the model.
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'min_max_scaler': self.min_max_scaler,
            'pca': self.pca,
            'feature_names': self.feature_names,
            'contamination': self.contamination,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model.
            
        Returns:
        --------
        ExplainableAnomalyDetector
            A loaded model instance.
        """
        model_data = joblib.load(filepath)
        
        detector = cls(
            contamination=model_data['contamination'],
            random_state=model_data['random_state']
        )
        
        detector.model = model_data['model']
        detector.scaler = model_data['scaler']
        detector.min_max_scaler = model_data['min_max_scaler']
        detector.pca = model_data['pca']
        detector.feature_names = model_data['feature_names']
        
        return detector

class AnomalyDetectionDashboard:
    """
    Interactive dashboard for the ExplainableAnomalyDetector.
    """
    
    def __init__(self, root=None):
        """
        Initialize the dashboard.
        
        Parameters:
        -----------
        root : tkinter.Tk, optional
            Root Tkinter window. If None, a new one will be created.
        """
        self.detector = None
        self.df = None
        self.results = None
        
        if root is None:
            self.root = tk.Tk()
        else:
            self.root = root
            
        self.root.title("Explainable Anomaly Detection Dashboard")
        self.root.geometry("1200x800")
        
        # Create a notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create frames for each tab
        self.setup_frame = ttk.Frame(self.notebook)
        self.viz_frame = ttk.Frame(self.notebook)
        self.details_frame = ttk.Frame(self.notebook)
        self.features_frame = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.setup_frame, text="Setup & Data")
        self.notebook.add(self.viz_frame, text="Anomaly Visualization")
        self.notebook.add(self.details_frame, text="Anomaly Details")
        self.notebook.add(self.features_frame, text="Feature Analysis")
        
        # Setup tab components
        self._setup_data_tab()
        
        # Visualization tab components
        self._setup_visualization_tab()
        
        # Details tab components
        self._setup_details_tab()
        
        # Feature analysis tab components
        self._setup_feature_tab()
        
        # Variable to store the canvas objects for figures
        self.canvas_objects = {}
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def _setup_data_tab(self):
        """Set up the data upload and parameter configuration tab."""
        # Create frames
        upload_frame = ttk.LabelFrame(self.setup_frame, text="Data Upload")
        upload_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        param_frame = ttk.LabelFrame(self.setup_frame, text="Model Parameters")
        param_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=10)
        
        # Data upload components
        ttk.Label(upload_frame, text="Upload a CSV or Excel file:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        upload_btn = ttk.Button(upload_frame, text="Browse...", command=self._browse_file)
        upload_btn.grid(row=0, column=1, padx=5, pady=5)
        
        self.file_path_var = tk.StringVar()
        ttk.Label(upload_frame, textvariable=self.file_path_var).grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        
        # Separator
        ttk.Separator(upload_frame, orient=tk.HORIZONTAL).grid(row=1, column=0, columnspan=3, sticky=tk.EW, pady=10)
        
        # Data preview
        ttk.Label(upload_frame, text="Data Preview:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.preview_text = scrolledtext.ScrolledText(upload_frame, height=10)
        self.preview_text.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky=tk.NSEW)
        
        # Parameter configuration
        ttk.Label(param_frame, text="Contamination (expected proportion of outliers):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.contamination_var = tk.StringVar(value="0.05")
        contamination_entry = ttk.Entry(param_frame, textvariable=self.contamination_var, width=10)
        contamination_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(param_frame, text="Random Seed:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.random_seed_var = tk.StringVar(value="42")
        random_seed_entry = ttk.Entry(param_frame, textvariable=self.random_seed_var, width=10)
        random_seed_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Run button
        self.run_btn = ttk.Button(param_frame, text="Detect Anomalies", command=self._run_detection, state=tk.DISABLED)
        self.run_btn.grid(row=2, column=0, columnspan=2, padx=5, pady=10)
        
        # Configure grid weights
        upload_frame.columnconfigure(2, weight=1)
        upload_frame.rowconfigure(3, weight=1)
    
    def _setup_visualization_tab(self):
        """Set up the anomaly visualization tab."""
        # Controls frame
        controls_frame = ttk.Frame(self.viz_frame)
        controls_frame.pack(fill=tk.X, expand=False, padx=10, pady=5)
        
        ttk.Label(controls_frame, text="Projection Method:").pack(side=tk.LEFT, padx=5, pady=5)
        
        self.projection_var = tk.StringVar(value="pca")
        projection_combo = ttk.Combobox(controls_frame, textvariable=self.projection_var, 
                                       values=["pca", "tsne"], state="readonly", width=10)
        projection_combo.pack(side=tk.LEFT, padx=5, pady=5)
        projection_combo.bind("<<ComboboxSelected>>", lambda e: self._update_visualization())
        
        refresh_btn = ttk.Button(controls_frame, text="Refresh", command=self._update_visualization)
        refresh_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Canvas frame for the figure
        self.viz_canvas_frame = ttk.Frame(self.viz_frame)
        self.viz_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    def _setup_details_tab(self):
        """Set up the anomaly details tab."""
        # Controls frame
        controls_frame = ttk.Frame(self.details_frame)
        controls_frame.pack(fill=tk.X, expand=False, padx=10, pady=5)
    
        ttk.Label(controls_frame, text="Anomaly Details:").pack(side=tk.LEFT, padx=5, pady=5)
    
        # Create a scrollable frame for anomaly details
        self.details_paned = ttk.PanedWindow(self.details_frame, orient=tk.VERTICAL)
        self.details_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
        # Top frame for anomaly scores plot
        self.score_frame = ttk.Frame(self.details_paned)
        self.details_paned.add(self.score_frame, weight=1)
    
        # Bottom frame for text explanations
        explanation_frame = ttk.LabelFrame(self.details_paned, text="Anomaly Explanations")
        self.details_paned.add(explanation_frame, weight=1)
    
        # Use a scrolled text widget for explanations
        self.explanation_text = scrolledtext.ScrolledText(explanation_frame, wrap=tk.WORD, height=10)
        self.explanation_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
        # Play Sound button
        self.play_sound_btn = ttk.Button(explanation_frame, text="🔊 Play Sound", command=self._play_explanation, state=tk.DISABLED)
        self.play_sound_btn.pack(side=tk.BOTTOM, padx=5, pady=5)
    
    def _save_audio_explanation(self):
        """Save audio explanations to files without playing them."""
        if self.detector is None:
            return
        
        try:
            self.status_var.set("Generating audio files...")
            
            # Run in separate thread to avoid freezing UI
            thread = threading.Thread(target=self._save_audio_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self.status_var.set(f"Error generating audio: {str(e)}")

    def _save_audio_thread(self):
        """Thread for saving audio explanations."""
        try:
            # Get explanations
            explanations = self.detector.explain_anomalies(max_anomalies=2)
        
            # Create a directory to save the files
            save_dir = "audio_explanations"
            os.makedirs(save_dir, exist_ok=True)
        
            # Save each explanation to file
            saved_files = []
            for i, explanation in enumerate(explanations):
                tts = gtts.gTTS(text=explanation, lang='en')
                filename = os.path.join(save_dir, f"anomaly_explanation_{i+1}.mp3")
                tts.save(filename)
                saved_files.append(filename)
        
            # Update UI with success message
            message = f"Audio files saved to {save_dir} directory"
            self.root.after(0, lambda: self.status_var.set(message))
            self.root.after(0, lambda: messagebox.showinfo("Audio Files Saved", 
                                                    f"Generated {len(saved_files)} audio files in {save_dir} directory"))
        except Exception as e:
            error_msg = f"Error generating audio files: {str(e)}"
            self.root.after(0, lambda: self.status_var.set(error_msg))
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
    
    def _setup_feature_tab(self):
        """Set up the feature analysis tab."""
        # Controls frame
        controls_frame = ttk.Frame(self.features_frame)
        controls_frame.pack(fill=tk.X, expand=False, padx=10, pady=5)

        ttk.Label(controls_frame, text="Analysis Type:").pack(side=tk.LEFT, padx=5, pady=5)

        self.feature_analysis_var = tk.StringVar(value="distributions")
        analysis_combo = ttk.Combobox(controls_frame, textvariable=self.feature_analysis_var, 
                                       values=["distributions", "correlation", "importance"], 
                                       state="readonly", width=15)
        analysis_combo.pack(side=tk.LEFT, padx=5, pady=5)
        analysis_combo.bind("<<ComboboxSelected>>", lambda e: self._update_feature_analysis())

        # Filter for specific anomaly (for importance)
        ttk.Label(controls_frame, text="Anomaly Index:").pack(side=tk.LEFT, padx=5, pady=5)

        self.anomaly_idx_var = tk.StringVar(value="")
        self.anomaly_idx_entry = ttk.Entry(controls_frame, textvariable=self.anomaly_idx_var, width=10)
        self.anomaly_idx_entry.pack(side=tk.LEFT, padx=5, pady=5)
        self.anomaly_idx_entry.bind("<Return>", lambda e: self._update_feature_analysis())

        update_btn = ttk.Button(controls_frame, text="Update", command=self._update_feature_analysis)
        update_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Play Sound button
        self.play_sound_btn = ttk.Button(controls_frame, text="🔊 Play Sound", command=self._play_explanation, state=tk.DISABLED)
        self.play_sound_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Canvas frame for the figure
        self.feature_canvas_frame = ttk.Frame(self.features_frame)
        self.feature_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    def _browse_file(self):
        """Open a file dialog to browse for data files."""
        filetypes = [
            ("All supported files", "*.csv *.xlsx *.xls"),
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx *.xls")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=filetypes
        )
        
        if filepath:
            self.file_path_var.set(filepath)
            self._load_data(filepath)
    
    def _load_data(self, filepath):
        """Load data from the selected file."""
        try:
            self.status_var.set("Loading data...")
            self.root.update_idletasks()
            
            # Check file extension
            ext = Path(filepath).suffix.lower()
            
            if ext == '.csv':
                self.df = pd.read_csv(filepath)
            elif ext in ['.xlsx', '.xls']:
                if not EXCEL_AVAILABLE:
                    raise ImportError("Excel support requires the openpyxl package. Please install it with 'pip install openpyxl'.")
                self.df = pd.read_excel(filepath)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            
            # Update preview
            preview = self.df.head(10).to_string()
            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(tk.END, preview)
            
            # Enable run button
            self.run_btn.config(state=tk.NORMAL)
            
            self.status_var.set(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        except Exception as e:
            self.status_var.set(f"Error loading data: {str(e)}")
            messagebox.showerror("Error", f"Could not load data: {str(e)}")
    
    def _run_detection(self):
        """Run the anomaly detection algorithm."""
        if self.df is None:
            self.status_var.set("No data loaded.")
            return
        
        try:
            self.status_var.set("Running anomaly detection...")
            self.root.update_idletasks()
            
            # Get parameters
            contamination = float(self.contamination_var.get())
            random_state = int(self.random_seed_var.get())
            
            # Create detector
            self.detector = ExplainableAnomalyDetector(
                contamination=contamination,
                random_state=random_state
            )
            
            # Run in a separate thread to keep UI responsive
            thread = threading.Thread(target=self._run_detection_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
    
    def _run_detection_thread(self):
        """Run detection in a separate thread."""
        try:
            # Fit and detect
            self.detector.fit(self.df)
            self.results = self.detector.detect_anomalies()
            
            # Update UI with results
            anomaly_count = len(self.detector.anomaly_indices)
            total_count = len(self.results)
            
            status = f"Detection complete: {anomaly_count} anomalies found out of {total_count} data points ({anomaly_count/total_count*100:.2f}%)."
            
            # Switch to visualization tab
            self.root.after(0, lambda: self.notebook.select(1))
            
            # Update visualizations
            self.root.after(0, self._update_visualization)
            self.root.after(0, self._update_anomaly_details)
            self.root.after(0, self._update_feature_analysis)
            
            # Update play button state
            self.root.after(0, lambda: self.play_sound_btn.config(state=tk.NORMAL))
            
            # Update status
            self.root.after(0, lambda: self.status_var.set(status))
            
        except Exception as e:
            error_msg = f"Error in detection thread: {str(e)}"
            self.root.after(0, lambda: self.status_var.set(error_msg))
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
    
    def _update_visualization(self):
        """Update the anomaly visualization."""
        if self.detector is None or self.results is None:
            return
        
        try:
            self.status_var.set("Updating visualization...")
            self.root.update_idletasks()
            
            # Clear existing canvas
            for widget in self.viz_canvas_frame.winfo_children():
                widget.destroy()
            
            # Get projection method
            projection = self.projection_var.get()
            
            # Create figure
            fig = self.detector.visualize_anomalies(projection_method=projection, figsize=(10, 8))
            
            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=self.viz_canvas_frame)
            canvas.draw()
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, self.viz_canvas_frame)
            toolbar.update()
            
            # Pack canvas
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.status_var.set("Visualization updated")
        except Exception as e:
            self.status_var.set(f"Error updating visualization: {str(e)}")
    
    def _update_anomaly_details(self):
        """Update the anomaly details tab."""
        if self.detector is None or self.results is None:
            return
        
        try:
            self.status_var.set("Updating anomaly details...")
            self.root.update_idletasks()
            
            # Clear existing canvas
            for widget in self.score_frame.winfo_children():
                widget.destroy()
            
            # Create figure for anomaly scores
            fig = self.detector.plot_anomaly_scores(figsize=(10, 5))
            
            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=self.score_frame)
            canvas.draw()
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, self.score_frame)
            toolbar.update()
            
            # Pack canvas
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Update text explanations
            explanations = self.detector.explain_anomalies(max_anomalies=5)
            
            self.explanation_text.delete(1.0, tk.END)
            for i, explanation in enumerate(explanations):
                self.explanation_text.insert(tk.END, f"Anomaly {i+1}:\n")
                self.explanation_text.insert(tk.END, explanation)
                self.explanation_text.insert(tk.END, "\n\n")
            
            self.status_var.set("Anomaly details updated")
        except Exception as e:
            self.status_var.set(f"Error updating anomaly details: {str(e)}")
    
    def _update_feature_analysis(self):
        """Update the feature analysis tab."""
        if self.detector is None or self.results is None:
            return
        
        try:
            self.status_var.set("Updating feature analysis...")
            self.root.update_idletasks()
            
            # Clear existing canvas
            for widget in self.feature_canvas_frame.winfo_children():
                widget.destroy()
            
            # Get analysis type
            analysis_type = self.feature_analysis_var.get()
            
            # Create figure based on analysis type
            if analysis_type == "distributions":
                fig = self.detector.plot_feature_distributions(figsize=(12, 10))
            elif analysis_type == "correlation":
                fig = self.detector.plot_correlation_heatmap(figsize=(10, 8))
            elif analysis_type == "importance":
                anomaly_idx_str = self.anomaly_idx_var.get().strip()
                
                if anomaly_idx_str and anomaly_idx_str.isdigit():
                    # For specific anomaly
                    anomaly_idx = int(anomaly_idx_str)
                    if anomaly_idx in self.detector.anomaly_indices:
                        fig = self.detector.plot_feature_importance(anomaly_idx=anomaly_idx, figsize=(10, 8))
                    else:
                        messagebox.showwarning("Warning", f"Index {anomaly_idx} is not an anomaly. Try one of these: {self.detector.anomaly_indices}")
                        fig = self.detector.plot_feature_importance(figsize=(10, 8))
                else:
                    # For all anomalies
                    fig = self.detector.plot_feature_importance(figsize=(10, 8))
            
            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=self.feature_canvas_frame)
            canvas.draw()
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, self.feature_canvas_frame)
            toolbar.update()
            
            # Pack canvas
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.status_var.set("Feature analysis updated")
        except Exception as e:
            self.status_var.set(f"Error updating feature analysis: {str(e)}")
    
    def _play_explanation(self):
        """Play audio explanation for anomalies."""
        if self.detector is None:
            return
        
        try:
            self.status_var.set("Playing audio explanation...")
            
            # Run in separate thread to avoid freezing UI
            thread = threading.Thread(target=self._play_explanation_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self.status_var.set(f"Error playing explanation: {str(e)}")
    
    def _play_explanation_thread(self):
        """Thread for playing audio explanations."""
        try:
            # Try to play explanations with voice
            self.detector.explain_with_voice(max_anomalies=2)
            self.root.after(0, lambda: self.status_var.set("Audio explanation complete"))
        except Exception as e:
            error_msg = f"Error in audio playback: {str(e)}"
            self.root.after(0, lambda: self.status_var.set(error_msg))
    
    def run(self):
        """Start the main event loop."""
        self.root.mainloop()

def create_synthetic_dataset(n_samples=1000, n_features=5, contamination=0.01, random_state=42):
    """
    Create a synthetic dataset with anomalies for demonstration purposes.
    
    Parameters:
    -----------
    n_samples : int, default=1000
        Number of samples to generate.
    n_features : int, default=5
        Number of features to generate.
    contamination : float, default=0.01
        The proportion of outliers in the data.
    random_state : int, default=42
        Random state for reproducibility.
        
    Returns:
    --------
    pandas.DataFrame
        The generated dataset.
    """
    np.random.seed(random_state)
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    # Generate normal data
    normal_data = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Add some correlations between features
    normal_data[:, 1] = normal_data[:, 0] * 0.5 + normal_data[:, 1] * 0.5
    if n_features > 3:
        normal_data[:, 3] = normal_data[:, 2] * 0.7 + normal_data[:, 3] * 0.3
    
    # Generate anomalies
    n_anomalies = int(contamination * n_samples)
    anomalies = np.random.normal(0, 1, size=(n_anomalies, n_features))
    
    # Make anomalies more extreme
    half = n_anomalies // 2
    anomalies[:half, 0] = np.random.uniform(5, 7, size=half)  # Extreme values in feature_1
    anomalies[half:, 2 % n_features] = np.random.uniform(-7, -5, size=n_anomalies-half)  # Extreme values in another feature
    
    # Combine normal data and anomalies
    data = np.vstack([normal_data, anomalies])
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    
    return df


def run_cli_demo():
    """Run a command-line interface demo of the anomaly detector."""
    print("="*80)
    print("Explainable Anomaly Detector - CLI Demo")
    print("="*80)
    
    # Create synthetic dataset
    print("\nGenerating synthetic dataset...")
    df = create_synthetic_dataset(n_samples=1000, n_features=5, contamination=0.01)
    print(f"Dataset created with shape: {df.shape}")
    
    # Create detector
    print("\nInitializing anomaly detector...")
    detector = ExplainableAnomalyDetector(contamination=0.01)
    
    # Fit model
    print("Fitting model...")
    detector.fit(df)
    
    # Detect anomalies
    print("Detecting anomalies...")
    results = detector.detect_anomalies()
    
    # Print summary
    anomaly_count = len(detector.anomaly_indices)
    print(f"\nDetection complete: {anomaly_count} anomalies found out of {len(results)} data points.")
    print(f"Anomaly indices: {detector.anomaly_indices}")
    
    # Generate and print explanations
    print("\nGenerating explanations...")
    explanations = detector.explain_anomalies(max_anomalies=3)
    for i, explanation in enumerate(explanations):
        print(f"\nAnomaly {i+1}:")
        print(explanation)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # PCA visualization
    print("Plotting PCA visualization...")
    fig = detector.visualize_anomalies(projection_method='pca')
    plt.savefig('anomaly_pca_visualization.png')
    plt.close(fig)
    
    # t-SNE visualization (if more than 2 features)
    if len(detector.feature_names) > 2:
        print("Plotting t-SNE visualization...")
        fig = detector.visualize_anomalies(projection_method='tsne')
        plt.savefig('anomaly_tsne_visualization.png')
        plt.close(fig)
    
    # Feature importance
    print("Plotting feature importance...")
    fig = detector.plot_feature_importance()
    plt.savefig('feature_importance.png')
    plt.close(fig)
    
    # Feature distributions
    print("Plotting feature distributions...")
    fig = detector.plot_feature_distributions()
    plt.savefig('feature_distributions.png')
    plt.close(fig)
    
    # Correlation heatmap
    print("Plotting correlation heatmap...")
    fig = detector.plot_correlation_heatmap()
    plt.savefig('correlation_heatmap.png')
    plt.close(fig)
    
    # Anomaly scores
    print("Plotting anomaly scores...")
    fig = detector.plot_anomaly_scores()
    plt.savefig('anomaly_scores.png')
    plt.close(fig)
    
    print("\nVisualizations saved to disk.")
    
    # Try voice explanations
    try:
        print("\nAttempting voice explanations...")
        detector.explain_with_voice(max_anomalies=1)
    except Exception as e:
        print(f"Voice explanation failed: {str(e)}")
    
    print("\nDemo completed.")


def main():
    """Main function to start the application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Explainable Anomaly Detection Tool")
    parser.add_argument("--cli", action="store_true", help="Run in command-line mode instead of GUI")
    args = parser.parse_args()
    
    if args.cli:
        run_cli_demo()
    else:
        # Start GUI
        root = tk.Tk()
        app = AnomalyDetectionDashboard(root)
        app.run()


if __name__ == "__main__":
    main()