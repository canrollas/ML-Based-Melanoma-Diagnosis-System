import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime
from PIL import Image
from tqdm import tqdm

class DataVisualizer:
    def __init__(self, csv_path='merged_data/metadata.csv', output_dir='data_visual_repr', images_dir='merged_data/images'):
        """
        Initialize the DataVisualizer with data source and output directory
        
        Args:
            csv_path (str): Path to the metadata CSV file
            output_dir (str): Directory to save visualizations
            images_dir (str): Directory containing the images
        """
        self.data = pd.read_csv(csv_path)
        self.output_dir = output_dir
        self.images_dir = images_dir
        os.makedirs(output_dir, exist_ok=True)
        self.analyze_data()
    
    def save_plot(self, plot_name):
        """
        Save the current plot to the output directory with timestamp
        
        Args:
            plot_name (str): Name of the plot
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{plot_name}_{timestamp}.png"
        plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
        plt.close()
    
    def analyze_data(self):
        """Perform initial analysis of the dataset"""
        print("Dataset Dimensions:", self.data.shape)
        print("\nDataset Columns:")
        for col in self.data.columns:
            print(f"- {col}")
    
    def plot_distribution(self, column, title=None, figsize=(10, 6)):
        """
        Visualize distribution of numerical variables
        
        Args:
            column (str): Column name to analyze
            title (str): Custom title for the plot
            figsize (tuple): Figure dimensions
        """
        if column not in self.data.columns:
            print(f"Warning: Column '{column}' not found in dataset!")
            return
            
        plt.figure(figsize=figsize)
        sns.histplot(data=self.data, x=column, kde=True)
        plt.title(title or f'Distribution of {column}')
        self.save_plot(f"distribution_{column}")
        
        # Print basic statistics
        print(f"\nBasic statistics for {column}:")
        print(self.data[column].describe())
    
    def plot_categorical_distribution(self, column, title=None, figsize=(10, 6)):
        """
        Visualize distribution of categorical variables
        
        Args:
            column (str): Column name to analyze
            title (str): Custom title for the plot
            figsize (tuple): Figure dimensions
        """
        if column not in self.data.columns:
            print(f"Warning: Column '{column}' not found in dataset!")
            return
            
        plt.figure(figsize=figsize)
        value_counts = self.data[column].value_counts()
        sns.barplot(x=value_counts.index, y=value_counts.values)
        plt.title(title or f'Distribution of {column}')
        plt.xticks(rotation=45)
        self.save_plot(f"categorical_{column}")
        
        # Print value distribution
        print(f"\nValue distribution for {column}:")
        print(value_counts)
        print(f"Percentage distribution:\n{(value_counts / len(self.data) * 100).round(2)}%")
    
    def plot_correlation_matrix(self, figsize=(12, 8)):
        """Visualize correlation between numerical variables"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        corr_matrix = self.data[numeric_cols].corr()
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix Between Variables')
        plt.show()
    
    def plot_boxplots(self, numeric_columns=None, figsize=(12, 6)):
        """Create box plots for numerical variables"""
        if numeric_columns is None:
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            
        plt.figure(figsize=figsize)
        self.data[numeric_columns].boxplot()
        plt.xticks(rotation=45)
        plt.title('Box Plot for Numerical Variables')
        plt.show()
    
    def analyze_missing_values(self, threshold=50):
        """
        Analyze missing values and provide trimming suggestions
        
        Args:
            threshold (int): Threshold percentage for suggesting column removal
        """
        missing_values = self.data.isnull().sum()
        missing_percentages = (missing_values / len(self.data)) * 100
        
        # Missing value analysis
        missing_df = pd.DataFrame({
            'Missing Count': missing_values,
            'Missing Percentage': missing_percentages
        }).sort_values('Missing Percentage', ascending=False)
        
        print("\nMissing value statistics:")
        print(missing_df)
        
        # Trimming suggestions
        high_missing = missing_df[missing_df['Missing Percentage'] > threshold]
        if not high_missing.empty:
            print(f"\nColumns with missing values above {threshold}%:")
            for col in high_missing.index:
                print(f"- {col}: {high_missing.loc[col, 'Missing Percentage']:.2f}% missing")
            print("\nConsider removing these columns from the analysis.")
        
        # Visualization
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(missing_percentages)), missing_percentages)
        plt.xticks(range(len(missing_percentages)), missing_df.index, rotation=90)
        plt.title('Missing Value Percentages by Column')
        plt.ylabel('Missing Value Percentage (%)')
        plt.tight_layout()
        self.save_plot("missing_values")
    
    def analyze_key_features(self):
        """Analyze key features of the dataset"""
        # Diagnosis distribution
        print("\nMelanoma Diagnosis Distribution:")
        self.plot_categorical_distribution('benign_malignant')
        
        # Age distribution
        print("\nAge Distribution:")
        self.plot_distribution('age_approx')
        
        # Gender distribution
        print("\nGender Distribution:")
        self.plot_categorical_distribution('sex')
        
        # Anatomical site distribution
        print("\nAnatomical Site Distribution:")
        self.plot_categorical_distribution('anatom_site_general')

    def analyze_image_dimensions(self):
        """Analyze and visualize the distribution of image dimensions"""
        print("\nAnalyzing image dimensions...")
        
        # Collect image dimensions
        dimensions = []
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            image_path = os.path.join(self.images_dir, row['isic_id'] + '.jpg')
            if os.path.exists(image_path):
                try:
                    with Image.open(image_path) as img:
                        dimensions.append(img.size)
                except Exception as e:
                    print(f"Error: Could not read file {image_path} - {str(e)}")
        
        # Split dimensions into separate lists
        widths, heights = zip(*dimensions)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(widths, heights, alpha=0.5)
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        plt.title('Distribution of Image Dimensions')
        
        # Show average dimensions
        avg_width = np.mean(widths)
        avg_height = np.mean(heights)
        plt.axvline(x=avg_width, color='r', linestyle='--', alpha=0.5, 
                   label=f'Avg Width: {avg_width:.0f}px')
        plt.axhline(y=avg_height, color='r', linestyle='--', alpha=0.5,
                   label=f'Avg Height: {avg_height:.0f}px')
        plt.legend()
        
        # Print statistics
        print(f"\nImage Dimension Statistics:")
        print(f"Average Width: {avg_width:.0f} pixels")
        print(f"Average Height: {avg_height:.0f} pixels")
        print(f"Minimum Size: {min(widths)}x{min(heights)} pixels")
        print(f"Maximum Size: {max(widths)}x{max(heights)} pixels")
        
        # Histogram
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(widths, bins=50)
        plt.xlabel('Width (pixels)')
        plt.ylabel('Number of Images')
        plt.title('Width Distribution')
        
        plt.subplot(1, 2, 2)
        plt.hist(heights, bins=50)
        plt.xlabel('Height (pixels)')
        plt.ylabel('Number of Images')
        plt.title('Height Distribution')
        
        plt.tight_layout()
        self.save_plot("image_dimensions")
