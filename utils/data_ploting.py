
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def plot_lidar_heatmap(data, log_scale='all', cmap='viridis', figsize=None):
    """
    Plot LIDAR data as heatmap(s) with different scaling options.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        LIDAR data with height as index and wavelengths as columns
    log_scale : str, default='all'
        Scaling option: 'raw' or 'None' for linear only,
                       'log10' for log10 only,
                       'log2' for log2 only,
                       'all' for all three (linear, log10, log2)
    cmap : str, default='viridis'
        Colormap to use
    figsize : tuple, optional
        Figure size. Auto-determined based on log_scale if None.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    
    # Determine number of subplots
    scales = []
    if log_scale.lower() == 'all':
        scales = ['linear', 'log10', 'log2']
        default_figsize = (18, 8)
    elif log_scale.lower() in ['raw', 'none']:
        scales = ['linear']
        default_figsize = (8, 6)
    elif log_scale.lower() == 'log10':
        scales = ['log10']
        default_figsize = (8, 6)
    elif log_scale.lower() == 'log2':
        scales = ['log2']
        default_figsize = (8, 6)
    else:
        raise ValueError("log_scale must be 'all', 'raw', 'log10', or 'log2'")
    
    if figsize is None:
        figsize = default_figsize
    
    fig, axes = plt.subplots(1, len(scales), figsize=figsize)
    
    # Handle single subplot case
    if len(scales) == 1:
        axes = [axes]
    
    for idx, scale in enumerate(scales):
        ax = axes[idx]
        
        # Prepare data based on scale
        if scale == 'linear':
            plot_data = data.values
            title = 'Intensity Heatmap (Linear Scale)'
            label = 'Intensity'
        elif scale == 'log10':
            plot_data = np.log10(data.values + 1e-10)
            title = 'Intensity Heatmap (Log10 Scale)'
            label = 'log10(Intensity)'
        elif scale == 'log2':
            plot_data = np.log2(data.values + 1e-10)
            title = 'Intensity Heatmap (Log2 Scale)'
            label = 'log2(Intensity)'
        
        # Create heatmap
        im = ax.imshow(plot_data, aspect='auto', cmap=cmap,
                      extent=[data.columns.min(), data.columns.max(), 
                             data.index.min(), data.index.max()])
        ax.set_xlabel('Wavelength (nm)', fontsize=11)
        ax.set_ylabel('Height (m)', fontsize=11)
        ax.set_title(title, fontsize=12)
        plt.colorbar(im, ax=ax, label=label)
    
    plt.tight_layout()
    return fig






def plot_height_binned_lidar(data, bin_size=200, log_scale='log2', figsize=(14, 8)):
    """
    Plot LIDAR data grouped by height intervals as line charts with different scaling options.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        LIDAR data with height as index and wavelengths as columns
    bin_size : int, default=200
        Height bin size in meters
    log_scale : str, default='log2'
        Scaling option: 'raw' or 'none' for linear only,
                       'log10' for log10 only,
                       'log2' for log2 only,
                       'all' for all three (linear, log10, log2)
    figsize : tuple, default=(14, 8)
        Figure size
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    grouped_data : pandas.DataFrame
        The grouped data (averaged by height bins)
    """
    
    # Create height bins
    height_bins = np.arange(data.index.min(), data.index.max() + bin_size, bin_size)
    height_labels = [f'{int(height)}-{int(height + bin_size)}m' for height in height_bins[:-1]]
    
    # Create a copy to avoid modifying the original
    data_copy = data.copy()
    data_copy['height_bin'] = pd.cut(data_copy.index, bins=height_bins, 
                                      labels=height_labels, include_lowest=True)
    
    # Group by height bin and average across columns (wavelengths)
    grouped_data = data_copy.groupby('height_bin', observed=True).mean()
    
    # Get wavelengths for x-axis
    wavelengths_array = grouped_data.columns.astype(float)
    
    # Determine number of subplots and scales
    scales = []
    if log_scale.lower() == 'all':
        scales = ['linear', 'log10', 'log2']
        figsize = (18, 5)
    elif log_scale.lower() in ['raw', 'none']:
        scales = ['linear']
    elif log_scale.lower() == 'log10':
        scales = ['log10']
    elif log_scale.lower() == 'log2':
        scales = ['log2']
    else:
        raise ValueError("log_scale must be 'all', 'raw', 'log10', or 'log2'")
    
    fig, axes = plt.subplots(1, len(scales), figsize=figsize)
    
    # Handle single subplot case
    if len(scales) == 1:
        axes = [axes]
    
    for plot_idx, scale in enumerate(scales):
        ax = axes[plot_idx]
        
        # Prepare data based on scale
        if scale == 'linear':
            plot_data = grouped_data.values
            ylabel = 'Average Intensity'
            title = f'Average Intensity vs Wavelength (Linear Scale, {bin_size}m bins)'
        elif scale == 'log10':
            plot_data = np.log10(grouped_data.values + 1e-10)
            ylabel = 'log10(Average Intensity)'
            title = f'Average Intensity vs Wavelength (Log10 Scale, {bin_size}m bins)'
        elif scale == 'log2':
            plot_data = np.log2(grouped_data.values + 1e-10)
            ylabel = 'log2(Average Intensity)'
            title = f'Average Intensity vs Wavelength (Log2 Scale, {bin_size}m bins)'
        
        # Plot each height bin as a line
        for idx, height_range in enumerate(grouped_data.index):
            ax.plot(wavelengths_array, plot_data[idx], marker='o', linewidth=2, 
                   label=height_range, alpha=0.7, markersize=3)
        
        ax.set_xlabel('Wavelength (nm)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    print(f"Height bins: {bin_size}m")
    print(f"Number of bins: {len(grouped_data)}")
    print(f"Grouped data shape: {plot_data.shape}")
    
    return fig, grouped_data



def plot_groundtruth_spectra(datasets_dict, wavelength_range=None, log_scale='log2', 
                             figsize=(14, 8), markers=None):
    """
    Plot ground truth spectral data for multiple species with different scaling options.
    
    Parameters:
    -----------
    datasets_dict : dict
        Dictionary with species names as keys and DataFrames as values
        Each DataFrame should have wavelengths in column 0 and intensity in column 1
    wavelength_range : tuple, optional
        Tuple of (min_wavelength, max_wavelength) to filter data
        If None, uses full range
    log_scale : str, default='log2'
        Scaling option: 'raw' or 'none' for linear only,
                       'log10' for log10 only,
                       'log2' for log2 only,
                       'all' for all three (linear, log10, log2)
    figsize : tuple, default=(14, 8)
        Figure size
    markers : dict, optional
        Dictionary mapping species names to marker styles
        If None, uses default markers
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    
    # Default markers
    default_markers = {
        'Dactylis': 'o',
        'Betula': 's',
        'Fagus': '^',
        'Quercus': 'd'
    }
    
    if markers is None:
        markers = default_markers
    
    # Determine scales
    scales = []
    if log_scale.lower() == 'all':
        scales = ['linear', 'log10', 'log2']
        figsize = (18, 5)
    elif log_scale.lower() in ['raw', 'none']:
        scales = ['linear']
    elif log_scale.lower() == 'log10':
        scales = ['log10']
    elif log_scale.lower() == 'log2':
        scales = ['log2']
    else:
        raise ValueError("log_scale must be 'all', 'raw', 'log10', or 'log2'")
    
    fig, axes = plt.subplots(1, len(scales), figsize=figsize)
    
    # Handle single subplot case
    if len(scales) == 1:
        axes = [axes]
    
    for plot_idx, scale in enumerate(scales):
        ax = axes[plot_idx]
        
        # Plot each species
        for species_name, data in datasets_dict.items():
            wavelengths = data[0].values
            intensity = data[1].values
            
            # Apply wavelength filter if specified
            if wavelength_range is not None:
                wl_min, wl_max = wavelength_range
                mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
                wavelengths = wavelengths[mask]
                intensity = intensity[mask]
            
            # Apply scaling
            if scale == 'linear':
                plot_intensity = intensity
            elif scale == 'log10':
                plot_intensity = np.log10(intensity + 1e-10)
            elif scale == 'log2':
                plot_intensity = np.log2(intensity + 1e-10)
            
            # Get marker for this species
            marker = markers.get(species_name, 'o')
            
            # Plot
            ax.plot(wavelengths, plot_intensity, marker=marker, linewidth=2.5, 
                   label=species_name, alpha=0.8, markersize=4)
        
        # Set labels and title
        ax.set_xlabel('Wavelength (nm)', fontsize=11)
        if scale == 'linear':
            ax.set_ylabel('Intensity', fontsize=11)
            title = 'Ground Truth Spectra (Linear Scale)'
        elif scale == 'log10':
            ax.set_ylabel('log10(Intensity)', fontsize=11)
            title = 'Ground Truth Spectra (Log10 Scale)'
        elif scale == 'log2':
            ax.set_ylabel('log2(Intensity)', fontsize=11)
            title = 'Ground Truth Spectra (Log2 Scale)'
        
        if wavelength_range is not None:
            wl_min, wl_max = wavelength_range
            title += f' ({wl_min}-{wl_max} nm)'
            ax.set_xlim(wl_min - 5, wl_max + 5)
        
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print summary
    print(f"Ground Truth Species Plotted: {', '.join(datasets_dict.keys())}")
    if wavelength_range is not None:
        print(f"Wavelength Range: {wavelength_range[0]} - {wavelength_range[1]} nm")
    else:
        print(f"Wavelength Range: Full spectrum")
    print(f"Log Scale: {log_scale}")
    
    return fig

