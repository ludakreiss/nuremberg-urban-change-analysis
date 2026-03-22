def raster_to_df(raster, prefix):
    # Convert dataset to get columns
    df = raster.to_dataset(name="val").drop_vars("spatial_ref", errors="ignore").to_dataframe()
    
    # Each band as a column
    df = df.unstack('band')
    
    # Rename columns
    num_bands = len(df.columns)
    df.columns = [f'{prefix}_b{i}' for i in range(1, num_bands + 1)]
    
    return df.reset_index()
