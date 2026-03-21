def compute_ndvi(nir, red):
    return (nir - red) / (nir + red)

def compute_ndbi(swir, nir):
    return (swir - nir) / (swir + nir)
