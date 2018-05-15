import numpy as np
from sklearn.neighbors import KDTree, BallTree

def weighted_distance(x,y,**kwargs):
    """
    Auxiliary function to define a chi-square-like metric
    """
    return np.sum((x-y)**2/kwargs['metric_params']['weights']**2)

def spatial_closest(ra_data,dec_data,
                    ra_true,dec_true,true_id):
    """
    Function to return the closest neighbor's ID in the true catalog
    using spatial matching. Units are not important but degrees are
    preferred so the output distance is returned in arcseconds.
    
    ***Caveats***: This method uses small angle approximation sin(theta)
    ~ theta for the declination axis. This should be fine to find the closest
    neighbor. This method does not use any weighting. All objects have a match.
    
    Args:
    -----
    
    ra_data: Right ascension of the measured objects (degrees preferred).
    dec_data: Declination of the measured objects (degrees preferred).
    ra_true: Right ascension of the true catalog (degrees preferred).
    dec_true: Declination of the true catalog (degrees preferred).
    true_id: List of IDs of objects in the true catalog
    Returns:
    --------
    
    dist: Distance to the closest neighbor in the true catalog. If inputs are
    in degrees, the returned distance is in arcseconds.
    true_id: ID in the true catalog for the closest match.
    matched: If a match was found matched=True, if not False. With this method
    all objects are matched so matched is True for all objects.
    """
    X = np.zeros((len(ra_true),2))
    X[:,0] = ra_true
    X[:,1] = dec_true
    tree = KDTree(X)
    Y = np.zeros((len(ra_data),2))
    Y[:,0] = ra_data
    Y[:,1] = dec_data
    dist, ind = tree.query(Y)
    return dist.flatten()*3600., true_id[ind.flatten()], np.ones(len(ind),dtype=bool)

def spatial_closest_mag(ra_data,dec_data,mag_data,
                              ra_true,dec_true,mag_true,true_id,
                              rmax=3,max_deltamag=1.):
    """
    Function to return the closest match in magnitude within a user-defined radius within certain
    magnitude difference.
    
    ***Caveats***: This method uses small angle approximation sin(theta)
    ~ theta for the declination axis. This should be fine to find the closest
    neighbor. This method does not use any weighting.
    
    Args:
    -----
    
    ra_data: Right ascension of the measured objects (degrees).
    dec_data: Declination of the measured objects (degrees).
    mag_data: Measured magnitude of the objects (assumed 1 band only).
    ra_true: Right ascension of the true catalog (degrees).
    dec_true: Declination of the true catalog (degrees).
    mag_true: True magnitude of the true catalog (assumed 1 band only).
    true_id: Array of IDs in the true catalog.
    rmax: Maximum distance in number of pixels to perform the query.
    max_deltamag: Maximum magnitude difference for the match to be good.
    
    Returns:
    --------
    
    dist: Distance to the closest neighbor in the true catalog. If inputs are
    in degrees, the returned distance is in arcseconds.
    true_id: ID in the true catalog for the closest match.
    matched: True if matched, False if not matched.
    """
    X = np.zeros((len(ra_true),2))
    X[:,0] = ra_true
    X[:,1] = dec_true
    tree = KDTree(X)
    Y = np.zeros((len(ra_data),2))
    Y[:,0] = ra_data
    Y[:,1] = dec_data
    ind, dist = tree.query_radius(Y,r=rmax*0.2/3600,return_distance=True)
    matched = np.zeros(len(ind),dtype=bool)
    ids = np.zeros(len(ind),dtype=long)
    dist_out = np.zeros(len(ind))
    print(ind)
    for i, ilist in enumerate(ind):
        if len(ilist)>0:
            dmag = np.fabs(mag_true[ilist]-mag_data[i])
            good_ind = np.argmin(dmag)
            ids[i]=true_id[ilist[good_ind]]
            dist_out[i]=dist[i][good_ind]*3600.
            if np.min(dmag)<max_deltamag:
                matched[i]=True
            else:
                matched[i]=False
        else:
            ids[i]=-99
            matched[i]=False
            dist_out[i]=-99.
    return dist_out, ids,matched

def weighted_match(ra_data,dec_data,flux_data,ra_err,dec_err,flux_err,
                   ra_true,dec_true,flux_true,true_id):
    """
    Function to return the closest match using positions, fluxes and errors
    using a BallTree with a chi-square-like metric ignoring correlations between position
    and flux.
    
    ***Caveats***: This method uses small angle approximation sin(theta)
    ~ theta for the declination axis. This should be fine to find the closest
    neighbor. This method ignores correlations between centroid position and flux.
    
    Args:
    -----
    
    ra_data: Right ascension of the measured objects.
    dec_data: Declination of the measured objects.
    flux_data: Array with the measured fluxes in the different bands 
    considered for the matching. Shape (N_entries,num_bands).
    ra_err: Uncertainty in the right ascension (use same units as ra_data).
    dec_err: Uncertainty in the declination (use same units as dec_data).
    flux_err: Uncertainty in the fluxes (same shape as flux_data).
    ra_true: Right ascension of the true catalog (same units as ra_data).
    dec_true: Declination of the true catalog (same units as dec_data).
    flux_true: Fluxes of the true catalog (same units as as flux_data).
    Shape (N_true_entries,num_bands)
    true_id: Array of IDs in the true catalog.
    
    Returns:
    --------
    
    dist: Distance in the N-dimensional space to the closest neighbor in the true catalog
    with N=num_bands+2.
    true_id: ID in the true catalog for the closest match.
    matched: True if matched, False if not matched.
    """ 
    if len(flux_true.shape)==1:
        nbands=1
    else:
        nbands = flux_true.shape[1]
    X = np.zeros((len(ra_true),2+nbands))
    Y = np.zeros((len(ra_data),2+nbands))
    Y_err = np.zeros_like(Y)
    X[:,0] = ra_true
    X[:,1] = dec_true
    Y[:,0] = ra_data
    Y[:,1] = dec_data
    Y_err[:,0] = ra_err
    Y_err[:,1] = dec_err
    if nbands==1:
        X[:,2] = flux_true
        Y[:,2] = flux_data
        Y_err[:,2] = flux_err
    else:
        X[:,2:] = flux_true
        Y[:,2:] = flux_data
        Y_err[:,2:] = flux_err
    tree = BallTree(X,metric='pyfunc',func=weighted_distance,metric_params={'weights': Y_err})
    Y = np.zeros((len(ra_data),2))
    Y[:,0] = ra_data
    Y[:,1] = dec_data
    dist, ind = tree.query(Y)
    return dist.flatten(), true_id[ind.flatten()], np.ones(len(ind))
