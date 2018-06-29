import numpy as np
from sklearn.neighbors import KDTree
from scipy.stats import chi2

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
    tree = KDTree(X, metric='euclidean')
    Y = np.zeros((len(ra_data),2))
    Y[:,0] = ra_data
    Y[:,1] = dec_data
    dist, ind = tree.query(Y)
    return dist.flatten()*3600., true_id[ind.flatten()], np.ones(len(ind),dtype=bool)

def spatial_closest_mag_1band(ra_data,dec_data,mag_data,
                              ra_true,dec_true,mag_true,true_id,
                              npix=4,max_deltamag=1.):
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
    mag_data: Measured magnitude of the objects.
    ra_true: Right ascension of the true catalog (degrees).
    dec_true: Declination of the true catalog (degrees).
    mag_true: True magnitude of the true catalog.
    true_id: Array of IDs in the true catalog.
    npix: Maximum distance in number of pixels to perform the query.
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
    tree = KDTree(X,metric='euclidean')
    Y = np.zeros((len(ra_data),2))
    Y[:,0] = ra_data
    Y[:,1] = dec_data
    ind,dist= tree.query_radius(Y,r=npix*0.2/3600,return_distance=True)
    matched = np.zeros(len(ind),dtype=bool)
    ids = np.zeros(len(ind),dtype=long)
    dist_out = np.zeros(len(ind))
    for i, ilist in enumerate(ind):
        if len(ilist)>0:
            dmag = np.fabs(mag_true[ilist]-mag_data[i])
            good_ind = np.argmin(dmag)
            ids[i]=true_id[ilist[good_ind]]
            dist_out[i]=dist[i][good_ind]
            if np.min(dmag)<max_deltamag:
                matched[i]=True
            else:
                matched[i]=False
        else:
            ids[i]=-99
            matched[i]=False
            dist_out[i]=-99.
    return dist_out*3600., ids,matched


def weighted_match(ra_data,dec_data,flux_data,ra_err,dec_err,flux_err,
                   ra_true,dec_true,flux_true,true_id,npix=4,min_prob=0.01,use_dist='angular'):
    """
    Function to return the closest match using positions, fluxes and errors
    using a KDTree. First it queries in a circle of radius npix, then it computes
    the chi-square for all objects in that circle and returns the one with the minimum
    total_chi-square.
    
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
    npix: Number of pixels in which to do the query
    min_prob: Minimum value of the returned chi-square probability for
    a source to claim that it has been matched.
    use_dist: If angular returns angular distance, if chi2 it returns the distance in
    terms of the chi square.
    Returns:
    --------
   
    dist_ang: Angular distance to match in arcseconds.
    dist_chi: Distance in the N-dimensional space weighted by the error in each dimension
    to the closest neighbor in the true catalog with N=num_bands+2.
    true_id: ID in the true catalog for the closest match.
    matched: True if matched, False if not matched.
    nmatches: Number of sources that would be a good match in terms of chi-square probability.
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
    tree = KDTree(X[:,:2],metric='euclidean')
    ind, dist = tree.query_radius(Y[:,:2],r=npix*0.2/3600.,return_distance=True)
    ids = np.zeros(len(ind),dtype=true_id.dtype)
    dist_chi = np.zeros(len(ind))
    dist_ang = np.zeros(len(ind))
    matched =np.zeros(len(ind),dtype=bool)
    nmatches = np.zeros(len(ind),dtype=int)
    for i, ilist in enumerate(ind):
        if len(ilist)>0:
            total_chisq = np.sum((X[ilist,:]-Y[i,:])**2/Y_err[i,:]**2,axis=1)
            good_ind = np.argmin(total_chisq)
            ids[i] = true_id[ilist[good_ind]]
            nmatches[i] = np.count_nonzero(total_chisq>min_prob)
            dist_ang[i] = dist[i][good_ind]*3600.
            dist_chi[i] = np.min(total_chisq)
            if 1-chi2.cdf(np.min(total_chisq),X.shape[1]-1)>min_prob:
                matched[i] = True
            else:
                matched[i] = False
        else:
            ids[i]=-99
            matched[i]=False
            dist_out[i]=-99.
    return dist_ang, dist_chi, ids, matched, nmatches
