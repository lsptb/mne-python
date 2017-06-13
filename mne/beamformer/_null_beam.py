"""Compute nulling beamformer."""

# Authors: Mark Wronkiewicz <wronk.mark@gmail.com>
#          Nick Foti
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg

from ..io.pick import (
    pick_types, pick_channels_forward, pick_channels_cov, pick_info)
from ..forward import _subject_from_forward
from ..minimum_norm.inverse import _get_vertno, combine_xyz, _check_reference
from ..cov import compute_whitener, compute_covariance
from ..source_estimate import _make_stc, SourceEstimate
from ..source_space import label_src_vertno_sel
from ..utils import logger, verbose, warn, estimate_rank
from .. import Epochs, Label
from ..beamformer import _reg_pinv, _setup_picks, _prepare_beamformer_input
from ..externals import six


@verbose
def nulling_beamformer(evoked, forward, noise_cov, data_cov, locations,
                       reg=0.05, L_val=4, pick_ori=None, picks=None, rank=None,
                       verbose=None):
    """Nulling beamformer for evoked data.

    .. note:: This implementation has not been heavily tested so please
              report any issue or suggestions.

    Parameters
    ----------
    evoked : Evoked
        Evoked data to invert
    forward : dict
        Forward operator
    noise_cov : Covariance
        The noise covariance
    data_cov : Covariance
        The data covariance
    locations : list of len 2 of array-like of int | Label
        Vertex indices (corresponding to src space) or Label indicating which
        vertices/region to compute activity for
    reg : float
        The regularization for the whitened data covariance.
    L_val: int
        Number of singular values to use in ROI-based beamformer. Not used if
        activity is only being estimated at individual dipoles.
    pick_ori : None | 'normal' | 'max-power'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept. If 'max-power', the source
        orientation that maximizes output source power is chosen.
    picks : array-like of int
        Channel indices to use for beamforming (if None all channels
        are used except bad channels).
    rank : None | int | dict
        Specified rank of the noise covariance matrix. If None, the rank is
        detected automatically. If int, the rank is specified for the MEG
        channels. A dictionary with entries 'eeg' and/or 'meg' can be used
        to specify the rank for each modality.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate
        Source time courses

    Notes
    -----
    The original reference is:
    Hui et al. Identifying true cortical interactions in MEG using the nulling
    beamformer. NeuroImage (2010) vol. 49 pp. 3161--3174
    """
    _check_reference(evoked)

    info = evoked.info
    data = evoked.data
    tmin = evoked.times[0]

    picks = _setup_picks(picks, info, forward, noise_cov)

    data = data[picks]

    if type(locations) == list:
        stc = _apply_pt_nulling_lcmv(
            data=data, info=info, tmin=tmin, forward=forward,
            noise_cov=noise_cov, data_cov=data_cov, reg=reg, vertno=locations,
            picks=picks, rank=rank, pick_ori=pick_ori)

    elif type(locations) == Label:
        stc = _apply_roi_nulling_lcmv(
            data=data, info=info, tmin=tmin, forward=forward,
            noise_cov=noise_cov, data_cov=data_cov, reg=reg, label=locations,
            L_val=L_val, picks=picks, rank=rank, pick_ori=pick_ori)
    else:
        raise RuntimeError('`locations` must be list or mne.Label, got %s' %
                           type(locations))

    return six.advance_iterator(stc)


def _get_src_vert_inds(vertno, src):
    """Helper to return inds of vertices in downsampled src space from full
    src space inds"""

    lh = [np.where(src[0]['vertno'] == vi)[0][0] for vi in vertno[0]]
    rh = [np.where(src[1]['vertno'] == vi)[0][0] + src[0]['nuse']
          for vi in vertno[1]]

    return lh + rh


def _get_pt_null_mat(n_src, vertno, src):
    """Helper to get `f` matrix for eq. 7 of Hui et al., 2010"""
    vert_inds = _get_fwd_vert_inds(vertno, src)

    f = np.zeros((n_src, len(vert_inds)))
    for vi, vert in enumerate(vert_inds):
        f[vert, vi] = 1.

    return f


def _get_roi_null_mat(rd, s_vals, V_mat):
    """Helper to get f constraint for ROI beamformer"""
    s_mat_inv = np.diag(1. / s_vals)

    return np.dot(np.dot(rd, V_mat), s_mat_inv).T


def _get_L_svd(mat, L_val):
    """Helper to get approximation of mat from largest L sing vals/vecs"""
    assert(L_val <= mat.shape[0], 'L_val too large')

    # Compute SVD
    U_mat, s_vals, Vh_mat = np.linalg.svd(mat)

    # Return U, s, V corresponding to largest L singular vals
    return U_mat[:, :L_val], s_vals[:L_val], Vh_mat[:L_val, :].T


def _apply_pt_nulling_lcmv(data, info, tmin, forward, noise_cov, data_cov, reg,
                           vertno, picks=None, pick_ori=None, rank=None,
                           verbose=None):

    is_free_ori, ch_names, proj, vertno, G = \
        _prepare_beamformer_input(info, forward, None, picks, pick_ori)
    if is_free_ori or pick_ori is not None:
        raise RuntimeError('Free orientation not supported in nulling beamformer')

    # Handle whitening + data covariance
    whitener, _ = compute_whitener(noise_cov, info, picks, rank=rank)

    # whiten the leadfield
    G = np.dot(whitener, G)

    # Apply SSPs + whitener to data covariance
    data_cov = pick_channels_cov(data_cov, include=ch_names)
    Cm = data_cov['data']
    if info['projs']:
        Cm = np.dot(proj, np.dot(Cm, proj.T))
    Cm = np.dot(whitener, np.dot(Cm, whitener.T))

    # Tikhonov regularization using reg parameter to control for
    # trade-off between spatial resolution and noise sensitivity. Directly
    # solving for Cm (as in np.solve(Cm, G)) is not stable; use _reg_p_inv
    Cm_inv = _reg_pinv(Cm.copy(), reg)
    del Cm

    n_sources = G.shape[1]

    # Compute spatial weight vectors for Eq. 7
    f = _get_pt_null_mat(n_sources, vertno, forward['src'])

    # Compute Cm^-1 * G in equation 7 of Hui et al. 2010
    D = np.dot(Cm_inv, G)
    # Compute inverse bracket term in equation 7 of Hui et al. 2010
    # Regularization necessary, np.dot(G.T, D) has bad condition number
    #TODO: Need to find automated way to choose/tune this regularizer but may
    # be fine to hard code it if E is represents a normalized matrix
    E = linalg.pinv(np.dot(G.T, D), .001)  # 1e-2 or 1e-3 works
    W = np.dot(np.dot(D, E), f).T

    # Preparing noise normalization
    noise_norm = np.sum(W ** 2, axis=1)
    noise_norm = np.sqrt(noise_norm)

    # Applying noise normalization
    noise_norm_inv = 1. / noise_norm
    noise_norm_inv[noise_norm == 0.] = 0.
    W *= noise_norm_inv[:, None]

    if isinstance(data, np.ndarray) and data.ndim == 2:
        data = [data]
        return_single = True
    else:
        return_single = False

    subject = _subject_from_forward(forward)
    for i, M in enumerate(data):
        if len(M) != len(picks):
            raise ValueError('data and picks must have the same length')

        if not return_single:
            logger.info("Processing epoch : %d" % (i + 1))

        # SSP and whitening
        if info['projs']:
            M = np.dot(proj, M)
        M = np.dot(whitener, M)

        # Project to source space using beamformer weights
        # Linear inverse: do computation here or delayed
        if M.shape[0] < W.shape[0]:
            sol = (W, M)
        else:
            sol = np.dot(W, M)

        tstep = 1.0 / info['sfreq']
        yield _make_stc(sol, vertices=vertno, tmin=tmin, tstep=tstep,
                        subject=subject)

    logger.info('[done]')


def _apply_roi_nulling_lcmv(data, info, tmin, forward, noise_cov, data_cov,
                            reg, label, L_val, picks=None, pick_ori=None,
                            rank=None, verbose=None):
    #TODO: Could extend to handle list of labels instead of just one

    is_free_ori, ch_names, proj, vertno, G = \
        _prepare_beamformer_input(info, forward, label, picks, pick_ori)
    if is_free_ori or pick_ori is not None:
        raise RuntimeError('Free orientation not supported in nulling beamformer')

    # Handle whitening + data covariance
    whitener, _ = compute_whitener(noise_cov, info, picks, rank=rank)

    # whiten the leadfield
    G = np.dot(whitener, G)

    # Apply SSPs + whitener to data covariance
    data_cov = pick_channels_cov(data_cov, include=ch_names)
    Cm = data_cov['data']
    if info['projs']:
        Cm = np.dot(proj, np.dot(Cm, proj.T))
    Cm = np.dot(whitener, np.dot(Cm, whitener.T))

    # Tikhonov regularization using reg parameter to control for
    # trade-off between spatial resolution and noise sensitivity. Directly
    # solving for Cm (as in np.solve(Cm, G)) is not stable; use _reg_p_inv
    Cm_inv = _reg_pinv(Cm.copy(), reg)
    del Cm

    n_sources = G.shape[1]

    ###########################################
    vert_inds = _get_src_vert_inds(vertno, forward['src'])
    G_u, G_s, G_v = _get_L_svd(G, L_val)
    rd = np.ones((1, n_sources))

    # Compute eigenvector constraint mat (RHS of equation) in Eq 12
    f = _get_roi_null_mat(rd, G_s, G_v)

    # Compute Cm^-1 * G_svd in equation 7 of Hui et al. 2010
    D = np.dot(Cm_inv, G_u)

    # Compute inverse bracket term in equation 7 of Hui et al. 2010
    # Regularization necessary, np.dot(G.T, D) has bad condition number
    #TODO: Need to find automated way to choose/tune this regularizer but may
    # be fine to hard code it if E is represents a normalized matrix
    E = linalg.pinv(np.dot(G_u.T, D), .001)  # 1e-2 or 1e-3 works

    W = np.dot(np.dot(D, E), f).T
    ###########################################

    # Preparing noise normalization
    noise_norm = np.sum(W ** 2, axis=1)
    noise_norm = np.sqrt(noise_norm)

    # Applying noise normalization
    noise_norm_inv = 1. / noise_norm
    noise_norm_inv[noise_norm == 0.] = 0.
    W *= noise_norm_inv[:, None]

    if isinstance(data, np.ndarray) and data.ndim == 2:
        data = [data]
        return_single = True
    else:
        return_single = False

    subject = _subject_from_forward(forward)
    for i, M in enumerate(data):
        if len(M) != len(picks):
            raise ValueError('data and picks must have the same length')

        if not return_single:
            logger.info("Processing epoch : %d" % (i + 1))

        # SSP and whitening
        if info['projs']:
            M = np.dot(proj, M)
        M = np.dot(whitener, M)

        # Project to source space using beamformer weights
        # Linear inverse: do computation here or delayed
        if M.shape[0] < W.shape[0]:
            sol = (W, M)
        else:
            sol = np.dot(W, M)

        tstep = 1.0 / info['sfreq']

        yield sol

        #XXX Hack: force all vertices to have same val
        #sol = np.tile(sol, (n_sources, 1))
        #yield _make_stc(sol, vertices=vertno, tmin=tmin, tstep=tstep,
        #                subject=subject)

    logger.info('[done]')
