from mne_bids import BIDSPath, read_raw_bids


def process_input(path,
                  subject,
                  task,
                  resampling_frq=None,
                  ref_chs=None,
                  filter_bounds=None,
                  verbose=False):

    """
    Parameters
    ----------
    path : str
        path to xarray dataset

    subjects : list of str or int

    task : str

    resampling_frq : int
        resampling frequency

    ref_chs : list of str

    filter_bounds : tuple of float
        first item in tuple is the lower pass-band edge and second item is the upper
        pass-band edge

    Returns
    -------
    mne.io.Raw

    """

    # open data
    print(f'>>>>>>>>preprocessing {subject}...')
    bids_path = BIDSPath(subject=subject,
                         session='01',
                         task=task,
                         root=path
                         )
    raw = read_raw_bids(bids_path, extra_params={'preload': True}, verbose=False)
    raw.pick_types(eeg=True)

    # interpolate bad channels
    if raw.info['bads'] != []:
        raw.interpolate_bads()

    # resampling
    raw.resample(resampling_frq)

    # filtering
    if filter_bounds is not None:
        raw.filter(
            l_freq=filter_bounds[0],
            h_freq=filter_bounds[1],
            verbose=verbose
        )

    # rereferencing
    if ref_chs is not None:
        raw.add_reference_channels(ref_channels='FCz')  # adding reference channel to the data
        raw.set_eeg_reference(ref_channels=ref_chs, verbose=verbose)

    return raw


# setup data


# import data
