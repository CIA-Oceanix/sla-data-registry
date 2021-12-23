import pyinterp
import numpy as np
import xarray as xr
from pathlib import Path

def get_swot_slice(path, drop_vars=('model_index',),
                   **slice_args):
    dt_start = pd.to_datetime(slice_args.get('time_min', "2012-10-01"))
    dt_end = pd.to_datetime(slice_args.get('time_max', "2013-10-01"))
    groups = [f"{dt.year}/{dt.month}" for dt in
              pd.date_range(start=dt_start.date().replace(day=1), end=dt_end, freq='MS')]

    dses = []
    for group in groups:
        with xr.open_zarr(zarr.DirectoryStore(path), drop_variables=drop_vars, group=group,
                          decode_times=False,
                          consolidated=True,
                          synchronizer=zarr.ProcessSynchronizer(f'data/swot.sync')) as ds:
            units, reference_date = ds.time.attrs['units'].split('since')
            ts = (dt_start - pd.to_datetime(reference_date)).to_timedelta64().astype(float)
            te = (dt_end - pd.to_datetime(reference_date)).to_timedelta64().astype(float)
            dses.append(
                ds
                    .pipe(lambda ds: ds.isel(time=(ds.time < te) & (ds.time >= ts))).compute()
                    .pipe(lambda ds: ds.isel(time=(ds.lat_nadir > slice_args.get('lat_min', -360))))
                    .pipe(lambda ds: ds.isel(time=(ds.lat_nadir < slice_args.get('lat_max', 360))))
                    .pipe(lambda ds: ds.isel(time=(ds.lon_nadir < slice_args.get('lon_max', 360))))
                    .pipe(lambda ds: ds.isel(time=(ds.lon_nadir > slice_args.get('lon_min', -360)))).compute()
            )

    dses = [_ds for _ds in dses if _ds.dims['time']]
    if len(dses) == 0:
        print(
            f"no data found at {path} for {slice_args} {groups} {pd.date_range(start=dt_start.replace(day=1), end=dt_end, freq='MS')}")
        return None
    return xr.concat(
        [xr.decode_cf(_ds) for _ds in dses if _ds.dims['time']],
        dim="time"
    )


def get_nadir_slice(path, **slice_args):
    dt_start = pd.to_datetime(slice_args.get('time_min', "2012-10-01"))
    dt_end = pd.to_datetime(slice_args.get('time_max', "2013-10-01"))
    groups = [f"{dt.year}/{dt.month}" for dt in
              pd.date_range(start=dt_start.date().replace(day=1), end=dt_end, freq='MS')]

    dses = []
    for group in groups:
        with xr.open_zarr(zarr.DirectoryStore(path),
                          group=group, decode_times=False, consolidated=True,
                          synchronizer=zarr.ProcessSynchronizer(f'data/nadir.sync')) as ds:
            units, reference_date = ds.time.attrs['units'].split('since')

            ts = (dt_start - pd.to_datetime(reference_date)).to_timedelta64() / pd.to_timedelta(1, unit=units.strip())
            te = (dt_end - pd.to_datetime(reference_date)) / pd.to_timedelta(1, unit=units.strip())
            dses.append(
                ds
                    .pipe(lambda ds: ds.isel(time=(ds.time < te) & (ds.time >= ts))).compute()
                    .pipe(lambda ds: ds.isel(time=(ds.lat > slice_args.get('lat_min', -360))))
                    .pipe(lambda ds: ds.isel(time=(ds.lat < slice_args.get('lat_max', 360))))
                    .pipe(lambda ds: ds.isel(time=(ds.lon < slice_args.get('lon_max', 360))))
                    .pipe(lambda ds: ds.isel(time=(ds.lon > slice_args.get('lon_min', -360)))).compute()
            )
    dses = [_ds for _ds in dses if _ds.dims['time']]
    if len(dses) == 0:
        print(
            f"no data at {path} found for {slice_args} {groups} {pd.date_range(start=dt_start, end=dt_end, freq='MS')}")
        return None
    return xr.concat(
        [xr.decode_cf(_ds) for _ds in dses if _ds.dims['time']],
        dim="time"
    )


def main():
    root_dir = Path('../sla-data-registry') 
    four_nadirs = {
        'nadir_en': ['ssh_model'],
        'nadir_tpn': ['ssh_model'],
        'nadir_g2': ['ssh_model'],
        'nadir_j1': ['ssh_model'],
    }
    five_nadirs = {
        **four_nadirs,
        'nadir_swot': ['ssh_model'],
    }

    grid_obs_vars = {
            'four_nadirs': four_nadirs,
            'five_nadirs': five_nadirs,
            'swot_nadirs_no_noise': {
                **five_nadirs,
                'swot': ['ssh_model'],
            },
            'swot_no_noise': {
                **five_nadirs,
                'swot': ['ssh_model'],
            },
            'swot_nadirs_old_errors': {
                **five_nadirs,
                'swot': ['ssh_model', 'phase_err', 'karin_err', 'roll_err', 'bd_err', 'timing_err'],
            },
            'swot_nadirs_new_errors_no_wet_tropo': {
                **five_nadirs,
                'swot': ['ssh_model', 'syst_error_uncalibrated', 'karin_noise'],
            },
            'swot_nadirs_new_errors_w_wet_tropo': {
                **five_nadirs,
                'swot': ['ssh_model', 'syst_error_uncalibrated', 'wet_tropo_res', 'karin_noise'],
            },
            'swot_new_errors_no_wet_tropo': {
                'swot': ['ssh_model', 'syst_error_uncalibrated', 'karin_noise'],
            },
            'swot_new_errors_w_wet_tropo': {
                'swot': ['ssh_model', 'syst_error_uncalibrated', 'wet_tropo_res', 'karin_noise'],
            },
    }
    
    tgt_grid = xr.open_dataset(root_dir / 'NATL60/NATL/data_new/dataset_nadir_0d.nc')[['time', 'lat', 'lon']]
    binning = pyinterp.Binning2D(pyinterp.Axis(tgt_grid.lon.values), pyinterp.Axis(tgt_grid.lat.values))

    grid_day_dses = []
    # dt0 = dt_start
    for dt_start, dt_end in zip(tgt_grid.time[:-1].values, tgt_grid.time[1:].values):
        # if dt_start < dt0:
        #     continue
        print(dt_start, end='\r')
        slice_args = {
            "time_min": dt_start,
            "time_max": dt_end,
        }

        t0 = time.time()
        obs_data = {
            **{f'nadir_{name}': get_nadir_slice(root_dir / f'sensor_zarr/zarr/nadir/{name}', **slice_args) for name in
               ['swot', 'en', 'tpn', 'g2', 'j1']},
            'swot': get_swot_slice(root_dir  / f'sensor_zarr/zarr/new_swot', **slice_args),
        }
        # print(time.time() - t0)

        tgt_day_vars = {}
        for tgt_var, obs_vars in grid_obs_vars.items():

            binning.clear()
            for k, ds in obs_data.items():

                if len(obs_vars.get(k, [])) == 0:
                    continue

                if ds is None:
                    continue

                ds_value = ds[obs_vars[k][0]]
                for v in obs_vars[k][1:]:
                    ds_value = ds_value + ds[v]

                values = np.ravel(ds_value.values)
                lons = np.ravel(ds.lon.values) - 360
                lats = np.ravel(ds.lat.values)

                msk = np.isfinite(values)
                binning.push(lons[msk], lats[msk], values[msk])

            tgt_day_vars[tgt_var] =  (('time', 'lat', 'lon'), binning.variable('mean').T[None, ...])

       
        grid_day_dses.append(
           xr.Dataset(
               tgt_day_vars,
               {'time': [dt_start + (dt_end - dt_start) / 2], 'lat': np.array(binning.y), 'lon': np.array(binning.x)}
            ).astype('float32', casting='same_kind')
        )

    full_cal_ds = xr.concat(grid_day_dses, dim='time')

    full_cal_ds.to_netcdf(root_dir / 'CalData/cal_data_new_errs.nc')

    


