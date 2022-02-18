from dask.diagnostics import ProgressBar
from pathlib import Path
import zarr
from tqdm import tqdm
import traceback
import pandas as pd
import numpy as np
import time
import xarray as xr
import sys
from calibration import get_slice
import importlib
print('Done importing')

def sync_read(dates, swot_files, err_files_1d=None, err_files_2d=None):
    try:
        zarr_dir = Path('sensor_zarr/zarr')
        store = zarr.DirectoryStore(zarr_dir / 'tmp_err_swot')
        xr.Dataset().to_zarr(store, mode="w", consolidated=True)
        ds_err2d = xr.open_dataset(err_files_2d.pop(0))
        ds_err1d = xr.open_dataset(err_files_1d.pop(0))
        ds_swot = xr.open_dataset(swot_files.pop(0))
        swot_t0 = ds_swot.time.min().values
        err_t0 = ds_err2d.time.min().values
        prev_group =''
        add_vars_2d = {}
        add_vars_1d = {}
        for date in dates:
            print(date)


            # Read next swot files
            while len(swot_files) > 0 and (ds_swot.time.max() - ds_swot.time.min()) < 30 * 3600 :
                ds_swot = xr.concat([ds_swot, xr.open_dataset(swot_files.pop(0))], dim='time')
                keep_vars = [
                    'lon',
                    'lat',
                    'x_al',
                    'x_ac',
                    'lon_nadir',
                    'lat_nadir',
                    'model_index',
                    # 'timing_err',
                    'ssh_obs',
                    # 'roll_err',
                    # 'phase_err',
                    'ssh_model',
                    # 'bd_err',
                    # 'karin_err'
                ]
                swot_chunk = (
                        ds_swot[keep_vars].assign_coords(
                            dt=(('time',), pd.to_datetime('2012-10-01') + pd.to_timedelta(ds_swot.time, 's'))
                        ).swap_dims({'time':'dt'})
                        .sel(dt=str(date.date()))
                )

            # Read next 2d error files
            while len(err_files_2d) > 0 and (ds_err2d.time.max() - ds_err2d.time.min()) < 30 * 3600 :
                ds_err2d = xr.concat([ds_err2d, xr.open_dataset(err_files_2d.pop(0))], dim='num_lines')

                err_2d_chunk = (
                    ds_err2d.assign_coords(
                        dt=(('num_lines',), pd.to_datetime('2012-10-01') + pd.to_timedelta(ds_err2d.time - err_t0 + swot_t0, 's'))
                    ).swap_dims({'num_lines':'dt'})
                    .assign_coords(_x_ac=lambda ds: (('num_pixels',),ds.xac.isel(dt=0).values))
                    .swap_dims({'num_pixels': '_x_ac'})
                    .sel(_x_ac=np.concatenate([np.arange(-60, -8, 2), np.arange(10, 62, 2)]) * 1000, method='nearest')
                    .sel(dt=str(date.date()))
                )

                add_vars_2d = {
                    v: (
                            swot_chunk.ssh_model.dims,
                            err_2d_chunk[v].interp(dt=swot_chunk.dt, method='nearest').transpose('dt', '_x_ac').values
                        )
                    for v in [
                        # 'karin_noise',
                        # 'wet_tropo_res',
                        # 'syst_error_uncalibrated'
                    ]
                }

            # Read next 1d error files
            while len(err_files_1d) > 0 and (ds_err1d.time.max() - ds_err1d.time.min()) < 30 * 3600 :
                ds_err1d = xr.concat([ds_err1d, xr.open_dataset(err_files_1d.pop(0))], dim='num_lines')

                err_1d_chunk = (
                    ds_err1d.assign_coords(
                        dt=(('num_lines',), pd.to_datetime('2012-10-01') + pd.to_timedelta(ds_err1d.time - err_t0 + swot_t0, 's'))
                    ).swap_dims({'num_lines':'dt'})
                    .sel(dt=str(date.date()))
                )
                
                xac = err_2d_chunk._x_ac
                roll_gyro2d = err_1d_chunk.roll_gyro * xac * 10**-6
                roll_knlg_2d = err_1d_chunk.roll_knlg * xac * 10**-6
                roll_orb_2d = err_1d_chunk.roll_orb * xac * 10**-6
                phase_2d = xr.where(xac < 0, xac * err_1d_chunk.phase1 * 10**-6, xac * err_1d_chunk.phase2 * 10**-6)
                timing_2d = xr.where(xac < 0, err_1d_chunk.timing1, err_1d_chunk.timing2)
                bd_2d = err_1d_chunk.bd * (xac * 10**-3)**2

                # new_err_syst_err = (
                #         0
                #         + roll_gyro2d
                #         + roll_knlg_2d
                #         + roll_orb_2d
                #         + timing_2d
                #         + bd_2d
                #         + phase_2d
                # )

                # new_err_syst_err - err_2d_chunk.syst_error_uncalibrated

                add_vars_1d = {
                    v: (
                            swot_chunk.ssh_model.dims,
                            da.interp(dt=swot_chunk.dt, method='nearest').transpose('dt', '_x_ac').values
                        )
                    for v, da in { 
                        'roll_gyro2d': roll_gyro2d,
                        'roll_knlg_2d': roll_knlg_2d,
                        'roll_orb_2d': roll_orb_2d,
                        'timing_2d': timing_2d,
                        'bd_2d': bd_2d,
                        'phase_2d': phase_2d,
                    }.items()
                }
            
            chunk =  (
                swot_chunk
                .assign( **add_vars_2d)
                .assign( **add_vars_1d)
            )
            add_vars_2d = {}
            add_vars_1d = {}
            group = f"{date.year}/{date.month}"
            # print(day_ds.time)
            # xr.Dataset.to_zarr
            if prev_group != group:
                chunk.to_zarr(store=store, group=group, mode='a')
            else:
                chunk.to_zarr(store=store, group=group, append_dim="dt")
            
            prev_group = group
            ds_swot = ds_swot.drop_sel(time=swot_chunk.time)
            ds_err2d = ds_err2d.drop_isel(num_lines=ds_err2d.time <= err_2d_chunk.time.max().values)
            ds_err1d = ds_err1d.drop_isel(num_lines=ds_err1d.time <= err_1d_chunk.time.max().values)

    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()

    
#


def compress_and_clean(src_folder, tgt_folder):
    try:
        src_store = zarr.DirectoryStore(src_folder)
        tgt_store = zarr.DirectoryStore(tgt_folder)
        xr.Dataset().to_zarr(tgt_store, mode="w", consolidated=True)
        groups = [f"{dt.year}/{dt.month}" for dt in pd.date_range("2012-10-01", "2013-10-01", freq='MS')]
        for g in tqdm(groups):
            ds = xr.open_zarr(src_store, group=g)
            new_ds = (
                ds
                .chunk({'dt': 1000000, 'nC': 52})
                .isel(dt=np.isfinite(ds.ssh_model).any('nC'))
                .astype('float32', casting='same_kind')
                .drop('time')
                .rename({'dt': 'time'})
            )
                

            new_ds.chunk({'time': 1000000, 'nC': 52}).to_zarr(tgt_store, group=g, consolidated=True, mode="w")
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()


def main():
    try:

        zarr_dir = Path('sensor_zarr/zarr')
        src_folder = zarr_dir / 'tmp_err_swot'
        tgt_folder = zarr_dir / 'new_swot_with_1d'

        all_2d_files = sorted(list(Path('swot_errors').glob('*2DERROR*')))
        all_1d_files = sorted(list(Path('swot_errors').glob('*1DERROR*')))
        swot_files = sorted(list(Path('output_SWOTsimulator/swot_HD').glob('BOOST-SWOT_SWOT_c*_p*.nc')))
        swot_files = swot_files
        err_files_2d = all_2d_files
        err_files_1d = all_1d_files
        len(swot_files), len(err_files_1d), len(err_files_2d)
        dates = pd.date_range('2012-10-01', '2013-10-01')

        print("writing raw zarr")
        locals().update(sync_read(dates=dates, swot_files=swot_files, err_files_2d=err_files_2d, err_files_1d=err_files_1d))

        print("writing compressed zarr")
        locals().update(compress_and_clean(src_folder, tgt_folder))

    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()

