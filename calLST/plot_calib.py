
from matplotlib import pyplot as plt
from ctapipe.visualization import CameraDisplay
import numpy as np
from ctapipe.instrument import TelescopeDescription, CameraGeometry
from ctapipe.coordinates import EngineeringCameraFrame
# read back the monitoring containers written with the tool calc_camera_calibration.py
from ctapipe.containers import FlatFieldContainer, WaveformCalibrationContainer, PedestalContainer, \
    PixelStatusContainer

from ctapipe.io.hdf5tableio import HDF5TableReader
from ctapipe_io_lst import load_camera_geometry
camera=load_camera_geometry()
camera = camera.transform_to(EngineeringCameraFrame())

__all__ = ['read_file',

           'plot_time',
           'plot_time_pe',
           'plot_charge',
           'plot_gains',
           'plot_all',
           'plot_gain',
           'plot_signal_pe',
           'plot_signal_pe_gain',
           'plot_only_gain',
           'plot_gain_charge',
           'plot_only_charge'
           ]

ff_data = FlatFieldContainer()
ped_data = PedestalContainer()
calib_data = WaveformCalibrationContainer()
status_data = PixelStatusContainer()

channel = ['HG', 'LG']

plot_dir="comp_plot"


def read_file(file_name='/astro/users/cassol/soft/python/flatfield/LST1/calibration.hdf5', tel_id=1):

    with HDF5TableReader(file_name) as h5_table:

        try:
             assert h5_table._h5file.isopen == True



             table = f"/tel_{tel_id}/flatfield"
             next(h5_table.read(table, ff_data))
             # print(f"read flatfield n. event: {ff_data.n_events}")
             table = f"/tel_{tel_id}/calibration"
             next(h5_table.read(table, calib_data))
             # print("read calib")
             table = f"/tel_{tel_id}/pedestal"
             next(h5_table.read(table, ped_data))
             table = f"/tel_{tel_id}/pixel_status"
             # print(f"read pedestal n. event: {ped_data.n_events}")
             next(h5_table.read(table, status_data))
             # print(f"read status")
        except Exception:
            print(f"----> no correct tables {table} in {file_name}")
            h5_table.close()
            return False

    h5_table.close()
    return True


def plot_charge(cal_data, ff_data, ped_data, run):
    mask = cal_data.unusable_pixels

    # select good pixels
    chan = 0
    select = np.logical_not(mask[chan])
    charge_median = ff_data.charge_median[chan]
    charge_std = ff_data.charge_std[chan]

    fig = plt.figure(chan, figsize=(12, 12))
    # charge
    plt.subplot(221)
    median = np.median(charge_median[select])
    rms = np.std(charge_median[select])
    plt.xlabel('HG charge (ADC)', fontsize=20)
    plt.ylabel('pixels', fontsize=20)

    label = f"Median {median:3.2f}, std {rms:5.2f}"
    plt.hist(charge_median[select], label=label, range=(4000, 30000), histtype='step', bins=50, stacked=True, alpha=0.5,
             fill=True)

    fig = plt.figure(chan, figsize=(12, 18))
    chan = 0
    select = np.logical_not(mask[chan])
    charge_median = ff_data.charge_median[chan]
    charge_std = ff_data.charge_std[chan]
    # charge
    plt.subplot(122)
    median = np.median(charge_median[select])
    rms = np.std(charge_median[select])

    plt.xlabel('LG charge (ADC)', fontsize=20)
    plt.ylabel('pixels', fontsize=20)

    label = f"Median {median:3.2f}, std {rms:5.2f}"
    plt.hist(charge_median[select], label=label, range=(200, 5000), histtype='step', bins=50, stacked=True, alpha=0.5,
             fill=True)


def plot_time_pe(cal_data, run=0, fill=True):

    mask = cal_data.unusable_pixels

    for chan in np.arange(2):
        n_pe = cal_data.n_pe[chan]
        time = cal_data.time_correction[chan]
        # select good pixels
        select = np.logical_not(mask[chan])

        fig = plt.figure(chan, figsize=(12, 6))
        fig.suptitle(f"channel: {channel[chan]}", fontsize=25)

        plt.subplot(121)
        median = int(np.median(n_pe[select]))
        rms = np.std(n_pe[select])

        plt.xlabel(' pe', fontsize=20)
        plt.ylabel('pixels', fontsize=20)

        label = f"Run {run} Median {median:3.2f}, std {rms:5.2f}"

        plt.hist(n_pe[select], label=label, histtype='step', range=(50, 110), bins=50, stacked=True, alpha=0.5,
                 fill=fill)

        plt.legend()

        # relative gain
        plt.subplot(122)
        plt.ylabel('pixels', fontsize=20)
        plt.xlabel('time', fontsize=20)

        median = np.median(time[select])
        rms = np.std(time[select])
        # plt.title(f"Relative gain {median:3.2f}, std {rms:5.2f}")
        label = f"Run {run} Median {median:3.2f}, std {rms:5.2f}"
        plt.hist(time[select], label=label, histtype='step', bins=50, stacked=True, alpha=0.5, fill=fill)

        plt.legend()
        if plot_dir != "none":
            plt.savefig(f"{plot_dir}/Run{run}_n_pe_time_{channel[chan]}.png, dpi = 600")

        # time
    fig = plt.figure(run, figsize=(16, 5))
    image = ff_data.time_median
    for chan in (np.arange(2)):
        ax = plt.subplot(1, 2, chan + 1)
        disp = CameraDisplay(camera)
        disp.highlight_pixels(mask[chan], linewidth=2)
        disp.image = image[chan]
        disp.cmap = plt.cm.coolwarm
        disp.axes.text(2.4, 0, f'{channel[chan]} time', rotation=90)
        disp.add_colorbar()
    if plot_dir != "none":
        plt.savefig(f"{plot_dir}/Run{run}_flatfield_time_over_camera.png", dpi = 600)


def plot_pe(cal_data, run=0):

    mask = cal_data.unusable_pixels
    fig = plt.figure(1, figsize=(12, 6))
    fig.suptitle(f"Run {run}", fontsize=25)
    plt.subplot(1, 2, 1)
    for chan in np.arange(2):
        n_pe = cal_data.n_pe[chan]

        # select good pixels
        select = np.logical_not(mask[chan])

        median = int(np.median(n_pe[select]))
        rms = np.std(n_pe[select])

        plt.xlabel(f'pe', fontsize=20)
        plt.ylabel('pixels', fontsize=20)

        label = f"{channel[chan]} Median {median:3.2f}, std {rms:5.2f}"

        plt.hist(n_pe[select], label=label, histtype='step', range=(50, 110), bins=50, stacked=True, alpha=0.5,
                 fill=True)

        plt.legend()

    HG = cal_data.n_pe[0]
    LG = cal_data.n_pe[1]
    HG = np.where(np.isnan(HG), 0, HG)
    LG = np.where(np.isnan(LG), 0, LG)
    mymin = np.median(LG) - 2 * np.std(LG)
    mymax = np.median(LG) + 2 * np.std(LG)
    plt.subplot(1, 2, 2)

    plt.hist2d(LG, HG, bins=[100, 100])
    plt.xlabel("LG", fontsize=20)
    plt.ylabel("HG", fontsize=20)

    x = np.arange(mymin,mymax)
    plt.plot(x, x)

    plt.ylim(mymin, mymax)
    plt.xlim(mymin, mymax)

    if plot_dir != "none":
       plt.savefig(f"{plot_dir}/Run{run}_n_pe.png", dpi = 600)

def plot_gain(ff_data,cal_data,run=0):
    # plot open pdf

    plt.rc('font', size=18)
    mask = cal_data.unusable_pixels

    ### first figure
    fig = plt.figure(1, figsize=(8, 8))

   # plt.tight_layout()
    fig.suptitle(f"Run {run}", fontsize=25)
    pad = 110
    for chan in np.arange(2):
        select = np.logical_not(mask[chan])
        gain_median = ff_data.relative_gain_median[chan]

        plt.ylabel('pixels', fontsize=20)
        plt.xlabel('relative charge', fontsize=20)
        plt.ylim(0,300)

        median = np.median(gain_median[select])
        rms = np.std(gain_median[select])
        # plt.title(f"Relative gain {median:3.2f}, std {rms:5.2f}")
        label = f"{channel[chan]} Median {median:3.2f}, std {rms:5.2f}"
        plt.hist(gain_median[select], label=label, histtype='step', range=(0.6, 1.5), bins=50, stacked=True, alpha=0.5,
                 fill=True)

        plt.legend()
        if plot_dir != "none":
            plt.savefig(f"{plot_dir}/relative_gain_Run{run}.png", dpi = 600)


def plot_gains(ff_data, cal_data, run=0,fill=True,tag=""):
    if fill == True:
        alpha=0.6
    else:
        alpha=0.75
    mask = cal_data.unusable_pixels

    for chan in np.arange(2):
        n_pe = cal_data.n_pe[chan]
        dc_for_pe = 1. / cal_data.dc_to_pe[chan]
        gain_median = ff_data.relative_gain_median[chan]
        charge_median = ff_data.charge_median[chan]
        charge_mean = ff_data.charge_mean[chan]

        charge_std = ff_data.charge_std[chan]
        #print(np.where(mask == 1))
        # select good pixels
        select = np.logical_not(mask[chan])
        # select = mask[chan]
        # print(select)

        fig = plt.figure(chan, figsize=(12, 12))

        plt.subplot(221)
        # charge

        median = np.median(charge_median[select])
        rms = np.std(charge_median[select])

        #plt.title(f"Median {median:3.2f}, std {rms:5.0f}")
        label=f"run {run}, {median:3.2f}+- {rms:3.2f}"
        plt.xlabel('charge (ADC)', fontsize=15)
        plt.ylabel('pixels', fontsize=15)

        #label = f"run {run}"
        if chan==0:
            plt.hist(charge_mean[select], label=label, linewidth=2, histtype='step',alpha=alpha, bins=50,range=(5000,10000),fill=fill)
        else:
            plt.hist(charge_mean[select], label=label, linewidth=2, histtype='step', alpha=alpha, bins=50,range=(300,700),fill=fill)
        plt.legend()
        plt.tight_layout()

        plt.subplot(222)

        fig.suptitle(f"channel: {channel[chan]}", fontsize=25)
        n_pe = cal_data.n_pe[chan]

        # select good pixels
        select = np.logical_not(mask[chan])

        median = np.median(n_pe[select])
        rms = np.std(n_pe[select])

        plt.xlabel(f'photon-electrons', fontsize=20)
        plt.ylabel('pixels', fontsize=20)
        plt.tight_layout()
        label = f"{run} {median:3.2f} +- {rms:5.2f}"

        plt.hist(n_pe[select], label=label, histtype='step',  linewidth=2,range=(50, 110), bins=50, stacked=True, alpha=alpha,
                 fill=fill)

        plt.legend()

        plt.subplot(223)
        median = np.median(dc_for_pe[select])
        rms = np.std(dc_for_pe[select])

        plt.xlabel('ADC counts per pe', fontsize=20)
        plt.ylabel('pixels', fontsize=20)

        label = f"{run} {median:3.2f} +- {rms:5.2f}"
        if (chan == 0):
            plt.hist(dc_for_pe[select], label=label,linewidth=2, histtype='step', range=(50, 150), bins=50, stacked=True, alpha=alpha,
                     fill=fill)
        else:
            plt.hist(dc_for_pe[select], label=label,linewidth=2, histtype='step', range=(1, 10), bins=50, stacked=True, alpha=alpha,
                     fill=fill)
        plt.tight_layout()
        plt.legend()

        # relative gain
        plt.subplot(224)
        plt.ylabel('pixels', fontsize=20)
        plt.xlabel('relative signal', fontsize=20)

        median = np.median(gain_median[select])
        rms = np.std(gain_median[select])
        # plt.title(f"Relative gain {median:3.2f}, std {rms:5.2f}")
        label = f"Run {run} Median {median:3.2f}, std {rms:5.2f}"
        plt.hist(gain_median[select], label=label, histtype='step', linewidth=2, range=(0.7, 1.3), bins=50, stacked=True, alpha=0.75,
                 fill=fill)

        plt.legend()
        plt.tight_layout()
        if plot_dir != "none":
            plt.savefig(f"{plot_dir}/gain_adc_for_pe_{channel[chan]}.png")


def plot_only_gain(ff_data, cal_data, run=0,fill=True):

    if fill==True:
        alpha=0.6
    else:
        alpha=0.75

    mask = cal_data.unusable_pixels

    for chan in np.arange(2):
        n_pe = cal_data.n_pe[chan]
        dc_for_pe = 1. / cal_data.dc_to_pe[chan]
        gain_median = ff_data.relative_gain_median[chan]
        charge_median = ff_data.charge_median[chan]
        charge_std = ff_data.charge_std[chan]
        #print(np.where(mask == 1))
        # select good pixels
        select = np.logical_not(mask[chan])
        # select = mask[chan]
        # print(select)

        fig = plt.figure(chan, figsize=(12, 6))
        fig.suptitle(f"channel: {channel[chan]}", fontsize=25)

        plt.subplot(121)
        median = np.median(dc_for_pe[select])
        rms = np.std(dc_for_pe[select])

        plt.xlabel('gain (DC/pe)', fontsize=20)
        plt.ylabel('pixels', fontsize=20)

        label = f"Run {run} Median {median:3.3f}, std {rms:5.3f}"
        if (chan == 0):
            plt.hist(dc_for_pe[select], label=label,linewidth=2,
        histtype='step', range=(60, 120), bins=50, stacked=True,
                     alpha=alpha,
                     fill=fill)
        else:
            plt.hist(dc_for_pe[select], label=label,linewidth=2,
                     histtype='step', range=(2, 8), bins=50, stacked=True,
                     alpha=alpha,
                     fill=fill)

        plt.legend()

        # relative gain
        plt.subplot(122)
        plt.ylabel('pixels', fontsize=20)
        plt.xlabel('relative gain', fontsize=20)

        median = np.median(gain_median[select])
        rms = np.std(gain_median[select])
        # plt.title(f"Relative gain {median:3.2f}, std {rms:5.2f}")
        label = f"Run {run} Median {median:3.3f}, std {rms:5.3f}"
        plt.hist(gain_median[select], label=label,
                 histtype='step',range=(0.8, 1.2),bins=50, stacked=True,
                 alpha=alpha, linewidth=2,
                 fill=fill)

        plt.legend()
        if plot_dir != "none":
            plt.savefig(f"{plot_dir}/gain_adc_for_pe_{channel[chan]}.png")


def plot_time(ff_data, cal_data, run=0,tag=""):
    mask = cal_data.unusable_pixels

    for chan in np.arange(2):
        n_pe = cal_data.n_pe[chan]
        time = cal_data.time_correction[chan]
        # select good pixels
        select = np.logical_not(mask[chan])

        fig = plt.figure(chan, figsize=(12, 6))
        # fig.suptitle(f"channel: {channel[chan]}", fontsize=25)

        plt.ylabel('pixels', fontsize=20)
        plt.xlabel('time corrections', fontsize=20)

        median = np.median(time[select])
        rms = np.std(time[select])
        # plt.title(f"Relative gain {median:3.2f}, std {rms:5.2f}")
        label = f"{tag} run {run} {channel[chan]} Median {median:3.2f}, std {rms:5.2f}"
        plt.hist(time[select], label=label, histtype='step', range=(-10, 10), bins=50, stacked=True, alpha=0.5, fill=True)

        plt.legend()
        if plot_dir != "none":
            plt.savefig(f"{plot_dir}/run{run}_time_corr_{channel[chan]}.png", dpi = 600)

    for chan in np.arange(2):
        time_median = ff_data.time_median[chan]
        time_std = ff_data.time_std[chan]
        # select good pixels
        select = np.logical_not(mask[chan])

        fig = plt.figure(10 + chan, figsize=(12, 6))
        plt.subplot(121)
        # fig.suptitle(f"channel: {channel[chan]}", fontsize=25)

        plt.ylabel('pixels', fontsize=20)
        plt.xlabel('time (ns)', fontsize=20)

        median = np.median(time_median[select])
        rms = np.std(time_median[select])
        # plt.title(f"Relative gain {median:3.2f}, std {rms:5.2f}")
        label = f"{tag} run {run} {channel[chan]} Median {median:3.2f}, std {rms:5.2f}"
        plt.hist(time_median[select], label=label, histtype='step', range=(10, 25), bins=50, stacked=True, alpha=0.5,
                 fill=True)

        plt.legend()
        plt.subplot(122)
        # fig.suptitle(f"channel: {channel[chan]}", fontsize=25)

        plt.ylabel('pixels', fontsize=20)
        plt.xlabel('time std (ns)', fontsize=20)

        median = np.median(time_std[select])
        rms = np.std(time_std[select])
        # plt.title(f"Relative gain {median:3.2f}, std {rms:5.2f}")
        label = f"{tag} run {run} {channel[chan]} Median {median:3.2f}, std {rms:5.2f}"
        plt.hist(time_std[select], label=label, histtype='step', range=(0.5, 2), bins=50, stacked=True, alpha=0.5,
                 fill=True)

        plt.legend()
        if plot_dir != "none":
            plt.savefig(f"{plot_dir}/run{run}_time_{channel[chan]}.png", dpi = 600)


def plot_all(ped_data, ff_data, calib_data, run=0, color='r'):

    # plot results
    mask = calib_data.unusable_pixels
    # print(np.where(mask==1))
    # charge
    channel = ["HG", "LG"]
    fig = plt.figure(10, figsize=(16, 5))
    image = ff_data.charge_median
    mask = ff_data.charge_median_outliers
    for chan in (np.arange(2)):
        ax = plt.subplot(1, 2, chan + 1)

        disp = CameraDisplay(camera)
        #        if chan == 0:
        #            disp.set_limits_minmax(5000,10000)
        #        else:
        #            disp.set_limits_minmax(200,800)
        mymin = np.median(image[chan]) - 2 * np.std(image[chan])
        mymax = np.median(image[chan]) + 2 * np.std(image[chan])
        disp.set_limits_minmax(mymin, mymax)
        disp.highlight_pixels(mask[chan], linewidth=2)
        disp.image = image[chan]
        disp.cmap = plt.cm.coolwarm
        disp.axes.text(2.4, 0, f'{channel[chan]} signal charge (ADC)', rotation=90)
        disp.add_colorbar()
    if plot_dir != "none":
        plt.savefig(f"{plot_dir}/flatfield_charge_over_camera_run{run}.png")
    # charge std

    fig = plt.figure(7, figsize=(16, 5))
    image = ff_data.charge_std
    mask = ff_data.charge_std_outliers
    for chan in (np.arange(2)):
        ax = plt.subplot(1, 2, chan + 1)
        disp = CameraDisplay(camera)
        #        if chan == 0:
        #            disp.set_limits_minmax(100,1500)
        #        else:
        #            disp.set_limits_minmax(20,90)
        mymin = np.median(image[chan]) - 2 * np.std(image[chan])
        mymax = np.median(image[chan]) + 2 * np.std(image[chan])
        disp.set_limits_minmax(mymin, mymax)
        disp.highlight_pixels(mask[chan], linewidth=2)
        disp.image = image[chan]
        disp.cmap = plt.cm.coolwarm
        disp.axes.text(2.4, 0, f'{channel[chan]} signal std [ADC]', rotation=90)
        disp.add_colorbar()
    if plot_dir != "none":
        plt.savefig(f"{plot_dir}/flatfield_charge_std_over_camera_run{run}.png", dpi = 600)
    # ped charge

    fig = plt.figure(9, figsize=(16, 5))
    image = ped_data.charge_median
    mask = ped_data.charge_median_outliers
    for chan in (np.arange(2)):
        ax = plt.subplot(1, 2, chan + 1)
        disp = CameraDisplay(camera)
        mymin = np.median(image[chan]) - 2 * np.std(image[chan])
        mymax = np.median(image[chan]) + 2 * np.std(image[chan])
        disp.set_limits_minmax(mymin, mymax)
        disp.highlight_pixels(mask[chan], linewidth=2)
        disp.image = image[chan]
        disp.cmap = plt.cm.coolwarm
        disp.axes.text(2.4, 0, f'{channel[chan]} pedestal [ADC]', rotation=90)
        disp.add_colorbar()
    if plot_dir != "none":
        plt.savefig(f"{plot_dir}/pedestal_charge_over_camera_run{run}.png", dpi = 300)

    # ped charge std
    # return
    fig = plt.figure(8, figsize=(16, 5))
    image = ped_data.charge_std
    mask =  ped_data.charge_std_outliers
    for chan in (np.arange(2)):
        ax = plt.subplot(1, 2, chan + 1)
        disp = CameraDisplay(camera)
        mymin = np.median(image[chan]) - 2 * np.std(image[chan])
        mymax = np.median(image[chan]) + 2 * np.std(image[chan])
        disp.set_limits_minmax(mymin, mymax)
        disp.highlight_pixels(mask[chan], linewidth=2)
        disp.image = image[chan]
        disp.cmap = plt.cm.coolwarm
        disp.axes.text(2.4, 0, f'{channel[chan]} pedestal std [ADC]', rotation=90)
        disp.add_colorbar()
    if plot_dir != "none":
        plt.savefig(f"{plot_dir}/pedestal_charge_std_over_camera_run{run}.png", dpi = 600)

    # time
    fig = plt.figure(11, figsize=(16, 5))
    image = ff_data.time_median
    mask = ff_data.time_median_outliers
    for chan in (np.arange(2)):
        ax = plt.subplot(1, 2, chan + 1)
        disp = CameraDisplay(camera)
        disp.highlight_pixels(mask[chan], linewidth=2)
        #mymin = np.median(image[chan]) - 3 * np.std(image[chan])
        #mymax = np.median(image[chan]) + 3 * np.std(image[chan])
        #disp.set_limits_minmax(mymin, mymax)
        disp.image = image[chan]
        disp.cmap = plt.cm.coolwarm
        disp.axes.text(2.4, 0, f'{channel[chan]} time', rotation=90)
        disp.add_colorbar()
    if plot_dir != "none":
        plt.savefig(f"{plot_dir}/flatfield_time_over_camera_run{run}.png", dpi = 600)

    # gain
    fig = plt.figure(12, figsize=(16, 5))
    image = ff_data.relative_gain_median
    mask = calib_data.unusable_pixels
    for chan in (np.arange(2)):
        ax = plt.subplot(1, 2, chan + 1)
        disp = CameraDisplay(camera)
        disp.highlight_pixels(mask[chan], linewidth=2)
        mymin = np.median(image[chan]) - 2 * np.std(image[chan])
        mymax = np.median(image[chan]) + 2 * np.std(image[chan])
        disp.set_limits_minmax(mymin, mymax)
        disp.image = image[chan]
        disp.cmap = plt.cm.coolwarm
        disp.set_limits_minmax(0.7, 1.3)
        disp.axes.text(2.4, 0, f'{channel[chan]} relative gain', rotation=90)
        disp.add_colorbar()
    if plot_dir != "none":
        plt.savefig(f"{plot_dir}/flatfield_gain_over_camera_run{run}.png", dpi = 600)

    # pe
    fig = plt.figure(13, figsize=(16, 5))
    image = calib_data.n_pe
    mask = calib_data.unusable_pixels
    image = np.where(np.isnan(image), 0, image)
    for chan in (np.arange(2)):
        ax = plt.subplot(1, 2, chan + 1)
        disp = CameraDisplay(camera)
        disp.highlight_pixels(mask[chan], linewidth=2)
        disp.image = image[chan]
        mymin= np.median(image[chan])- 2*np.std(image[chan])
        mymax= np.median(image[chan])+ 2*np.std(image[chan])
        disp.set_limits_minmax(mymin, mymax)
        print(f"mymin mymax {mymin} {mymax}")

        disp.cmap = plt.cm.coolwarm
        disp.axes.text(2.4, 0, f'{channel[chan]} photon-electrons', rotation=90)
        disp.add_colorbar()
    if plot_dir != "none":
        plt.savefig(f"{plot_dir}/flatfield_pe_over_camera_run{run}.png", dpi = 600)
    # plot some histograms
    for chan in np.arange(2):
        n_pe = calib_data.n_pe[chan]

        gain_median = ff_data.relative_gain_median[chan]
        charge_median = ff_data.charge_median[chan]
        charge_mean = ff_data.charge_mean[chan]
        charge_std = ff_data.charge_std[chan]
        median_ped = ped_data.charge_median[chan]
        mean_ped = ped_data.charge_mean[chan]
        ped_std = ped_data.charge_std[chan]

        # select good pixels
        select = np.logical_not(mask[chan])
        # select = mask[chan]

        fig = plt.figure(chan, figsize=(12, 18))
        fig.suptitle(f"channel: {channel[chan]}", fontsize=25)

        # charge
        plt.subplot(321)
        median = int(np.median(charge_mean[select]))
        rms = np.std(charge_mean[select])

        plt.title(f"Median {median:3.2f}, std {rms:5.0f}")
        plt.xlabel('charge (ADC)', fontsize=20)
        plt.ylabel('pixels', fontsize=20)

        label = f"before: std {rms:5.1f}"
        plt.hist(charge_mean[select], bins=50)

        plt.legend()
        # signal std
        plt.subplot(322)
        plt.ylabel('pixels', fontsize=20)
        plt.xlabel('charge std', fontsize=20)
        median = np.median(charge_std[select])
        rms = np.std(charge_std[select])
        plt.title(f"Median {median:3.2f}, std {rms:3.2f}")
        plt.hist(charge_std[select], bins=50)

        # pedestal charge
        plt.subplot(323)
        plt.ylabel('pixels', fontsize=20)
        plt.xlabel('pedestal', fontsize=20)
        median = np.median(mean_ped[select])
        rms = np.std(mean_ped[select])
        plt.title(f"Median {median:3.2f}, std {rms:3.2f}")
        plt.hist(mean_ped[select], bins=50)

        # pedestal std
        plt.subplot(324)
        plt.ylabel('pixels', fontsize=20)
        plt.xlabel('pedestal std', fontsize=20)
        median = np.median(ped_std[select])
        rms = np.std(ped_std[select])
        plt.title(f"Median {median:3.2f}, std {rms:3.2f}")
        plt.hist(ped_std[select], bins=50)

        # relative gain
        plt.subplot(325)
        plt.ylabel('pixels', fontsize=20)
        plt.xlabel('relative gain', fontsize=20)
        plt.hist(gain_median[select], bins=50)

        median = np.median(gain_median[select])
        rms = np.std(gain_median[select])
        plt.title(f"Relative gain {median:3.2f}, std {rms:5.2f}")

        # photon electrons
        plt.subplot(326)
        plt.ylabel('pixels', fontsize=20)
        plt.xlabel('pe', fontsize=20)
        median = np.median(n_pe[select])
        rms = np.std(n_pe[select])
        plt.title(f"Median {median:3.2f}, std {rms:3.2f}")
        plt.hist(n_pe[select], bins=50)
        if plot_dir != "none":
            plt.savefig(f"{plot_dir}/charge_gain_distributions_{channel[chan]}_run{run}.png", dpi = 600)


def plot_gain_over_camera(calib_data,run):
    fig = plt.figure(1, figsize=(12, 6))
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])

    fig.suptitle(f"Run {run}", fontsize=25)


    pad = 120
    camera = CameraGeometry.from_name("LSTCam", 2)
    image = 1/calib_data.dc_to_pe
    mask = calib_data.unusable_pixels
    image = np.where(np.isnan(image), 0, image)
    for chan in (np.arange(2)):
        pad += 1
        plt.subplot(pad)
        plt.tight_layout()
        disp = CameraDisplay(camera)
        #disp.highlight_pixels(mask[chan], linewidth=2)

        disp.image = image[chan]
        mymin= np.median(image[chan])- 2*np.std(image[chan])
        mymax= np.median(image[chan])+ 2*np.std(image[chan])
        disp.set_limits_minmax(mymin, mymax)
        disp.cmap = plt.cm.coolwarm
        plt.title(f'{channel[chan]} ADC per photon-electrons')
        #disp.axes.text(lposx, 0, f'{channel[chan]} photon-electrons', rotation=90)
        disp.add_colorbar()
        plt.tight_layout()
        if plot_dir != "none":
            plt.savefig(f"{plot_dir}/charge_gain_over_camera_run{run}.png", dpi = 600)


def plot_charges(ped_data, ff_data, calib_data, run=0, color='r',fill=True, tag=""):
    mask = calib_data.unusable_pixels
    channel = ["HG", "LG"]
    if fill == True:
        alpha=0.6
    else:
        alpha=0.75
    for chan in np.arange(2):
        n_pe = calib_data.n_pe[chan]

        gain_median = ff_data.relative_gain_median[chan]
        charge_median = ff_data.charge_median[chan]
        charge_mean = ff_data.charge_mean[chan]
        charge_std = ff_data.charge_std[chan]
        median_ped = ped_data.charge_median[chan]
        mean_ped = ped_data.charge_mean[chan]
        ped_std = ped_data.charge_std[chan]

        # select good pixels
        select = np.logical_not(mask[chan])
        # select = mask[chan]

        fig = plt.figure(chan, figsize=(12, 12))
        fig.suptitle(f"channel: {channel[chan]}", fontsize=25)
        pad=221
        # charge
        plt.subplot(pad)
        median = int(np.median(charge_mean[select]))
        rms = np.std(charge_mean[select])

        #plt.title(f"Median {median:3.2f}, std {rms:5.0f}")
        label=f"run {run}, median {median:3.0f}+- {rms:3.1f}"
        plt.xlabel('charge [DC]', fontsize=20)
        plt.ylabel('pixels', fontsize=20)

        #label = f"run {run}"
        if chan==0:
            plt.hist(charge_mean[select], label=label, linewidth=2, histtype='step',alpha=alpha, bins=50,range=(5000,10000),fill=fill)
        else:
            plt.hist(charge_mean[select], label=label, linewidth=2, histtype='step', alpha=alpha, bins=50,range=(300,700),fill=fill)
        plt.legend()
        # signal std
        pad+=1
        plt.subplot(pad)
        plt.ylabel('pixels', fontsize=20)
        plt.xlabel('charge std [DC]', fontsize=20)
        median = np.median(charge_std[select])
        rms = np.std(charge_std[select])
        #plt.title(f"Median {median:3.2f}, std {rms:3.2f}")
        label = f"run {run}, median {median:3.0f}+- {rms:3.1f}"
        if chan==0:
            plt.hist(charge_std[select],label=label, linewidth=2, histtype='step', bins=50,alpha=alpha,range=(700,1200),fill=fill)
        else:
            plt.hist(charge_std[select], label=label, linewidth=2, histtype='step', bins=50, alpha=alpha, range=(40,70),fill=fill)
        plt.legend()
        # pedestal charge
        pad+=1
        plt.subplot(pad)
        plt.ylabel('pixels', fontsize=20)
        plt.xlabel('pedestal [DC]', fontsize=20)
        median = np.median(mean_ped[select])
        rms = np.std(mean_ped[select])
        label = f"run {run}: median {median:3.1f} +- {rms:3.1f}"
        #plt.title(f"Median {median:3.2f}, std {rms:3.2f}")
        if chan==0:
            plt.hist(mean_ped[select],label=label, linewidth=2, histtype='step', bins=50,alpha=alpha,range=(-30,30),fill=fill)
        else:
            plt.hist(mean_ped[select], label=label, linewidth=2, histtype='step', bins=50, alpha=alpha,range=(-30,30),fill=fill)
        plt.legend()
        pad+=1
        # pedestal std
        plt.subplot(pad)
        plt.ylabel('pixels', fontsize=20)
        plt.xlabel('pedestal std [DC]', fontsize=20)
        median = np.median(ped_std[select])
        rms = np.std(ped_std[select])
        label = f"run {run}, median {median:3.1f} +- {rms:3.1f}"
        #plt.title(f"Median {median:3.2f}, std {rms:3.2f}")
        if chan==0:
            plt.hist(ped_std[select],label=label, linewidth=2, histtype='step', bins=50,alpha=alpha,range=(100,300),fill=fill)
        else:
            plt.hist(ped_std[select], linewidth=2, label=label, histtype='step', bins=50, alpha=alpha,range=(17,30),fill=fill)
        plt.legend()
        if plot_dir != "none":
            #plt.savefig(f"{plot_dir}/signal_pedestal_{channel[chan]}.png", dpi = 300)
            plt.savefig(f"{plot_dir}/signal_pedestal_{channel[chan]}.png")

def plot_signal_pe(ff_data, cal_data, run=0,fill=True,tag=""):
    mask = cal_data.unusable_pixels

    for chan in np.arange(2):
        n_pe = cal_data.n_pe[chan]
        #dc_for_pe = 1. / cal_data.dc_to_pe[chan]
        gain_median = ff_data.relative_gain_median[chan]
        charge_median = ff_data.charge_median[chan] - ped_data.charge_median[chan]
        charge_mean = ff_data.charge_mean[chan]

        charge_std = ff_data.charge_std[chan]
        #print(np.where(mask == 1))
        # select good pixels
        select = np.logical_not(mask[chan])
        # select = mask[chan]
        # print(select)

        fig = plt.figure(chan, figsize=(12, 6))
        plt.subplot(121)
        # charge

        median = np.median(charge_median[select])
        rms = np.std(charge_median[select])

        #plt.title(f"Median {median:3.2f}, std {rms:5.0f}")
        label=f"{tag} run {run}, median {median:3.2f}+- {rms:3.2f}"
        plt.xlabel('charge (ADC)', fontsize=20)
        plt.ylabel('pixels', fontsize=20)

        #label = f"run {run}"
        if chan==0:
            plt.hist(charge_mean[select], label=label, linewidth=2, histtype='step',alpha=0.75, bins=50,range=(6000,7700),fill=fill)
        else:
            plt.hist(charge_mean[select], label=label, linewidth=2, histtype='step', alpha=0.75, bins=50,range=(300,500),fill=fill)
        plt.legend()


        plt.subplot(122)

        fig.suptitle(f"channel: {channel[chan]}", fontsize=25)
        n_pe = cal_data.n_pe[chan]

        # select good pixels
        select = np.logical_not(mask[chan])

        median = np.median(n_pe[select])
        rms = np.std(n_pe[select])

        plt.xlabel(f'pe', fontsize=20)
        plt.ylabel('pixels', fontsize=20)

        label = f"{tag} {channel[chan]} Median {median:3.2f}, std {rms:5.2f}"

        plt.hist(n_pe[select], label=label,  linewidth=2,range=(50, 110), bins=50, stacked=True, alpha=0.75,
                 fill=fill)

        plt.legend()

        if plot_dir != "none":
            plt.savefig(f"{plot_dir}/signal_and_pe_{channel[chan]}.png")

def plot_gain_charge(ff_data, cal_data, run=0,fill=True, tag=""):

    if fill==True:
        alpha=0.6
    else:
        alpha=0.75

    mask = cal_data.unusable_pixels

    for chan in np.arange(2):
        n_pe = cal_data.n_pe[chan]
        dc_for_pe = 1. / cal_data.dc_to_pe[chan]
        gain_median = ff_data.relative_gain_median[chan]
        charge_median = ff_data.charge_median[chan]
        charge_std = ff_data.charge_std[chan]
        #print(np.where(mask == 1))
        # select good pixels
        select = np.logical_not(mask[chan])
        # select = mask[chan]
        # print(select)

        fig = plt.figure(chan, figsize=(16, 6))
        fig.suptitle(f"channel: {channel[chan]}", fontsize=25)

        plt.subplot(121)
        median = np.median(dc_for_pe[select])
        rms = np.std(dc_for_pe[select])

        plt.xlabel('gain (DC/pe)', fontsize=20)
        plt.ylabel('pixels', fontsize=20)

        label = f"Run {run} {tag} Median {median:3.3f}, std {rms:5.3f}"
        if (chan == 0):
            plt.hist(dc_for_pe[select], label=label,linewidth=2,
        histtype='step', range=(60, 110), bins=50, stacked=True,
                     alpha=alpha,
                     fill=fill)
        else:
            plt.hist(dc_for_pe[select], label=label,linewidth=2,
                     histtype='step', range=(3, 7), bins=50, stacked=True,
                     alpha=alpha,
                     fill=fill)

        plt.legend()

        # relative gain
        plt.subplot(122)
        plt.ylabel('pixels', fontsize=20)
        plt.xlabel('charge', fontsize=20)

        median = np.median(charge_median[select])
        rms = np.std(charge_median[select])
        # plt.title(f"Relative gain {median:3.2f}, std {rms:5.2f}")
        label = f"Run {run} {tag} Median {median:3.1f}, std {rms:5.1f}"
        plt.hist(charge_median[select], label=label,
                 histtype='step',range=(0, 10000),bins=50, stacked=True,

                 alpha=alpha, linewidth=2,
                 fill=fill)

        plt.legend()
        if plot_dir != "none":
            plt.savefig(f"{plot_dir}/gain_charge_{channel[chan]}.png")


def plot_signal_pe_gain(ped_data, ff_data, cal_data, run=0,fill=True,tag=""):
    mask = cal_data.unusable_pixels
    alpha=0.75
    for chan in np.arange(2):

        dc_for_pe = 1. / cal_data.dc_to_pe[chan]

        charge_median = ff_data.charge_median[chan]
        ped_charge_median =ped_data.charge_median[chan]

        select = np.logical_not(mask[chan])

        fig = plt.figure(chan, figsize=(24, 6))

        plt.subplot(141)
        # charge

        median = np.median(charge_median[select])
        rms = np.std(charge_median[select])

        #plt.title(f"Median {median:3.2f}, std {rms:5.0f}")
        label=f"{tag} run {run}, median {median:3.1f}, std {rms:3.1f}"
        plt.xlabel('charge (ADC)', fontsize=20)
        plt.ylabel('pixels', fontsize=20)

        #label = f"run {run}"
        if chan==0:
            plt.hist(charge_median[select], label=label, linewidth=2, histtype='step',alpha=0.75, bins=50,range=(6000,7700),fill=fill)
        else:
            plt.hist(charge_median[select], label=label, linewidth=2, histtype='step', alpha=0.75, bins=50,range=(300,500),fill=fill)
        plt.legend(prop={'size':15})

        plt.subplot(142)
        # charge

        median = np.median(ped_charge_median[select])
        rms = np.std(ped_charge_median[select])

        #plt.title(f"Median {median:3.2f}, std {rms:5.0f}")
        label=f"{tag} run {run}, median {median:3.1f}, std {rms:3.1f}"
        plt.xlabel('pedestals (ADC)', fontsize=20)
        plt.ylabel('pixels', fontsize=20)

        #label = f"run {run}"
        if chan==0:
            plt.hist(ped_charge_median[select], label=label, linewidth=2, histtype='step',alpha=alpha, bins=50,range=(-50,50),fill=fill)
        else:
            plt.hist(ped_charge_median[select], label=label, linewidth=2, histtype='step', alpha=alpha, bins=50,range=(-50,50),fill=fill)
        plt.legend(prop={'size':15})


        plt.subplot(143)

        fig.suptitle(f"channel: {channel[chan]}", fontsize=25)
        n_pe = cal_data.n_pe[chan]

        # select good pixels
        select = np.logical_not(mask[chan])

        median = np.median(n_pe[select])
        rms = np.std(n_pe[select])

        plt.xlabel(f'pe', fontsize=20)
        plt.ylabel('pixels', fontsize=20)

        label = f"{tag} {channel[chan]} Median {median:3.2f}, std {rms:5.2f}"

        plt.hist(n_pe[select], label=label,  linewidth=2,range=(50, 110), bins=50, stacked=True, alpha=0.75,
                 fill=fill)

        plt.legend(prop={'size':15})
        plt.subplot(144)
        median = np.median(dc_for_pe[select])
        rms = np.std(dc_for_pe[select])

        plt.xlabel('gain (ADC/pe)', fontsize=20)
        plt.ylabel('pixels', fontsize=20)

        label = f"Median {median:3.2f}, std {rms:5.2f}"
        if (chan == 0):
            plt.hist(dc_for_pe[select], label=label,linewidth=2,
        histtype='step', range=(60, 110), bins=50, stacked=True,
                     alpha=alpha,
                     fill=fill)
        else:
            plt.hist(dc_for_pe[select], label=label,linewidth=2,
                     histtype='step', range=(3, 7), bins=50, stacked=True,
                     alpha=alpha,
                     fill=fill)

        plt.legend(prop={'size':15})

        if plot_dir != "none":
            plt.savefig(f"{plot_dir}/signal_and_pe_gain_{channel[chan]}.png")


def plot_only_charge(ff_data, cal_data, run=0,fill=True):

    if fill==True:
        alpha=0.6
    else:
        alpha=0.75

    mask = cal_data.unusable_pixels

    for chan in np.arange(2):
        #n_pe = cal_data.n_pe[chan]
        #dc_for_pe = 1. / cal_data.dc_to_pe[chan]
        gain_median = ff_data.relative_gain_median[chan]
        charge_median = ff_data.charge_median[chan]
        #charge_std = ff_data.charge_std[chan]
        #print(np.where(mask == 1))
        # select good pixels
        select = np.logical_not(mask[chan])
        # select = mask[chan]
        # print(select)

        fig = plt.figure(chan, figsize=(12, 6))

        fig.tight_layout( w_pad=2, h_pad=2.0, rect=[0, 0.00, 1, 0.95])

        fig.suptitle(f"channel: {channel[chan]}", fontsize=20)

        plt.subplot(1,2,1)

        median = np.median(charge_median[select])
        rms = np.std(charge_median[select])

        plt.xlabel('charge [ADC]', fontsize=20)
        plt.ylabel('pixels', fontsize=20)

        label = f"{run} median {median:5.0f} std {rms:5.0f}"
        if (chan == 0):
            plt.hist(charge_median[select], label=label,linewidth=2,
        histtype='step', range=(5000, 8000), bins=50, stacked=True,
                     alpha=alpha,
                     fill=fill)
        else:
            plt.hist(charge_median[select], label=label,linewidth=2,
                     histtype='step', range=(300, 500), bins=50, stacked=True,
                     alpha=alpha,
                     fill=fill)

        plt.legend(prop={'size':15})
        fig.tight_layout(rect=[0, 0.00, 1, 0.95])
        # relative gain
        plt.subplot(1,2,2)

        plt.ylabel('pixels', fontsize=20)
        plt.xlabel('relative charge', fontsize=20)

        median = np.median(gain_median[select])
        rms = np.std(gain_median[select])
        # plt.title(f"Relative gain {median:3.2f}, std {rms:5.2f}")
        label = f"{run} median {median:5.2f}, std {rms:5.3f}"

        plt.hist(gain_median[select], label=label,
                 histtype='step',range=(0.8, 1.2),bins=50, stacked=True,
                 alpha=alpha, linewidth=2,
                 fill=fill)

        plt.legend(prop={'size':15})
        fig.tight_layout(rect=[0, 0.00, 1, 0.95])
        if plot_dir != "none":
            plt.savefig(f"{plot_dir}/charge_{channel[chan]}.png")

