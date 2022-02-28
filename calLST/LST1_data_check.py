#!/usr/bin/env python3
"""
check LST1 data
"""
from ctapipe.utils.datasets import get_dataset_path
import numpy as np
import logging
from traitlets.config.loader import Config
import sys

# Change it with your lstchain directory
sys.path.insert(0,'/astro/users/cassol/soft/python/cta-lstchain')
from matplotlib import pyplot as plt
cmaps = [
    plt.cm.jet, plt.cm.winter, plt.cm.ocean, plt.cm.bone, plt.cm.gist_earth,
    plt.cm.hot, plt.cm.cool, plt.cm.coolwarm
]

from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from ctapipe.io import event_source
from ctapipe.io import EventSeeker
from ctapipe.image.extractor import *
from lstchain.calib.camera.r0 import LSTR0Corrections


#from lstchain.io.lsteventsource import LSTEventSource

''' triggerType // // // // // // // //
// bit 0: Mono
// bit 1: stereo
// bit 2: Calibration
// bit 3: Single Phe
// bit 4: Softrig(from the UCTS)
// bit 5: Pedestal
// bit 6: slow control
// bit 7: busy
// // // // // // // // // // // // // // /
'''

logging.basicConfig(level=logging.DEBUG)

fig = plt.figure(figsize=(12, 8))


tel_id = 0
n_chan = 2
geom = CameraGeometry.from_name("LSTCam-002")
#integrator = SimpleIntegrator()
config = Config({
    "LocalPeakWindowSum": {
        "window_shift": 6,
        "window_width": 12
    }
})
integrator = LocalPeakWindowSum(config=config)



def get_input():
    print("============================================")
    print("n or [enter]    - go to Next event")
    print("d               - Display the event")
    print("w               - Display the waveform")
    print("p               - Print all event data")
    print("i               - event Info")
    print("s               - save event image")
    print("q               - Quit")
    return input("Choice: ")


def get_pixel():
    return input("which pixel index? ")

def get_trigger_type(trigger_mask):

    trigger_name = ["Mono","Stereo","Calibration","SinglePhe",
             "SoftTrigFromUCTS","Pedestal","SlowControl", "Busy"]

    trigger_type = "Unknown"
    for bit in np.arange(1,9):
        if trigger_mask & bit != 0:
            trigger_type = trigger_name[bit]

    return trigger_type

def display_event(event):
    """an extremely inefficient display. It creates new instances of
    CameraDisplay for every event and every camera, and also new axes
    for each event. It's hacked, but it works
    """
    print("Displaying... please wait (this is an inefficient implementation)")
    global fig
    global geom
    global n_chan
    fig.clear()
    #trigger = get_trigger_type(event.lst.tel[0].evt.tib_masked_trigger)
    trigger = event.lst.tel[tel_id].evt.tib_masked_trigger
    plt.suptitle("EVENT {}, trigger type: {}".format(event.r0.event_id,trigger))

    disps = []


    signals, peakpos = integrator(event.r1.tel[tel_id].waveform)
    pix = np.arange(0, 1855)
    #high_charge=pix[signals[0] > 3000]
    #print(f"High signal pixels: {high_charge}")
    #print(f"High signal       : {signals[0,high_charge]}")

    for chan in range(n_chan):
        print("\t draw channel {}...".format(chan))
        ax = plt.subplot(2, 2, chan+1)
        disp = CameraDisplay(geom, ax=ax, title="channel {}".format(chan))
        disp.enable_pixel_picker()
        disp.axes.text(2.4, 0, 'Charge (ADC)', rotation=90)
        disp.pixels.set_antialiaseds(False)
        disp.autoupdate = False
        disp.image = signals[chan]
        disp.set_limits_percent(95)
        disp.add_colorbar()
        disp.cmap = plt.cm.coolwarm
        disps.append(disp)
        ax = plt.subplot(2, 2, chan + n_chan+1)
        disp = CameraDisplay(geom, ax=ax, title="")
        disp.enable_pixel_picker()
        disp.pixels.set_antialiaseds(False)
        disp.autoupdate = False
        disp.image = peakpos[chan]
        #disp.set_limits_percent(95)
        disp.set_limits_minmax(20, 40)
        disp.add_colorbar()
        disp.axes.text(2.4, 0, 'Time (ns)', rotation=90)
        disp.cmap = plt.cm.coolwarm
        disps.append(disp)



    return disps


def display_waveform(event, pix_rank):

    global fig
    id = int(pix_rank)
    fig.clear()
    plt.suptitle("EVENT {}, Pixel {}".format(event.r0.event_id, pix_rank))

    waveform=event.r1.tel[tel_id].waveform
    signals, peakpos = integrator(event.r1.tel[tel_id].waveform)


    samples=np.arange(0,36)
    for chan in range(n_chan):

        time=peakpos[chan,id]
        peak=waveform[chan,id,int(time)]
        start = int(time - integrator.window_shift)
        if start<0:
            start=0
        stop = int(start + integrator.window_width)
        if stop>35:
            stop=36
        used_samples = np.arange(start, stop)
        used=waveform[chan, id,start:stop]


        print("\t draw channel {}...".format(chan))
        ax = plt.subplot(1, 2, chan + 1)
        plt.xlabel('samples [ns]')
        plt.ylabel('ADC')
        plt.title(f"channel {chan}")
        ax.plot(time,peak,'rs')
        ax.plot(samples,waveform[chan,id,],'b')
        ax.plot(used_samples,used, 'r')
        plt.text(time + 2, peak + 10, f'charge: {int(signals[chan,id])} \n time:   {int(peakpos[chan,id])} ns')


if __name__ == '__main__':

    if len(sys.argv) > 1:
        filename = sys.argv.pop(1)
    else:
        filename = get_dataset_path("/ctadata/LSTCAM/ZFITS/20181217/LST-1.1.Run00078.0000.fits.fz")

    plt.style.use("ggplot")
    plt.show(block=False)
    pedestal_path= sys.argv.pop(1)
    tel_id=1
    # loop over events and display menu at each event:
    reader = event_source(input_url=filename)
    print("--> NUMBER OF FILES", reader.multi_file.num_inputs())
    seeker = EventSeeker(reader)
    r0calib = LSTR0Corrections(
        pedestal_path=pedestal_path,
        r1_sample_start=2, r1_sample_end=38)

    for i, event in enumerate(reader):

        all_modules_on=np.sum(event.lst.tel[tel_id].evt.module_status)
        #
        if(all_modules_on > 1):
            print(f"Event {event.lst.tel[tel_id].evt.event_id}, num modules on: {all_modules_on}")
        filename = "event_{0:04d}.png".format(event.r0.event_id)

        r0calib.calibrate(event)
        #event.r1.tel[tel_id].waveform = subtract_baseline(event.r1.tel[tel_id].waveform, 0, 5)
        # cleaned = event.r0.tel[tel_id].waveform[:, :, 2:38]
        #signals, peakpos = integrator.extract(event.r1.tel[tel_id].waveform)

        #if(signals[0,0]<1500 or signals[1,0]<300):
        #    continue
        if event.r0.event_id< 38560:
            continue
        while True:
            response = get_input()
            if response.startswith("d"):
                disps = display_event(event)
                filename = "event_{0:04d}.png".format(event.r0.event_id)
                plt.pause(0.1)
            if response.startswith("w"):
                n_rank = get_pixel()
                filename = "event_{0:04d}_pixel_{1}.png".format(event.r0.event_id, n_rank)
                display_waveform(event,n_rank)
                plt.pause(0.1)
            elif response.startswith("p"):
                print("--event-------------------")
                print(event)
                print("--event.r0---------------")
                print(event.r0)
                #print("--event.mc----------------")
                #print(event.mc)
                print("--event.r0.tel-----------")
                for teldata in event.r0.tel.values():
                    print(teldata)
            elif response == "" or response.startswith("n"):
                break
#            if response.startswith("e"):
#                n_eve = get_event()
#                if n_eve > event.r0.event_id:

            elif response.startswith('i'):
                subarray = event.inst.subarray
                for tel_id in sorted(event.r0.tel):
                    for chan in event.r1.tel[tel_id].waveform:
                        npix = len(subarray.tel[tel_id].camera.pix_x)
                        nsamp = event.r1.tel[tel_id].waveform.shape[2]
                        print(
                            "CT{:4d} ch{} pixels,samples:{}"
                            .format(tel_id, chan, npix, nsamp)
                        )
            elif response.startswith('s'):

                print("Saving to", filename)
                plt.savefig(filename)

            elif response.startswith('q'):
                break

        if response.startswith('q'):
            break
