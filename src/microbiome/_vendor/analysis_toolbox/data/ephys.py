# Vendored from analysis_toolbox (data/ephys.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


import numpy as np
import matplotlib.pyplot as plt
from microbiome._vendor.analysis_toolbox.utils import now_str
import os


channel_labels = [
    # Channel label                      Channel number
    "Left ephys",  # 1
    "Right ephys",  # 2
    "Camera trigger",  # 3
    "Observed stimulus velocity",  # 4
    "Gain",  # 5
    "Current epoch",  # 6
    "Trial number",  # 7
    "Stimulus brightness",  # 8
    "Command stimulus velocity",  # 9
    "Stimulus ID",  # 10
]


def plot_all_channels(ep, rate=6000, ch_labels=[""] * 10):

    ep_fig, ep_ax = plt.subplots(
        ep.shape[0], 1, figsize=(9.5, 1.2 * ep.shape[0]), sharex=True
    )

    for channel in range(ep.shape[0]):
        ep_ax[channel].plot(
            np.linspace(0, ep.shape[1] / rate, num=ep.shape[1]), ep[channel]
        )
        try:
            ep_ax[channel].set_title(ch_labels[channel])
        except:
            continue

    ep_ax[-1].set_xlabel("Time [s]")
    ep_fig.tight_layout()

    return ep_fig, ep_ax


def open_ephys_metadata(xml):

    import xml.etree.ElementTree as et
    import collections
    import pandas as pd
    from microbiome._vendor.analysis_toolbox.utils import tryfloat

    tree = et.parse(xml)

    root = tree.getroot()

    StimConds = []
    for r in list(root):

        StimCond = collections.OrderedDict()

        for e in r:
            StimCond[e.tag] = tryfloat(e.text)

        StimConds.append(StimCond)

    columns = list(StimConds[0].keys())
    columns.remove("epoch")

    index = [s["epoch"] for s in StimConds]

    return pd.DataFrame(StimConds, index=index, columns=columns)


def kickAssSwimDetect01(ch1, ch2, thre):

    import numpy as np

    print("processing channel data\n")
    ker = np.exp(-(np.arange(-60, 61, 1, "f4") ** 2) / (2 * (20**2)))
    ker = ker / ker.sum()

    smch1 = np.convolve(ch1, ker, "same")
    pow1 = (ch1 - smch1) ** 2
    fltCh1 = np.convolve(pow1, ker, "same")

    smch2 = np.convolve(ch2, ker, "same")
    pow2 = (ch2 - smch2) ** 2
    fltCh2 = np.convolve(pow2, ker, "same")

    aa1 = np.diff(fltCh1)
    peaksT1 = (aa1[0:-1] > 0) * (aa1[1:] < 0)
    peaksIndT1 = np.argwhere(peaksT1 > 0).squeeze()

    aa2 = np.diff(fltCh2)
    peaksT2 = (aa2[0:-1] > 0) * (aa2[1:] < 0)
    peaksIndT2 = np.argwhere(peaksT2 > 0).squeeze()

    x_ = np.arange(0, 0.10001, 0.00001)
    th1 = np.zeros(
        fltCh1.size,
    )
    th2 = np.zeros(
        fltCh2.size,
    )
    back1 = np.zeros(
        fltCh1.size,
    )
    back2 = np.zeros(
        fltCh2.size,
    )

    if len(ch1) < 360000:
        d_ = 6000 * 1
    else:
        d_ = 6000 * 60
        # % 5 minutes threshold window

    last_i = 0

    for i in np.arange(0, fltCh1.size - d_, d_):
        peaksIndT1_ = np.argwhere(peaksT1[0 : (i + d_)] > 0).squeeze()
        peaksIndT2_ = np.argwhere(peaksT2[0 : (i + d_)] > 0).squeeze()

        a1, _ = np.histogram(fltCh1[peaksIndT1_], x_)
        a2, _ = np.histogram(fltCh2[peaksIndT2_], x_)

        a1 = a1.astype("f4")
        a2 = a2.astype("f4")

        mx1 = (np.argwhere(a1 == a1.max())).min()
        mn1_ind = np.argwhere(a1[0:mx1] < (a1[mx1] / 200))
        if mn1_ind.size > 0:
            mn1 = mn1_ind.max()
        else:
            mn1 = 0
        mx2 = (np.argwhere(a2 == a2.max())).min()
        mn2_ind = np.argwhere(a2[0:mx2] < (a2[mx2] / 200))
        if mn2_ind.size > 0:
            mn2 = mn2_ind.max()
        else:
            mn2 = 0

        th1[i : (i + d_ + 1)] = x_[mx1] + thre * (x_[mx1] - x_[mn1])
        th2[i : (i + d_ + 1)] = x_[mx2] + thre * (x_[mx2] - x_[mn2])
        back1[i : (i + d_ + 1)] = x_[mx1]
        back2[i : (i + d_ + 1)] = x_[mx2]

        last_i = i

    th1[(last_i + d_ + 1) :] = th1[last_i + d_]
    th2[(last_i + d_ + 1) :] = th2[last_i + d_]
    back1[(last_i + d_ + 1) :] = back1[last_i + d_]
    back2[(last_i + d_ + 1) :] = back2[last_i + d_]

    print("\nAssigning bursts and swims\n")

    burstIndT1 = peaksIndT1[np.argwhere((fltCh1 - th1)[peaksIndT1] > 0).squeeze()]
    burstT1 = np.zeros(fltCh1.size)
    burstT1[burstIndT1] = 1

    burstIndT2 = peaksIndT2[np.argwhere((fltCh2 - th2)[peaksIndT2] > 0).squeeze()]
    burstT2 = np.zeros(fltCh1.size)
    burstT2[burstIndT2] = 1

    burstBothT = np.zeros(fltCh1.size)
    burstBothT[burstIndT1] = 1
    burstBothT[burstIndT2] = 2

    burstBothIndT = np.argwhere(burstBothT > 0).squeeze()

    interSwims = np.diff(burstBothIndT)

    swimEndIndB = np.argwhere(interSwims > 600).squeeze()

    swimEndIndB = np.append(swimEndIndB, burstBothIndT.size - 1)

    swimStartIndB = 0
    swimStartIndB = np.append(swimStartIndB, swimEndIndB[:-1] + 1)
    nonSuperShort = np.argwhere(swimEndIndB != swimStartIndB).squeeze()

    swimEndIndB = swimEndIndB[nonSuperShort]
    swimStartIndB = swimStartIndB[nonSuperShort]

    # swimStartIndB is an index for burstBothIndT
    # burstBothIndT is an idex for time

    swimStartIndT = burstBothIndT[swimStartIndB]
    swimStartT = np.zeros(fltCh1.size)
    swimStartT[swimStartIndT] = 1

    swimEndIndT = burstBothIndT[swimEndIndB]
    swimEndT = np.zeros(fltCh1.size)
    swimEndT[swimEndIndT] = 1

    swimdata = dict()
    # channel 1 swim power
    swimdata["fltCh1"] = fltCh1.astype("f4")
    # channel 2 swim power
    swimdata["fltCh2"] = fltCh2.astype("f4")
    # TODO: not sure what this is
    swimdata["back1"] = back1.astype("f4")
    # TODO: not sure what this is
    swimdata["back2"] = back2.astype("f4")
    # TODO: not sure what this is
    swimdata["th1"] = th1.astype("f4")
    # TODO: not sure what this is
    swimdata["th2"] = th2.astype("f4")

    # Bursts
    #  trace showing if the bursts is in 1 or 2 channels
    swimdata["burstBothT"] = burstBothT
    #  indices of the bursts from both channels
    swimdata["burstBothIndT"] = burstBothIndT
    # indices of the bursts from channel 1
    swimdata["burstIndT1"] = burstIndT1
    # indices of the bursts from channel 2
    swimdata["burstIndT2"] = burstIndT2

    # Swims
    # indices of the swim starts in terms of burst indices
    swimdata["swimStartIndB"] = swimStartIndB
    # indices of the swim ends in terms of burst indices
    swimdata["swimEndIndB"] = swimEndIndB
    # indices of the swim starts in terms of time indices
    swimdata["swimStartIndT"] = swimStartIndT
    # indices of the swim ends in terms of time indices
    swimdata["swimEndIndT"] = swimEndIndT
    # boolean array showing swim starts
    swimdata["swimStartT"] = swimStartT
    # boolean array showing swim ends
    swimdata["swimEndT"] = swimEndT

    return swimdata


class ephys(object):
    def __init__(self, ephys_file, channel_labels=[], metadata="", debug=False):
        """Load 10 channel binary file without any further processing.

        ephys_file [str]: path of the ephys file with file extensions
                          of either .10chFlt for raw data or .npy for
                          processed data

        channel_labels [list]: list of strings containing labels for
                               the respective channels

        metadata [str]: directory containing the metadata file. While
                        initializing the ephys data doesn't require a
                        metadata file, methods processing the data might.

        debug [bool]: option to be verbose

        """

        self.ephys_file = ephys_file
        self.path = "/".join(ephys_file.split("/")[:-1]) + "/"
        self.expt_name = ephys_file.split("/")[-1].split(".")[0]
        self.expt_fish = self.path.split("/")[-4]
        self.expt_date = self.path.split("/")[-5]
        self.metadata_dir = metadata
        self.debug = debug

        # load metadata
        from microbiome._vendor.analysis_toolbox.dataorg.dataset import load_metadata

        if self.metadata_dir:
            self.metadata, self.metadata_pd = load_metadata(self.metadata_dir)

        if self.metadata_dir:
            self.savepath = self.metadata_dir + "analysis/"
            self.procpath = self.metadata_dir + "processed/"
        else:
            self.savepath = self.path.replace("ephys/", "analysis/")
            self.procpath = self.path.replace("ephys/", "processed/")

        print("\nOpening ephys data")
        print("==================")

        # load processed ephys file using numpy
        if ephys_file.split(".")[-1] == "npy":
            print("Post-processed npy file detected.")
            self.ep = np.load(ephys_file)
            self.num_ch = self.ep.shape[0]
            print(ephys_file)

        # load raw ephys file using Davis' function
        elif ephys_file.split(".")[-1][2:] == "chFlt":
            print("Raw ephys file detected. \n" "Raw ephys files might be unaligned...")

            try:
                from fish.ephys.ephys import load
            except ImportError:
                import sys
                import os

                sys.path.append(os.environ["USERPROFILE"] + r"\fish")

            from fish.ephys.ephys import load

            self.num_ch = int(ephys_file.split(".")[-1][:2])
            self.ep = load(ephys_file, num_channels=self.num_ch)

        # make sure that every channel gets a label
        if channel_labels:  # if channel_labels list is not empty
            print(f"Number of channels in ephys file: {self.ep.shape[0]}")
            print(f"Number of channel labels: {len(channel_labels)}")
            assert self.ep.shape[0] == len(channel_labels)
            self.ch_labels = channel_labels

        # extract channels and compute imaging times
        self.extract_ephys()
        self.define_ephys()

        print("Electrophysiological data loaded!\n")
        print(
            "========================================================================\n"
        )

    def load_ephys_metadata(self):

        from microbiome._vendor.analysis_toolbox.data.ephys import open_ephys_metadata

        _, aquaVR_fileext = os.path.splitext(self.ephys_file)
        aquaVR_ver = self.ephys_file.split("_")[-1].replace(f"{aquaVR_fileext}", "")
        ephys_metadata_path = self.ephys_file.replace(
            f"{aquaVR_ver}{aquaVR_fileext}", f"end_{aquaVR_ver}.xml"
        )

        if os.path.exists(ephys_metadata_path):
            self.ephys_metadata = open_ephys_metadata(ephys_metadata_path)

    def replace_ephys(self, ep):

        self.ep = ep

        self.extract_ephys()
        self.define_ephys()

    def extract_ephys(self):

        for ch in range(self.num_ch):
            setattr(self, f"channel{ch+1}", self.ep[ch])

        self.rate = 6000
        self.t = self.ep.shape[1]
        self.t_s = np.linspace(0, self.t / self.rate, self.t)

    def define_ephys(self):

        # legacy names
        self.leftch = self.channel1
        self.rightch = self.channel2
        self.im_times = self.channel3

        # define channels based on channel labels
        if hasattr(self, "ch_labels"):
            for ch, ch_label in enumerate(self.ch_labels):

                label = ch_label.lower().replace(" ", "_")
                setattr(self, label, getattr(self, f"channel{ch+1}"))

    def extract_twophoton_image_times(self):

        self.d_imtimes = self.find_dimtimes()  # approx first derivative of camtrig
        self.im_starts, self.im_stops = self.find_imagetimes()

        # find the start and stop times
        self.imstart_times = np.where(self.im_starts)[0]
        self.imstop_times = np.where(self.im_stops)[0]

        self.plane_starts = self.im_starts
        self.planestart_times = np.copy(self.imstart_times)

    def extract_spim_image_times(self):

        self.d_imtimes = self.find_dimtimes()  # approx first derivative of camtrig
        self.im_starts, self.im_stops = self.find_imagetimes()

        # find the start and stop times
        self.imstart_times = np.where(self.im_starts)[0]
        self.imstop_times = np.where(self.im_stops)[0]

        (
            self.imt_diff,
            self.stack_plane_starts,
            self.stack_starts,
            self.plane_starts,
        ) = self.find_acq_stt()

        self.stackplanestart_times = np.where(self.stack_plane_starts)[0]

        self.stackstart_times = np.where(self.stack_starts)[0]
        self.planestart_times = np.where(self.plane_starts)[0]

        from microbiome._vendor.analysis_toolbox.math_helper import step

        self.stack_num = step(self.stack_starts, val_start=-1).astype("int")
        self.plane_num = step(self.plane_starts, val_start=-1).astype("int")

    def plot_all_channels(self):

        try:
            ep_fig, ep_ax = plot_all_channels(self.ep, self.rate, self.ch_labels)
        except:
            ep_fig, ep_ax = plot_all_channels(self.ep, self.rate)

        return ep_fig, ep_ax

    def find_dimtimes(self):
        """
        Find the first derivative of channel 3 (i.e. im_times), the
        camera trigger.
        """

        imt_d = self.im_times[1:] - self.im_times[0:-1]
        imt_d = np.insert(imt_d, 0, 0)
        if self.im_times[0] > 1:
            imt_d[2] = np.abs(imt_d[np.nonzero(np.round(imt_d))[0][0]])

        return imt_d

    def find_acq_stt(self):
        """Find acquisition structure

        Find where the "stacks" and "planes" were taken. "Stacks"
        does not necessarily correspond to actual brain stacks. When
        mode is set to 'HS Single Plane', each "stack" consists of 50
        planes, but only of a single slice of the brain.

        """

        # Amplitude-based detection: not accurate -- not every pulse
        # is a delta-dirac wave. There are intermediate values and
        # square waves (i.e. staying in an ON state for more than an
        # instanteous amount of time.

        d_stack = 3.6
        d_plane_stack = 0.5

        stack_starts_ = self.d_imtimes > d_stack
        stack_plane_starts_ = self.d_imtimes > d_plane_stack

        nplanes = int(self.metadata_pd.loc["dimensions", "value"].split("x")[-1])
        inferred_stack_starts_times = np.where(stack_plane_starts_)[0][
            :: nplanes + 1
        ]  # more accurate because lower threshold
        supposed_stack_starts_times = np.where(stack_starts_)[0]

        # check if the very first one is congruent
        if inferred_stack_starts_times[0] < supposed_stack_starts_times[0]:
            stack_starts_[inferred_stack_starts_times[0]] = True

        # interval-based detection
        # replace d_imtimes with actual intervals
        imt_diff = np.copy(self.d_imtimes[1:])
        imt_diff = np.insert(imt_diff, 0, 0)
        imt_diff[imt_diff > d_plane_stack] = np.insert(
            np.diff(np.where(stack_plane_starts_)[0]), 0, 0
        )

        # calculate threshold to discern between stacks and images
        unique, counts = np.unique(np.diff(self.imstart_times), return_counts=True)
        print(f"Intervals: {unique}\nCounts: {counts}")

        hs_single_plane = False
        if "hs_single_plane" in self.metadata_pd.index:
            if self.metadata_pd.loc["hs_single_plane"].value == "T":
                hs_single_plane = True

        if hs_single_plane:

            print("Assuming high speed single plane...")

            # For some reason, the intervals in imt_diff don't
            # correspond to the intervals in
            # np.diff(self.imstart_times) so use imt_diff itself to
            # compute the histogram of intervals
            unique, counts = np.unique(imt_diff, return_counts=True)
            print(f"Recomputed intervals: {unique}\nCounts: {counts}")

            # Assuming that the last two interval values of a set of
            # ascending interval values correspond to that of the
            # inter-plane interval, then the next biggest can be
            # used as a threshold to detect these events
            plane_thresh = unique[-3]

            # Even though stacks are still acquired (with 50 planes),
            # this is not a brain stack. Only plane_starts are relevant
            plane_starts = imt_diff > plane_thresh
            stack_plane_starts = np.zeros(plane_starts.shape).astype("bool")
            stack_starts = np.zeros(plane_starts.shape).astype("bool")

        else:
            print("Assuming high speed stack...")
            # Find weighted average
            # The following works for monodirectional stack, but since
            # there is a second interval in between the up and down motion
            # of the stack, that is of a higher value than the plane interval
            # and lower than the stack interval, we cannot simply use the
            # value of the plane interval as the threshold.
            # threshold = np.average(unique, weights=counts)

            # The intervals should be the same but not necessary. This
            # hasn't given me any problems so far for multiplane stacks
            unique_, counts_ = np.unique(imt_diff, return_counts=True)
            unique_round_ = np.floor(unique_).astype("int")
            positive_unique_round_ = unique_round_[unique_round_ > 0]
            positive_counts_ = counts_[unique_round_ > 0]
            print(
                f"Recomputed itervals :\n{positive_unique_round_}\nCounts: {positive_counts_}"
            )

            # check that unique is in positive_unique_round_
            d_unique = -2
            unique_test = unique.copy()
            print(f"Trying unique: {unique_test}...")

            while not np.all(np.in1d(unique_test, positive_unique_round_)):
                if d_unique < 2:
                    unique_test = unique.copy() + d_unique
                    print(f"Trying unique: {unique_test}...")
                    d_unique += 1
                else:
                    print("Ran out of d_unique to try...")
                    unique = positive_unique_round_[positive_unique_round_ > 1]
                    counts = positive_counts_[positive_unique_round_ > 1]
                    break

            else:

                print(f"Intervals used: {unique}")
                unique = unique_test

            # Assuming that the last two interval values of a set of
            # ascending interval values correspond to that of the last-
            # plane-to-stack-onset interval, and that that interval is
            # larger than the inter-plane interval, then use the next
            # biggest interval as the threshold.

            print(f"Intervals: {unique}\nCounts: {counts}")
            counts = counts[unique < 100]
            unique = unique[unique < 100]
            stack_thresh = unique[np.argmax(counts)]  # was unique[-3]

            # interval is typically the first or second highest in count
            # however, you want to select the bigger interval
            # check if the bigger interval exist in the unique counts
            if np.isin(stack_thresh + 1, unique):
                stack_thresh = stack_thresh + 1

            print(f"Stack interval threshold: {stack_thresh}")

            # find stack starts using the intervals
            stack_starts = imt_diff > stack_thresh

            # since the first stack isn't preceeded by any prior acquisition,
            # use the amplitude of the pulses (as measured by the first
            # derivative, to indicate where the first stack occurred
            # Why didn't JUST use this as the indicator of stack start?
            # Probably because it's not always 3.6? Definitely also bad that
            # this value is hard-coded
            stack_starts[np.where(stack_starts_)[0][0]] = True

            # Using the same logic, the next two largest intervals should
            # be the interplane intervals
            plane_thresh = unique[0]
            stack_plane_starts = imt_diff >= plane_thresh

            # stack_plane_starts = np.logical_and(imt_diff <= stack_thresh,
            #                                     imt_diff >= plane_thresh)

            # We know where are the stack onsets, we know where are the
            # stack & plane onsets; their difference would be the plane
            # onsets. Not all the stacks are detected in `stack_plane_starts`,
            # but that's fine -- since it should contain all the plane
            # starts and that's what we are after. Remove any stack
            # starts that it might have captured.
            plane_starts = np.zeros(stack_starts.shape).astype("bool")
            plane_start_times = np.setdiff1d(
                np.where(stack_plane_starts)[0], np.where(stack_starts)[0]
            )
            plane_starts[plane_start_times] = True

        return imt_diff, stack_plane_starts, stack_starts, plane_starts

    @staticmethod
    def find_starts_stops(bool_arr):
        """
        Convolution helps with detection, but changes temporal aspects
        so it's not ideal...
        """

        # convolve/smooth imaging pulses timeseries to remove noise
        smoothed = np.convolve(np.ones(6), bool_arr, mode="same")

        # find non-zero time points of positive jumps
        rounded = smoothed.astype(int)  # round times
        starts = (rounded[1:] > 0) * (rounded[:-1] == 0)
        starts = np.insert(starts, 0, 0)

        stops = (rounded[1:] == 0) * (rounded[:-1] > 0)
        stops = np.insert(stops, 0, 0)

        return rounded, starts, stops

    def find_imagetimes(self):
        """
        This is not actually the real image times, since smoothing the
        camera trigger and then performing a detection of that actually
        induces some artefact in the first step. The smoothing actually
        overstates the earliness of the camera trigger since both ends
        of the pulse are smoothed (i.e. actual camera trigger happens
        after the computed camera trigger).

        This is however, still useful for finding the intervals between
        planes acquired and from the last plane of a stack to the start
        of the next stack.
        """

        imt_round, imt_starts, imt_stops = self.find_starts_stops(self.im_times)

        if imt_round[0] != 0:
            # use the first instance to find the first spike
            imt_fstart = (imt_round[1:] > 0) * (imt_round[:-1] == imt_round[0])
            imt_starts[np.where(imt_fstart)[0][0] + 1] = True

        return imt_starts, imt_stops

    def find_swims(
        self,
        channel,
        scaling=2.1,
        show_plot=False,
        real_time=False,
        enumerate_swims=False,
    ):

        from fish.ephys.ephys import load, windowed_variance, estimate_swims

        swim_power = windowed_variance(channel)[0]
        swim_startstop = estimate_swims(swim_power, scaling=scaling)

        from microbiome._vendor.analysis_toolbox.preprocess_helper import binarize_swimming

        swimming, swimstart_idx, swimstop_idx = binarize_swimming(
            swim_startstop[0], swim_startstop[1]
        )
        swim_power_clipped = swim_power * swimming

        print("Number of swim starts: %i" % swimstart_idx.shape)
        print("Number of swim stops: %i" % swimstop_idx.shape)

        if show_plot:

            if real_time:
                timepoints = np.linspace(0, len(channel) / self.rate, num=len(channel))
            else:
                timepoints = np.linspace(0, len(channel), num=len(channel))

            # plot swim times
            swim_fig, swim_ax = plt.subplots(5, 1, figsize=(11.5, 10), sharex=True)
            swim_ax[0].plot(timepoints, channel)
            swim_ax[0].set_title("Motor neuron recording")
            swim_ax[0].set_ylim([channel.mean() - 0.2, channel.mean() + 0.2])

            swim_ax[1].plot(timepoints, swim_power)
            swim_ax[1].set_title("Swim power")
            swim_ax[1].set_ylim(
                [swim_power.min() - 0.0002, np.percentile(swim_power, 99.99)]
            )

            swim_ax[2].plot(timepoints, swim_power_clipped)
            swim_ax[2].set_title("Swim power clipped")
            swim_ax[2].set_ylim(
                [
                    swim_power_clipped.min() - 0.0002,
                    np.percentile(swim_power_clipped, 99.99),
                ]
            )

            swim_ax[3].plot(timepoints, swim_startstop[0])
            swim_ax[3].set_title("Swim starts")
            if enumerate_swims:
                for idx_i, idx in enumerate(swimstart_idx):
                    swim_ax[3].text(timepoints[idx], 1.0, str(idx_i))

            swim_ax[4].plot(timepoints, swim_startstop[1])
            swim_ax[4].set_title("Swim stops")
            if enumerate_swims:
                for idx_i, idx in enumerate(swimstop_idx):
                    swim_ax[4].text(timepoints[idx], 1.0, str(idx_i))

            # make pretty
            swim_ax[0].set_xlim([0, timepoints[-1]])

            if real_time:
                swim_ax[0].set_xlabel("Time (s)")
            else:
                swim_ax[0].set_xlabel("Time points")

            plt.tight_layout()

            return (
                swim_power,
                swim_power_clipped,
                swim_startstop[0],
                swim_startstop[1],
                swimstart_idx,
                swimstop_idx,
                swimming,
                swim_fig,
                swim_ax,
            )

        else:
            return (
                swim_power,
                swim_power_clipped,
                swim_startstop[0],
                swim_startstop[1],
                swimstart_idx,
                swimstop_idx,
                swimming,
            )

    def test_swim_thresh(
        self, channel, detect_threshs, duration, num_chunks, save_name=None
    ):

        search_dur = np.clip(
            duration, 61, a_max=None
        )  # at least 60 seconds because the default window to estimate_threshold is 60 seconds

        search_starts = []
        search_stops = []

        for chunk in range(num_chunks):

            search_starts.append(
                np.random.randint(np.floor(len(channel) - (search_dur * self.rate)))
            )
            search_stops.append(search_starts[-1] + (search_dur * self.rate))

        for detect_thresh in detect_threshs:

            swim_fig, swim_ax = plt.subplots(
                3, num_chunks, figsize=(6 * num_chunks, 4), squeeze=False
            )

            for chunk, (search_start, search_stop) in enumerate(
                zip(search_starts, search_stops)
            ):

                search_ep = channel[search_start:search_stop]

                swim_power, _, _, _, _, swimming = self.find_swims(
                    search_ep, scaling=detect_thresh, show_plot=False
                )

                timepoints = np.arange(search_start, search_stop) / self.rate

                swim_ax[0, chunk].plot(timepoints, search_ep)
                swim_ax[0, chunk].set_title("Motor neuron recording")
                swim_ax[0, chunk].set_ylim(
                    [search_ep.mean() - 0.2, search_ep.mean() + 0.2]
                )
                swim_ax[0, chunk].set_xlabel("Time [s]")

                swim_ax[1, chunk].plot(timepoints, swim_power)
                swim_ax[1, chunk].set_title("Swim power")
                swim_ax[1, chunk].set_ylim(
                    [swim_power.mean() - 0.0002, np.percentile(swim_power, 99.9)]
                )
                swim_ax[1, chunk].set_xlabel("Time [s]")
                swim_ax[0, chunk].get_shared_x_axes().join(
                    swim_ax[0, chunk], swim_ax[1, chunk]
                )

                swim_ax[2, chunk].plot(timepoints, swimming)
                swim_ax[2, chunk].set_title("Swimming")
                swim_ax[2, chunk].set_xlabel("Time [s]")
                swim_ax[2, chunk].get_shared_x_axes().join(
                    swim_ax[0, chunk], swim_ax[2, chunk]
                )

                swim_ax[0, chunk].set_xlim(
                    [
                        np.median(timepoints) - duration // 2,
                        np.median(timepoints) + duration // 2,
                    ]
                )
                # swim_ax[1,chunk].set_xlim([np.median(timepoints) - duration//2, np.median(timepoints) + duration // 2])
                # swim_ax[2,chunk].set_xlim([np.median(timepoints) - duration//2, np.median(timepoints) + duration // 2])

            swim_fig.tight_layout()
            swim_fig.subplots_adjust(top=0.85)
            swim_fig.suptitle("Swim detection threshold: %.2f" % detect_thresh)

            if save_name:
                swim_fig.savefig(
                    save_name + f"_thresh-{detect_thresh}_" + now_str() + ".png"
                )

    def find_flashes(self, channel, flash_baseline=None, show_plot=False):

        if flash_baseline == None:
            flash_baseline = np.bincount(channel.astype("int")).argmax()
            print("Flash baseline not set; using: %i" % flash_baseline)

        else:
            print("Flash baseline set to %i" % flash_baseline)

        # find non-zero time points of positive jumps
        flash_jumps = (channel[1:] > flash_baseline) * (channel[:-1] == flash_baseline)
        # shift the array to the right by 1
        flash_jumps = np.insert(flash_jumps, 0, 0)

        self.flash_onset = flash_jumps

        if show_plot:
            flash_fig, flash_ax = plt.subplots(
                2, 1, figsize=(9, 4), sharex=True, facecolor="w"
            )

            flash_ax[0].plot(channel)
            flash_ax[0].set_title("Color intensity")

            flash_ax[1].plot(flash_jumps)
            flash_ax[1].set_title("Flashes")

            flash_ax[0].set_xlim(0, len(channel))
            plt.tight_layout()

    def plot_imagetimes(self, collapse=True):

        if collapse:
            imt_fig, imt_ax1 = plt.subplots(figsize=(9, 2.5), facecolor="w")
            imt_ax = [imt_ax1]
            imt_ax[0].plot(self.im_times, color="k")

            imt_ax2 = plt.twinx(imt_ax[0])
            imt_ax.append(imt_ax2)
            imt_ax[1].plot(self.im_starts, label="camtrig onset")
            imt_ax[1].plot(self.im_stops, label="camtrig offset")

            imt_ax[1].legend()

        else:

            imt_fig, imt_ax = plt.subplots(
                3, 1, figsize=(9, 6), sharex=True, facecolor="w"
            )

            imt_ax[0].plot(self.im_times)
            imt_ax[0].set_title("Imaging times")

            imt_ax[1].plot(self.im_starts)
            imt_ax[1].set_title("Imaging start times")

            imt_ax[2].plot(self.im_stops)
            imt_ax[2].set_title("Imaging stop times")

        imt_fig.tight_layout()
        imt_fig.suptitle("NOT EXACTLY THE ONSET AND OFFSET TIMES")

        try:
            imt_ax[0].set_xlim(0, np.where(self.stack_starts)[0][1])
        except:
            imt_ax[0].set_xlim(0, np.where(self.im_starts)[0][50])

        return imt_fig, imt_ax

    def plot_camtrig_onsets(self, onset_type=None, xlim_pos="start", collapse=True):

        if collapse:

            ss_fig, ss_ax1 = plt.subplots(figsize=(9, 2.5), facecolor="w")
            ss_ax = [ss_ax1]
            ss_ax.append(plt.twinx(ss_ax1))

            ss_ax[0].plot(self.im_times, label="Imaging times", color="k")
            ss_ax[0].plot(self.d_imtimes, label="d(Imaging times)", color="tab:blue")

            if "stacks" in onset_type:
                ss_ax[1].plot(
                    self.stack_starts, label="Stack starts", color="tab:orange"
                )
            if "planes" in onset_type:
                ss_ax[1].plot(self.plane_starts, label="Plane starts", color="tab:red")
            if "both" in onset_type:
                ss_ax[1].plot(
                    self.stack_plane_starts,
                    label="Stack+Plane starts",
                    color="tab:olive",
                )

            ss_ax[0].legend()
            ss_ax[1].legend()

        else:

            ss_fig, ss_ax = plt.subplots(
                3, 1, figsize=(9, 6), sharex=True, facecolor="w"
            )

            ss_ax[0].plot(self.im_times)
            ss_ax[0].set_title("Imaging times")

            ss_ax[1].plot(self.d_imtimes)
            ss_ax[1].set_title("Derivative of imaging times")

            if "stacks" in onset_type:
                ss_ax[2].plot(self.stack_starts)
                ss_ax[2].set_title("Stack starts")
            if "planes" in onset_type:
                ss_ax[2].plot(self.plane_starts)
                ss_ax[2].set_title("Plane starts")
            if "both" in onset_type:
                ss_ax[2].plot(self.stack_plane_starts)
                ss_ax[2].set_title("Stack+Plane starts")

        ss_fig.tight_layout()

        if xlim_pos == "start":
            try:
                ss_ax[0].set_xlim(0, np.where(self.stack_starts)[0][2])
            except:
                ss_ax[0].set_xlim(0, np.where(self.im_starts)[0][50])
        elif xlim_pos == "end":
            try:
                ss_ax[0].set_xlim(
                    np.where(self.stack_starts)[0][-3],
                    np.where(self.stack_starts)[0][-1],
                )
            except:
                ss_ax[0].set_xlim(
                    np.where(self.im_starts)[0][-51], np.where(self.im_starts)[0][-1]
                )

        return ss_fig, ss_ax

    def save_current_ep_tofile(self):

        import os

        from microbiome._vendor.analysis_toolbox.utils import change_file_ext

        savename = change_file_ext(self.ephys_file, ".npy")

        if not os.path.exists(savename):
            print(f"Saving processed ephys file to {savename}")
            np.save(savename, self.ep)
        else:
            raise Exception("Processed ephys file already exists! Please check...")
