import os
import glob
import numpy
import pandas
from tqdm.auto import tqdm

from lmtanalysis import Measure

import seaborn as sns
from matplotlib import pyplot as plt

import tkinter as tk
from tkinter import filedialog


class DetectionFeatures:
    @staticmethod
    def total_distance(df, region=None):
        name = "Distance"
        magic_thres = 85.5
        dists = numpy.sqrt(numpy.square(df[["x", "y"]].diff()).sum(1))

        if region is None:
            value = dists[dists <= magic_thres].sum()
        else:
            value = dists[df[region] & (dists <= magic_thres)].sum()
            name += f"_{region}"

        return pandas.Series({name: value * Measure.scaleFactor})

    @staticmethod
    def total_time(df, region=None):
        name = "Time_spent"
        dists = numpy.sqrt(numpy.square(df[["sec"]].diff()).sum(1))

        if region is None:
            value = dists.sum()
        else:
            value = dists[df[region]].sum()
            name += f"_{region}"

        return pandas.Series({name: value})

    @staticmethod
    def speed_avg(df, region=None):
        name = "Speed_average"
        if region is not None:
            name += f"_{region}"
        return pandas.Series(
            {
                name: DetectionFeatures.total_distance(df, region).values[0]
                / DetectionFeatures.total_time(df, region).values[0]
            }
        )


class EventFeatures:
    @staticmethod
    def duration_stats(df):
        return pandas.Series(
            {
                "Duration_mean": df["duration"].mean(),
                "Duration_median": df["duration"].median(),
                "Duration_std": df["duration"].std(),
                "Duration_total": df["duration"].sum(),
            }
        )


def computeDetectionFeatures(animal_pool, start="0min", end="60min", freq=("5min",)):
    """Computes common features of the mouse trajectories in the animal pool,
       for a give time interval ranges given by start, end and list of freq
       (temporal subdivisions). Start and end stay fixed. only freq is a list.

       Use pandas convention to specifiy time parameters.
           * 3 hours  : "3H"
           * 5 minutes: "5min"

        Features computed:
            * time spent
            * distance travelled
            * average speed

    Args:
        animal_pool (AnimalPool): the animal pool
        start (str, optional): Start time. Defaults to "0min".
        end (str, optional)  : End time.   Defaults to "60min".
        freq (list[str], optional) : Time step.  Defaults to ["5min"].

    Returns:
        DataFrame: Multiindex Dataframe containing mouse meta information and
                   computed features:

    """
    detection_table = animal_pool.getDetectionTable()
    results = []
    for fr in freq:
        time_delta_rng = pandas.timedelta_range(start=start, end=end, freq=fr)

        grp = detection_table.groupby(
            ["RFID", pandas.cut(detection_table.time, bins=time_delta_rng)]
        )
        res = pandas.concat(
            [
                grp.name.first(),
                grp.genotype.first(),
                grp.apply(DetectionFeatures.total_time),
                grp.apply(DetectionFeatures.total_time, "in_arena_center"),
                grp.apply(DetectionFeatures.total_distance),
                grp.apply(DetectionFeatures.total_distance, "in_arena_center"),
                grp.apply(DetectionFeatures.speed_avg),
                grp.apply(DetectionFeatures.speed_avg, "in_arena_center"),
            ],
            axis=1,
        )

        results.append(res)

    return results


def computeMonadicEventFeatures(animal_pool, start="0min", end="60min", freq=("5min",)):
    """Computes common features of the mice' monadic events in the animal pool,
       for given time interval ranges given by start, end and list of freq
       (temporal subdivisions). Start and end stay fixed. only freq is a list.

       Use pandas convention to specifiy time parameters.
           * 3 hours  : "3H"
           * 5 minutes: "5min"

        Features computed:
            * number of events
            * duration of events
                * mean/median/std


    Args:
        animal_pool (AnimalPool): the animal pool
        start (str, optional): Start time. Defaults to "0min".
        end (str, optional)  : End time.   Defaults to "60min".
        freq (list[str], optional) : Time step.  Defaults to "5min".

    Returns:
        DataFrame: Multiindex Dataframe containing mouse meta information and
                   computed event features:

    """
    events_table = animal_pool.getAllEventsTable()
    results = []
    for fr in freq:
        time_delta_rng = pandas.timedelta_range(start=start, end=end, freq=fr)

        grp = events_table.groupby(
            ["event_name", "RFID", pandas.cut(events_table.time, bins=time_delta_rng)]
        )
        res = pandas.concat(
            [
                grp.name.first(),
                grp.genotype.first(),
                grp.size(),
                grp.apply(EventFeatures.duration_stats),
            ],
            axis=1,
        )
        res = res.rename(columns={0: "Number_of_events"})

        results.append(res)
    return results


def computeDyadicEventFeature(
    animal_pool, event_list, start="0min", end="60min", freq=("5min",)
):
    """Computes common features of dyadic events in the animal pool per genotype
       for given time interval ranges given by start, end and list of freq
       (temporal subdivisions). Start and end stay fixed. only freq is a list.

       Use pandas convention to specifiy time parameters.
           * 3 hours  : "3H"
           * 5 minutes: "5min"

        Features computed:
            * number of events
            * duration of events
                * mean/median/std


    Args:
        animal_pool (AnimalPool): the animal pool
        event_list (list[str])     : event list,
        start (str, optional): Start time. Defaults to "0min".
        end (str, optional)  : End time.   Defaults to "60min".
        freq (str, optional) : Time step.  Defaults to "5min".

    Returns:
        DataFrame: Multiindex Dataframe containing primary and secondary
        information and computed event feature:

    """
    events_table = animal_pool.getDyadicGenotypeGroupedEventTable(event_list)
    results = []
    for fr in freq:
        time_delta_rng = pandas.timedelta_range(start=start, end=end, freq=fr)

        grp = events_table.groupby(
            [
                "event_name",
                "genotype_primary",
                "genotype_secondary",
                pandas.cut(events_table.time, bins=time_delta_rng),
            ]
        )
        res = pandas.concat(
            [grp.size(), grp.apply(EventFeatures.duration_stats)], axis=1
        )
        res = res.rename(columns={0: "Number_of_events"})
        results.append(res)
    return results


class _Extractor:
    sheet_prefix = None

    def __init__(self, fn_list=None, freq="60min"):
        self.freq = freq
        self.fn_list = fn_list

        if fn_list is None:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)

            self.fn_list = filedialog.askopenfilenames(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")],
                parent=root,
            )

            if self.fn_list is None:
                root.destroy()
                raise RuntimeError("No files selected")
            root.destroy()

        self.tab = self.read_tabs()

    def read_tabs(self):
        res = []
        for xlsx_fn in self.fn_list:
            tab = pandas.read_excel(
                xlsx_fn,
                sheet_name=f"{self.sheet_prefix} {self.freq}",
                index_col=[0, 1, 2, 3],
            ).dropna()
            tab = tab.reset_index()

            tab.replace("mut1", "mut", inplace=True)
            tab.replace("mut2", "mut", inplace=True)
            tab.replace("wt1", "wt", inplace=True)
            tab.replace("wt2", "wt", inplace=True)

            tab.replace("mut3", "mut", inplace=True)
            tab.replace("mut3", "mut", inplace=True)
            tab.replace("wt4", "wt", inplace=True)
            tab.replace("wt4", "wt", inplace=True)

            tab["source_xlsx"] = os.path.basename(xlsx_fn)

            res.append(tab)

        return pandas.concat(res, axis=0)

    def export_xlsx(self):

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        fn_to_save = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            initialdir=os.path.dirname(self.fn_list[0]),
            filetypes=[("Excel files", "*.xlsx")],
            initialfile=f"export_{self.sheet_prefix}_{self.freq}",
            parent=root,
        )

        if fn_to_save:
            self.tab.to_excel(fn_to_save)

        root.destroy()


class ExtractMonadic(_Extractor):
    sheet_prefix = "Monadic Events"

    def read_tabs(self):
        tab = super().read_tabs()
        tab["condition"] = tab["genotype"]
        return tab.set_index("event_name")

    def data(self, event_name, column):
        return self.tab.loc[event_name][["condition", "time", column]]

    def plot(self, event_name, column):

        df = self.data(event_name, column)

        if self.freq == "60min":
            f, ax = plt.subplots()

            sns.boxplot(
                x="condition",
                y=column,
                data=df,
                boxprops=dict(alpha=0.1),
                fliersize=0,
                ax=ax,
            )
            sns.stripplot(x="condition", y=column, data=df, ax=ax)
            ax.set_title(event_name)

            sns.despine(ax=ax)

        else:
            g = sns.catplot(x="time", y=column, data=df, kind="strip", row="condition",)
            g.set_xticklabels(rotation=90)
            plt.tight_layout()
            plt.gcf().suptitle(event_name)

    def export_xlsx(self):

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        fn_to_save = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            initialdir=os.path.dirname(self.fn_list[0]),
            filetypes=[("Excel files", "*.xlsx")],
            initialfile=f"export_dyadic_{self.freq}",
            parent=root,
        )

        if fn_to_save:
            self.tab.to_excel(fn_to_save)

        root.destroy()


class ExtractDyadic(_Extractor):
    sheet_prefix = "Dyadic Events"

    def read_tabs(self):
        tab = super().read_tabs()
        tab["condition"] = tab["genotype_primary"] + " - " + tab["genotype_secondary"]
        return tab.set_index("event_name")

    def data(self, event_name, column):
        return self.tab.loc[event_name][["condition", "time", column]]

    def plot(self, event_name, column):

        df = self.data(event_name, column)

        if self.freq == "60min":
            f, ax = plt.subplots()

            sns.boxplot(
                x="condition",
                y=column,
                data=df,
                boxprops=dict(alpha=0.1),
                fliersize=0,
                ax=ax,
            )
            sns.stripplot(x="condition", y=column, data=df, ax=ax)
            ax.set_title(event_name)

            sns.despine(ax=ax)

        else:
            g = sns.catplot(x="time", y=column, data=df, kind="strip", row="condition",)
            g.set_xticklabels(rotation=90)
            plt.tight_layout()
            plt.gcf().suptitle(event_name)


class ExtractDetection(_Extractor):
    sheet_prefix = "Detection"

    def read_tabs(self):
        tab = super().read_tabs()
        tab["condition"] = tab["genotype"]
        return tab

    def data(self, column):
        return self.tab[["condition", "time", column]]

    def plot(self, column):

        df = self.data(column)

        if self.freq == "60min":
            f, ax = plt.subplots()

            sns.boxplot(
                x="condition",
                y=column,
                data=df,
                boxprops=dict(alpha=0.1),
                fliersize=0,
                ax=ax,
            )
            sns.stripplot(x="condition", y=column, data=df, ax=ax)

            sns.despine(ax=ax)

        else:
            g = sns.catplot(x="time", y=column, data=df, kind="strip", row="condition",)
            g.set_xticklabels(rotation=90)
            plt.tight_layout()

