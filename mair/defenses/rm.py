from collections import OrderedDict
from datetime import datetime
import itertools

import numpy as np

import torch

import matplotlib.pyplot as plt
from ._vis import init_plot, make_twin, plot_line, _to_array


class HistoryMeter(object):
    def __init__(self, name):
        self.name = name
        self.history = {}
        self.val = None

    def __getitem__(self, idx):
        return self.history.get(idx)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.name)

    def update(self, val, idx):
        self.history[idx] = val
        self.val = val

    def get_str(self):
        return None
    
    def state_dict(self):
        state_dict = OrderedDict(class_name=self.__class__.__name__,
                                 name=self.name,
                                 history=self.history,
                                 val=self.val,
                                )
        return state_dict
    
    def load_state_dict(self, state_dict):
        assert state_dict['class_name'] == self.__class__.__name__
        self.name = state_dict['name']
        self.history = state_dict['history']
        self.val = state_dict['val']
        return True


class IntMeter(HistoryMeter):
    def get_str(self):
        return "%d" % self.val

    def __getitem__(self, idx):
        val = self.history.get(idx)
        return val if val is not None else np.nan


class FloatMeter(HistoryMeter):
    def get_str(self):
        return "%.4f" % self.val

    def __getitem__(self, idx):
        val = self.history.get(idx)
        return val if val is not None else np.nan


class RecordManager(object):
    def __init__(self):
        self.records = OrderedDict()
        self.record_type = None
        self.save_path = None
        self.log = []  # Add logs since TextIOWrapper cannot saved.
        self.count = 0

        # For best records
        self.best_records = None
        self.best_option = None
        self._LBS = []
        self._HBS = []
        self._LBOS = []
        self._HBOS = []
        
        self._init_head = False
        self._init_time = datetime.now()
        self._progress_start_time = None
        self._progress_end_time = None
        self._progress_times = []

        self._spinner = itertools.cycle(["-", "/", "|", "\\"])
        
        self._print_form = None
        self._print_len = None

    def __getitem__(self, idx):
        return {key: meter.get(idx) for key, meter in self.records.items()}

    def __repr__(self):
        return "%s(keys=[%s])" % (
            self.__class__.__name__,
            ", ".join(self.records.keys()),
        )

    def initialize(self, record_type=None, save_path=None, best_option=None):
        self.record_type = record_type
        self.best_option = best_option
        self.save_path = save_path
        self._init_head = True
        if self.best_option is not None:
            self.define_best(self.best_option)

    def update(self, record_type=None, save_path=None, best_option=None):
        self.record_type = record_type
        self.best_option = best_option
        self.save_path = save_path
        self._init_head = True
        if self.best_option is not None:
            self.define_best(self.best_option)

    def progress_start(self):
        self._progress_start_time = datetime.now()

    def progress_end(self):
        self._progress_end_time = datetime.now()
        t = self._progress_end_time - self._progress_start_time
        self.print_only(self.record_type,
            "Progress: " + next(self._spinner) + " [" + str(t) + "/it]" + " " * 20,
            end="\r",
        )
        self._progress_times.append(t.total_seconds())

    def print(self, record_type, str, *args, **kargs):
        if record_type is not None:
            print(str, *args, **kargs)
            self.log.append(str)
            self.save_log()

    def print_only(self, record_type, str, *args, **kargs):
        if record_type is not None:
            print(str, *args, **kargs)

    def _add_progress_time(self, dict_record):
        dict_record["s/it"] = np.array(self._progress_times).mean()
        self._progress_times = []

    def add(self, dict_record):
        self.count += 1
        flag_new_key_addded = False
        dict_print = OrderedDict()
        self._add_progress_time(dict_record)

        for key, val in dict_record.items():
            # Check whether there exist meter for key.
            meter = self.records.get(key)
            if meter is None:
                val_type, val = self.check_type_and_transform(val)
                meter = self.assign_meter(val_type, key)
                self.records[key] = meter
                flag_new_key_addded = True
            meter.update(val, self.count)

            # If string, print.
            string = meter.get_str()
            if string:
                dict_print[key] = string

        if self.record_type == "Epoch":
            del dict_print["Iter"]

        # Print head
        if flag_new_key_addded or self._init_head:
            self._print_head(dict_print)

        # Print row
        self.print(self.record_type, self._print_form.format(*dict_print.values()))
        self.print(self.record_type, "-" * self._print_len)

    def _print_head(self, dict_print, slack=3):
        lengths = []
        for i, (key, value) in enumerate(dict_print.items()):
            length = max(len(value), len(key)) + slack
            lengths.append(length)

        self._print_form = "".join(
            ["{:<" + str(length) + "." + str(length) + "}" for length in lengths]
        )

        text = self._print_form.format(*dict_print.keys())
        self._print_len = len(text)
        self.print(self.record_type, "-" * self._print_len)
        self.print(self.record_type, text)
        self.print(self.record_type, "=" * self._print_len)
        self._init_head = False

    @staticmethod
    def assign_meter(val_type, key):
        if val_type == int:
            return IntMeter(key)
        elif val_type == float:
            return FloatMeter(key)
        return HistoryMeter(key)

    def define_best(self, best_option):
        for key, judge in best_option.items():
            if judge == "LB":
                self._LBS.append(key)
            elif judge == "HB":
                self._HBS.append(key)
            elif judge == "LBO":
                self._LBOS.append(key)
            elif judge == "HBO":
                self._HBOS.append(key)
            else:
                raise ValueError(
                    "Values of save_best should be in ['LB', 'HB' ,'LBO', 'HBO']."
                )

    def check_best(self, dict_record):
        if self.best_option is None:
            return False
        
        if self.count == 0:
            return False

        if dict_record is None:
            return False
        else:
            for key in self._LBS + self._HBS + self._LBOS + self._HBOS:
                if dict_record.get(key) is None:
                    raise ValueError("%s is not in records."%key)

        if self.best_records is not None:
            flag_tie = True
            # Lower is better
            for key in self._LBS:
                best_value = self.best_records[key]
                curr_value = dict_record[key]
                if np.isnan(curr_value):
                    return False
                if best_value < curr_value:
                    return False
                elif best_value > curr_value:
                    flag_tie = False
                else:
                    pass

            # Higher is better
            for key in self._HBS:
                best_value = self.best_records[key]
                curr_value = dict_record[key]
                if np.isnan(curr_value):
                    return False
                if best_value > curr_value:
                    return False
                elif best_value < curr_value:
                    flag_tie = False
                else:
                    pass

            # If tie, go for options
            if flag_tie is True:
                # Lower is better
                for key in self._LBOS:
                    best_value = self.best_records[key]
                    curr_value = dict_record[key]
                    if best_value < curr_value:
                        return False

                # Higher is better
                for key in self._HBOS:
                    best_value = self.best_records[key]
                    curr_value = dict_record[key]
                    if best_value > curr_value:
                        return False
        self.best_records = dict_record
        return True

    def generate_summary(self):
        self.print(self.record_type, "=" * self._print_len)
        self.print(self.record_type, "Total Epoch: %d" % self.records["Epoch"].val)
        self.print(self.record_type, "Start Time: " + str(self._init_time))
        self.print(self.record_type, "Time Elapsed: " + str(datetime.now() - self._init_time))

        if self.best_option is not None:
            self.print(self.record_type, "Best Records: ")
            for key, val in self.best_records.items():
                if key in ["Epoch", "Iter"]:
                    self.print(self.record_type, "- %s: %d" % (key, val))
                if key in self.best_option.keys():
                    self.print(self.record_type, "- %s: %.4f" % (key, val))

        self.print(self.record_type, "-" * self._print_len)

    def save_log(self):
        if self.save_path is not None:
            with open(self.save_path + "/log.txt", "w") as f:
                f.write("\n".join(self.log) + "\n")

    def plot(
        self,
        inputs,
        x=None,
        figsize=(6, 6),
        title="",
        xlabel="",
        ylabel="",
        xlim=None,
        ylim=None,
        pad_ratio=0,
        tight=True,
        linestyles=None,
        linewidths=None,
        colors=None,
        labels=None,
        alphas=None,
        ylabel_second="",
        ylim_second=None,
        legend=True,
        loc="best",
    ):

        # Check version and number of elements
        ver = 1  # e.g.["a"] or ["a", "b", "c"]
        length = 0
        y_keys_flat = []
        js = []
        lines2 = []
        labels2 = []

        for j, y_key in enumerate(inputs):
            if isinstance(y_key, list):

                if len(inputs) > 2:
                    raise RuntimeError("Axes can have the maximum value as 2.")

                for y in y_key:
                    y_keys_flat.append(y)
                    length += 1
                    js.append(j)

                ver = 2  # e.g. [["a", "b"], "c"] or [["a", "b"], ["c", "d"]]

            else:
                y_keys_flat.append(y_key)
                length += 1
                js.append(j)

        idxs = list(range(self.count + 1))[1:]
        if x == "Epoch":
            v = self.records["Epoch"].history
            # Extract the last idx of each epoch.
            idxs = [idx for idx in idxs[:-1] if v[idx + 1] != v[idx]] + [idxs[-1]]
            x = [v[idx] for idx in idxs]
        else:
            x = idxs

        data = {}
        for key, meter in self.records.items():
            if isinstance(meter, IntMeter) or isinstance(meter, FloatMeter):
                data[key] = [meter.history[idx] for idx in idxs]

        inputs = [data[y_key] for y_key in y_keys_flat]

        # Draw plots
        ax = init_plot(
            ax=None,
            figsize=figsize,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            pad_ratio=pad_ratio,
            tight=tight,
        )

        linestyles, linewidths, colors, labels, alphas = _to_array(
            [linestyles, linewidths, colors, labels, alphas], length
        )

        i = 0
        if ver == 1:
            for input in inputs:
                plot_line(
                    ax,
                    input,
                    x,
                    linestyle=linestyles[i],
                    linewidth=linewidths[i],
                    color=colors[i],
                    label=labels[i],
                    alpha=alphas[i],
                )
                i += 1

        elif ver == 2:
            ax2 = make_twin(ax=ax, ylabel=ylabel_second, ylim=ylim_second)
            axes = [ax, ax2]
            for j, input in enumerate(inputs):
                plot_line(
                    axes[js[j]],
                    input,
                    x,
                    linestyle=linestyles[i],
                    linewidth=linewidths[i],
                    color=colors[i],
                    label=labels[i],
                    alpha=alphas[i],
                )
                i += 1

            lines2, labels2 = ax2.get_legend_handles_labels()

        else:
            raise RuntimeError("Unreadable inputs")

        if legend:
            lines, labels = ax.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc=loc)

        ax.set_xticks(x)
        plt.show()

    @staticmethod
    def check_type_and_transform(val):
        # if isinstance(val, torch.Tensor):
        #     if val.numel() == 1:
        #         val = val.item()
        # if isinstance(val, np.ndarray):
        #     if val.size == 1:
        # Instead above,
        try:
            val = val.item()
        except:
            pass
        return type(val), val

    def state_dict(self):
        records = OrderedDict()
        for key in self.records.keys():
            records[key] = self.records[key].state_dict()
        
        state_dict = OrderedDict(class_name=self.__class__.__name__,
                                 records=records,
                                 record_type=self.record_type,
                                 best_records=self.best_records,
                                 best_option=self.best_option,
                                 log=self.log,
                                 count=self.count,
                                 _init_head=self._init_head,
                                 _init_time=self._init_time,
                                 _progress_start_time=self._progress_start_time,
                                 _progress_end_time=self._progress_end_time,
                                 _progress_times=self._progress_times,
                                 _spinner=self._spinner,
                                 save_path=self.save_path,
                                 _LBS=self._LBS,
                                 _HBS=self._HBS,
                                 _LBOS=self._LBOS,
                                 _HBOS=self._HBOS,
                                )
        return state_dict
        
    def load_state_dict(self, state_dict):
        assert state_dict['class_name'] == self.__class__.__name__
        for key in state_dict.keys():
            if key in ['class_name']:
                continue
            elif key in ['records']:
                records = OrderedDict()
                records_dict = state_dict['records']
                for record_key in records_dict.keys():
                    records[record_key] = globals()[records_dict[record_key]['class_name']](None)
                    records[record_key].load_state_dict(records_dict[record_key])
                self.records = records
            else:
                setattr(self, key, state_dict[key])
        return True