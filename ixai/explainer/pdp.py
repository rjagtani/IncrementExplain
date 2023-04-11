from ixai.explainer.base import BaseIncrementalExplainer
from ixai.utils.tracker import MultiValueTracker, WelfordTracker, ExponentialSmoothingTracker
from collections import deque, OrderedDict
from ixai.storage import BatchStorage

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import copy


class IncrementalPDP(BaseIncrementalExplainer):

    def __init__(self, model_function, feature_names,
                 pdp_feature, gridsize, storage,
                 smoothing_alpha, dynamic_setting,
                 storage_size):
        super(IncrementalPDP, self).__init__(model_function=model_function, feature_names=feature_names)
        self.pdp_feature = pdp_feature
        self.model_function = model_function
        self.gridsize = gridsize
        self.ylim = (0., 1)
        self._smoothing_alpha = 0.001 if smoothing_alpha is None else smoothing_alpha
        if dynamic_setting:
            assert 0. < smoothing_alpha <= 1., f"The smoothing parameter needs to be in the range" \
                                               f" of ']0,1]' and not " \
                                               f"'{self._smoothing_alpha}'."
            base_tracker = ExponentialSmoothingTracker(alpha=self._smoothing_alpha)
        else:
            base_tracker = WelfordTracker()
        self.pdp_y_tracker = MultiValueTracker(copy.deepcopy(base_tracker))
        self.pdp_x_tracker = MultiValueTracker(copy.deepcopy(base_tracker))
        self.ice_curves_y = deque()
        self.ice_curves_x = deque()
        self.fi_list = []
        self.seen_samples = 0
        self.storage = storage
        # TODO - Remove this
        self.storage_size = storage_size
        self.waiting_period = 20

    def _add_ice_curve_to_pdp(self, ice_curve_y, ice_curve_x):
        self.pdp_y_tracker.update(ice_curve_y)
        self.pdp_x_tracker.update(ice_curve_x)

    def _add_ice_curve_to_storage(self, ice_curve_y, ice_curve_x):
        if self.seen_samples < (self.storage_size + self.waiting_period):
            self.ice_curves_y.append(ice_curve_y)
            self.ice_curves_x.append(ice_curve_x)
        else:
            self.ice_curves_y.popleft()
            self.ice_curves_x.popleft()
            self.ice_curves_y.append(ice_curve_y)
            self.ice_curves_x.append(ice_curve_x)

    def explain_one(
            self,
            x_i
    ):
        # Warm up for explanations
        if self.seen_samples <= self.waiting_period:
            self.storage.update(x=x_i)
            self.seen_samples+=1
        else:
            # TODO - Find efficient way to store min and max for each column
            x_data, _ = self.storage.get_data()
            x_data = pd.DataFrame(x_data)
            min_value = np.min(x_data[self.pdp_feature])
            max_value = np.max(x_data[self.pdp_feature])
            feature_grid_values = np.linspace(start=min_value, stop=max_value, num=self.gridsize)
            feature_grid_dict = OrderedDict({i: value for i, value in enumerate(feature_grid_values)})
            predictions_dict = OrderedDict()
            for i, sampled_feature in enumerate(feature_grid_values):
                # TODO - remove hardcoding
                prediction = self.model_function({**x_i, self.pdp_feature: sampled_feature})[1]
                predictions_dict[i] = prediction
            self._add_ice_curve_to_pdp(ice_curve_y=predictions_dict, ice_curve_x=feature_grid_dict)
            self._add_ice_curve_to_storage(ice_curve_y=predictions_dict, ice_curve_x=feature_grid_dict)
            self.fi_list.append(np.std(list(self.pdp_y_tracker.get().values())))
            self.storage.update(x=x_i)
            self.seen_samples += 1


    def plot_pdp(self, title: str = None):
        if title is None:
            title = f"Incremental PDP Curve for feature {self.pdp_feature}"

        alphas = np.linspace(start=0.1, stop=1., num=self.storage_size)

        fig, axis = plt.subplots(1, 1)
        for ice_curve_x, ice_curve_y, alpha in zip(self.ice_curves_x, self.ice_curves_y, alphas):
            axis.plot(ice_curve_x.values(), ice_curve_y.values(), ls='-', c='black', alpha=alpha, linewidth=1)
        #print(self.pdp_x_tracker.get())
        #print(self.pdp_y_tracker.get())
        axis.plot(self.pdp_x_tracker.get().values(), self.pdp_y_tracker.get().values(), ls='-', c='red', alpha=1., linewidth=2)

        plt.title(title)
        plt.ylim(self.ylim)
        plt.xlabel(f"feature: {self.pdp_feature}")
        plt.ylabel("Model Output")
        xlim_lower = np.min(list(self.pdp_x_tracker.get().values())) * 0.95
        xlim_upper = np.max(list(self.pdp_x_tracker.get().values())) * 1.05
        plt.xlim((xlim_lower, xlim_upper))
        plt.show()

    def plot_feature_importance(self, title=None):
        if title is None:
            title = f"PDP Feature Importance Curve for feature {self.pdp_feature}"

        fig, axis = plt.subplots(1, 1)
        axis.plot(range(len(self.fi_list)), self.fi_list, ls='-', c='black', alpha=1, linewidth=2)
        plt.title(title)
        plt.xlabel(f"feature: {self.pdp_feature}")
        plt.ylabel("PDP Feature Importance")
        plt.show()


    # def plot_ice_curve(self, ice_curve_y, ice_curve_x, feature_name, title: str = None):
    #     if title is None:
    #         title = f"ICE Curve for feature {feature_name}"
    #
    #     plt.plot(ice_curve_x, ice_curve_y)
    #     plt.title(title)
    #     plt.ylim(self.ylim)
    #     plt.xlabel(f"{feature_name} range")
    #     plt.ylabel("Model Output")
    #     plt.xlim(self.xlim[feature_name])
    #     plt.show()


class BatchPDP:

    def __init__(self, pdp_feature, gridsize, model_function, ylim=None, storage=None):
        if storage is None:
            self._storage = BatchStorage(store_targets=False)
        else:
            self._storage = storage
        self.pdp_feature = pdp_feature
        self.gridsize = gridsize
        if ylim is None:
            self.ylim = (0,1)
        else:
            self.ylim = ylim
        self.model_function = model_function
        self.ice_curves_y = []
        self.ice_curves_x = []

    def _add_ice_curve_to_storage(self, ice_curve_y, ice_curve_x):
        self.ice_curves_y.append(ice_curve_y)
        self.ice_curves_x.append(ice_curve_x)

    def update_storage(self, x_i: dict):
        self._storage.update(x=x_i)

    def explain_one(self, x_i):
        self._storage.update(x_i)
        x_data, _ = self._storage.get_data()
        self.explain_many(
            x_data=x_data)

    def explain_many(self, x_data):
        x_data = pd.DataFrame(x_data)
        min_value = np.min(x_data[self.pdp_feature])
        max_value = np.max(x_data[self.pdp_feature])
        feature_grid_values = np.linspace(start=min_value, stop=max_value, num=self.gridsize)
        for x_i in x_data.to_dict('records'):
            predictions = np.empty(shape=self.gridsize)
            for i, sampled_feature in enumerate(feature_grid_values):
                #print(self.model_function({**x_i, feature_name: sampled_feature}))
                # TODO - remove hardcoding
                prediction = self.model_function({**x_i, self.pdp_feature: sampled_feature})[1]
                predictions[i] = prediction
            self._add_ice_curve_to_storage(predictions, feature_grid_values)

    def plot_pdp(self, title: str = None):
        if title is None:
            title = f"Batch PDP Curve for feature {self.pdp_feature}"

        ice_curves_x = np.asarray(self.ice_curves_x)
        mean_x = np.mean(ice_curves_x, axis=0)
        # TODO - make this a separate function
        self.pdp_x_values = mean_x
        ice_curves_y = np.asarray(self.ice_curves_y)
        mean_y = np.mean(ice_curves_y, axis=0)
        self.pdp_y_values = mean_y
        alphas = [0.5] * len(self.ice_curves_x)

        fig, axis = plt.subplots(1, 1)
        for ice_curve_x, ice_curve_y, alpha in zip(self.ice_curves_x, self.ice_curves_y, alphas):
            axis.plot(ice_curve_x, ice_curve_y, ls='-', c='black', alpha=alpha, linewidth=1)

        axis.plot(mean_x, mean_y, ls='-', c='red', alpha=1., linewidth=2)

        plt.title(title)
        plt.ylim(self.ylim)
        plt.xlabel(f"feature: {self.pdp_feature}")
        plt.ylabel("Model Output")
        xlim_low = np.min(mean_x) * 0.95
        xlim_upp = np.max(mean_x) * 1.05
        plt.xlim((xlim_low, xlim_upp))
        plt.show()

    @staticmethod
    def _calculate_percentage_overlap(range1, range2):
        """Calculate the percentage overlap between two ranges"""
        # Find the minimum and maximum values of the overlap between the two ranges
        overlap_min = max(range1[0], range2[0])
        overlap_max = min(range1[1], range2[1])
        # If there is no overlap, return 0%

        if overlap_min >= overlap_max:
            return 0



        # Calculate the length of the overlap and the lengths of each range
        overlap_length = overlap_max - overlap_min
        range1_length = range1[1] - range1[0]
        range2_length = range2[1] - range2[0]

        # Calculate the percentage overlap
        # TODO - check denominator
        percentage = (overlap_length / min(range1_length, range2_length)) * 100

        if percentage == 0:
            overlap_range = None
        else:
            overlap_range = (overlap_min,overlap_max)

        return percentage, overlap_range

    @staticmethod
    def _area_between_curves(x_coords, y_coords1, y_coords2):
        # Calculate the area between the two curves using the trapezoidal rule.
        area = 0
        for i in range(len(x_coords) - 1):
            height1 = y_coords1[i]
            height2 = y_coords2[i]
            width = x_coords[i + 1] - x_coords[i]
            area += 0.5 * width * (height1 + height2)

        return abs(area)



    def calculate_similarity_and_overlap_score(self, incremental_explainer):
        inc_pdp_x = np.array(list(incremental_explainer.pdp_x_tracker.get().values()))
        inc_pdp_y = np.array(list(incremental_explainer.pdp_y_tracker.get().values()))
        #print(self.pdp_x_values)
        #print(self.pdp_y_values)
        batch_pdp_range = (np.min(self.pdp_x_values),np.max(self.pdp_x_values))
        inc_pdp_range = (np.min(inc_pdp_x), np.max(inc_pdp_x))
        overlap_percentage, overlap_range = self._calculate_percentage_overlap(batch_pdp_range, inc_pdp_range)

        area_overlap = 0
        if overlap_percentage != 0:
            if overlap_range is not None:
                inc_pdp_x = inc_pdp_x[np.where((inc_pdp_x >= overlap_range[0]) & (inc_pdp_x <= overlap_range[1]))]

            print(inc_pdp_x)
            new_y_values = np.interp(inc_pdp_x,self.pdp_x_values,self.pdp_y_values)
            distance = np.abs(new_y_values - inc_pdp_y)

            # TODO - Confirm choice of distance measure
            mean_distance = np.mean(distance)
            # OR
            area_overlap = self._area_between_curves(inc_pdp_x, new_y_values, inc_pdp_y)

            # x_min = inc_pdp_range[0]
            # x_max = inc_pdp_range[1]
            #
            # def integrand(y1, y2):
            #     return np.abs(y1 - y2)
            #
            # # Integrate the absolute difference between the curves
            # area_overlap = quad(integrand, x_min, x_max, args=(new_y_values, inc_pdp_y))

        return overlap_percentage, area_overlap

    @staticmethod
    def plot_similarity_metrics(x_val, y_val, title=None):
        if title is None:
            title = "Similarity Measure"
        fig, axis = plt.subplots(1, 1)
        axis.plot(x_val, y_val, ls='-', c='red', alpha=1., linewidth=2)
        plt.title(title)
        plt.xlabel(f"Iteration")
        plt.ylabel("Similarity Measure")
        plt.show()




# class IncrementalPDP(BaseIncrementalExplainer):
#
#     def __init__(self, model_function, feature_names,
#                  pdp_feature_list, xlim, gridsize, storage,
#                  storage_size):
#         super(IncrementalPDP, self).__init__(model_function=model_function, feature_names=feature_names)
#         self.pdp_feature_list = pdp_feature_list
#         self.model_function = model_function
#         self.xlim = xlim
#         self.gridsize = gridsize
#         self.ylim = (0., 1)
#         self.ice_curves_y = {feature_name: deque() for feature_name in self.pdp_feature_list}
#         self.ice_curves_x = {feature_name: deque() for feature_name in self.pdp_feature_list}
#         self.seen_samples = 0
#         self.storage = storage
#         # TODO - Remove this
#         self.storage_size = storage_size
#
#
#     def _add_ice_curve_to_storage(self, ice_curve_y, ice_curve_x, feature_name):
#         if self.seen_samples < self.storage_size:
#             self.ice_curves_y[feature_name].append(ice_curve_y)
#             self.ice_curves_x[feature_name].append(ice_curve_x)
#         else:
#             self.ice_curves_y[feature_name].popleft()
#             self.ice_curves_x[feature_name].popleft()
#             self.ice_curves_y[feature_name].append(ice_curve_y)
#             self.ice_curves_x[feature_name].append(ice_curve_x)
#
#
#     def explain_one(
#             self,
#             x_i
#     ):
#         #x_data, _ = self.storage.get_data()
#         for i, feature_name in enumerate(self.pdp_feature_list):
#             feature_limits = self.xlim[feature_name]
#             min_value = feature_limits[0]
#             max_value = feature_limits[1]
#             feature_grid_values = np.linspace(start=min_value, stop=max_value, num=self.gridsize)
#             predictions = np.empty(shape=self.gridsize)
#             for i, sampled_feature in enumerate(feature_grid_values):
#                 #print(self.model_function({**x_i, feature_name: sampled_feature}))
#                 predictions[i] = self.model_function({**x_i, feature_name: sampled_feature})[1]
#                 #print(predictions.shape)
#             self._add_ice_curve_to_storage(ice_curve_y=predictions, ice_curve_x=feature_grid_values, feature_name=feature_name)
#             self.storage.update(x=x_i)
#             self.seen_samples += 1
#
#
#     def plot_pdp(self, feature_name, title: str = None):
#         if title is None:
#             title = f"PDP Curve for feature {feature_name}"
#
#         ice_curves_x = np.asarray(self.ice_curves_x[feature_name])
#         mean_x = np.mean(ice_curves_x, axis=0)
#         ice_curves_y = np.asarray(self.ice_curves_y[feature_name])
#         mean_y = np.mean(ice_curves_y, axis=0)
#
#         alphas = np.linspace(start=0.1, stop=1., num=self.storage_size)
#
#         fig, axis = plt.subplots(1, 1)
#         for ice_curve_x, ice_curve_y, alpha in zip(self.ice_curves_x[feature_name], self.ice_curves_y[feature_name], alphas):
#             axis.plot(ice_curve_x, ice_curve_y, ls='-', c='black', alpha=alpha, linewidth=1)
#
#         axis.plot(mean_x, mean_y, ls='-', c='red', alpha=1., linewidth=2)
#
#         plt.title(title)
#         plt.ylim(self.ylim)
#         plt.xlabel(f"feature: {feature_name}")
#         plt.ylabel("Model Output")
#         plt.xlim(self.xlim[feature_name])
#         plt.show()
#
#     def plot_ice_curve(self, ice_curve_y, ice_curve_x, feature_name, title: str = None):
#         if title is None:
#             title = f"ICE Curve for feature {feature_name}"
#
#         plt.plot(ice_curve_x, ice_curve_y)
#         plt.title(title)
#         plt.ylim(self.ylim)
#         plt.xlabel(f"{feature_name} range")
#         plt.ylabel("Model Output")
#         plt.xlim(self.xlim[feature_name])
#         plt.show()


