import river.metrics
from river.utils import Rolling
from river.ensemble import AdaptiveRandomForestClassifier

from river.datasets.synth import Agrawal

from ixai.explainer.pdp import IncrementalPDP, BatchPDP
from ixai.explainer import IncrementalSage
from ixai.imputer import MarginalImputer
from ixai.storage import GeometricReservoirStorage
from ixai.utils.wrappers.river import RiverWrapper
from matplotlib import pyplot as plt

N_SAMPLES = 10000
DEBUG = True
FEATURE_NAME = 'commission'
CLASSIFICATION_NUM = 1
RANDOM_SEED = 42

def plot_pfi_feature_importance(pfi_list, feature_name, title = None):
    if title is None:
        title = f"PFI Feature Importance Curve for feature {feature_name}"

    fig, axis = plt.subplots(1, 1)
    axis.plot(range(len(pfi_list)), pfi_list, ls='-', c='black', alpha=1, linewidth=2)
    plt.title(title)
    plt.xlabel(f"feature: {feature_name}")
    plt.ylabel("PFI Feature Importance")
    plt.show()

if __name__ == "__main__":

    # Get Data -------------------------------------------------------------------------------------
    stream = Agrawal(classification_function=CLASSIFICATION_NUM, seed=RANDOM_SEED)
    feature_names = list([x_0 for x_0, _ in stream.take(1)][0].keys())
    n_samples = N_SAMPLES

    loss_metric = river.metrics.CrossEntropy()
    training_metric = Rolling(river.metrics.CrossEntropy(), window_size=1000)

    model = AdaptiveRandomForestClassifier(n_models=50, max_depth=10, leaf_prediction='mc')
    model_function = RiverWrapper(model.predict_proba_one)

    # Get imputer and explainers -------------------------------------------------------------------
    storage = GeometricReservoirStorage(
        size=100,
        store_targets=False,
        constant_probability=0.8
    )

    imputer = MarginalImputer(
        model_function=model_function,
        storage_object=storage,
        sampling_strategy="joint"
    )

    batch_explainer = BatchPDP(pdp_feature=FEATURE_NAME,
                             gridsize=8,
                             model_function=model_function)

    incremental_explainer = IncrementalPDP(feature_names=feature_names, pdp_feature=FEATURE_NAME,
                             gridsize=8, storage=storage, smoothing_alpha=0.1,
                             model_function=model_function, dynamic_setting=True,
                             storage_size=100)

    incremental_pfi = IncrementalSage(
        model_function=model_function,
        loss_function=loss_metric,
        imputer=imputer,
        storage=storage,
        feature_names=feature_names,
        smoothing_alpha=0.1,
        n_inner_samples=1
    )

    # Training Phase -----------------------------------------------------------------------------------------------
    if DEBUG:
        print(f"Starting Training for {n_samples}")
    for (n, (x_i, y_i)) in enumerate(stream, start=1):
        y_i_pred = model_function(x_i)
        training_metric.update(y_true=y_i, y_pred=y_i_pred)
        model.learn_one(x_i, y_i)
        if DEBUG and n % 1000 == 0:
            print(f"{n}: performance {training_metric.get()}\n")
        if n > n_samples:
            break

    pfi_list = []
    for (n, (x_i, y_i)) in enumerate(stream, start=1):
        incremental_explainer.explain_one(x_i)
        incremental_pfi.explain_one(x_i, y_i, update_storage=False)
        pfi_dict = incremental_pfi.importance_values
        if bool(pfi_dict):
            pfi_list.append(pfi_dict[FEATURE_NAME])
        batch_explainer.update_storage(x_i)

        # if n % 1000 == 0:
        #     incremental_explainer.plot_pdp()

        if n >= n_samples:
            batch_explainer.explain_one(x_i)
            batch_explainer.plot_pdp()
            incremental_explainer.plot_pdp()
            incremental_explainer.plot_feature_importance()
            plot_pfi_feature_importance(pfi_list, FEATURE_NAME)
            break
