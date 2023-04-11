import river.metrics
from river.utils import Rolling
from river.ensemble import AdaptiveRandomForestClassifier

from river.datasets.synth import Agrawal

from ixai.explainer.pdp import IncrementalPDP, BatchPDP
from ixai.storage import GeometricReservoirStorage
from ixai.utils.wrappers.river import RiverWrapper

N_SAMPLES = 20000
DEBUG = True

if __name__ == "__main__":

    # Get Data -------------------------------------------------------------------------------------
    stream = Agrawal(classification_function=1, seed=42)
    feature_names = list([x_0 for x_0, _ in stream.take(1)][0].keys())
    n_samples = N_SAMPLES

    loss_metric = river.metrics.CrossEntropy()
    training_metric = Rolling(river.metrics.CrossEntropy(), window_size=1000)

    model = AdaptiveRandomForestClassifier(n_models=15, max_depth=10, leaf_prediction='mc')
    model_function = RiverWrapper(model.predict_proba_one)

    # Get imputer and explainers -------------------------------------------------------------------
    storage = GeometricReservoirStorage(
        size=100,
        store_targets=False,
        constant_probability=0.8
    )

    batch_explainer = BatchPDP(pdp_feature='salary',
                             gridsize=8,
                             model_function=model_function)

    incremental_explainer = IncrementalPDP(feature_names=feature_names, pdp_feature='salary',
                             gridsize=8, storage=storage, smoothing_alpha=0.01,
                             model_function=model_function, dynamic_setting=True,
                             storage_size=100)

    # Training Phase -----------------------------------------------------------------------------------------------
    if DEBUG:
        print(f"Starting Training for {n_samples}")
    for (n, (x_i, y_i)) in enumerate(stream, start=1):
        y_i_pred = model_function(x_i)
        training_metric.update(y_true=y_i, y_pred=y_i_pred)
        model.learn_one(x_i, y_i)
        batch_explainer.update_storage(x_i)
        if DEBUG and n % 1000 == 0:
            print(f"{n}: performance {training_metric.get()}\n")
        if n > n_samples:
            batch_explainer.explain_one(x_i)
            batch_explainer.plot_pdp()
            break

    overlap_list = []
    area_list = []
    x_val = []
    for (n, (x_i, y_i)) in enumerate(stream, start=1):
        incremental_explainer.explain_one(x_i)

        if n % 1000 == 0:
            incremental_explainer.plot_pdp(title=f"PDP after {n} samples -2")
            overlap_pct, area_bw = batch_explainer.calculate_similarity_and_overlap_score(incremental_explainer)
            overlap_list.append(overlap_pct)
            area_list.append(area_bw)
            x_val.append(n)


        if n >= n_samples:
            BatchPDP.plot_similarity_metrics(x_val, overlap_list, title = "Overlap Percentage")
            BatchPDP.plot_similarity_metrics(x_val, area_list, title = "Distance Measure")

            #incremental_explainer.plot_pdp()

            break
