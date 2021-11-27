

# TODO organize imports; these are a mess
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from merf.merf import MERF
from merf.viz import plot_merf_training_stats

from sklearn.inspection import plot_partial_dependence
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# TODO switch to OneHotEncoder
from hot_encoding import _dummy_var_creation

from scipy.stats.stats import pearsonr

"""
TODO list 
Add baseline information and perform tests

Simple fixes
- fix hardcoding & add sample ID for all
- consider shadowed scope vars
- fix rounding for feature importance output file

Features/Considerations for development
- consider adding time as both categorical and continuous variable
    and allow flexibility
- consider what random effects make sense -- allow user to choose? 
- longitudinal vs time series
- gradient descent (ML) vs inverse matrix computation formula (stats) 
    (for one-hot encoding)


# reading
# https://slundberg.github.io/shap/notebooks/plots/dependence_plot.html
# https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Force%20Plot%20Colors.html?highlight=force%20plot
# https://www.sciencedirect.com/science/article/pii/S1532046418301758

"""

# access values from snakemake's Snakefile
in_file = snakemake.input["in_file"]
out_file = snakemake.output["out_file"]

random_forest_type = snakemake.params["random_forest_type"]
random_effect = snakemake.params["random_effect"]
sample_ID = snakemake.params["sample_ID"]
response_var = snakemake.params["response_var"]
delta_flag = snakemake.params["delta_flag"]
max_iters = snakemake.params["iterations"]
re_timepoint = snakemake.params["re_timepoint"]

# set graphics dimensions
width = 18
height = 10

# features to show
max_display = 8
# for partial dep plots and more
top_features = 8

# TODO review this
join_flag = True
out_file_prefix = out_file.split(".pdf")[0]


def setup_df_do_encoding(df_encoded, random_effect, vars_to_encode,
                         response_var, delta_flag, join_flag):
    df_encoded, dummy_dict = _dummy_var_creation(vars_to_encode, df_encoded,
                                                 join_flag)
    df_encoded.to_csv(out_file_prefix + "-df-after-encoding.txt", index=False,
                      sep="\t")

    if delta_flag == "deltas":
        drop_cols = ["StudyID.Timepoint"] + \
                    [random_effect] + vars_to_encode + [response_var]
    if delta_flag == "raw":
        drop_cols = [random_effect] + vars_to_encode + [response_var]

    df_encoded_cols_dropped = df_encoded.drop(drop_cols, axis=1)
    # TODO remove columns that have "ref"
    feature_list = df_encoded_cols_dropped.columns

    df_encoded_cols_dropped.to_csv(out_file_prefix +
                                   "-dummy-variables-dropped-columns.txt",
                                   index=False, sep="\t")

    feature_list = [x for x in feature_list if "_reference" not in x]

    x_for_split = df_encoded[feature_list + ["StudyID"]]
    x = df_encoded[feature_list]  # StudyID cannot be in features
    if re_timepoint == "re_timepoint":
        # z = df_encoded[["Timepoint_reference"]]
        z = np.ones((len(x), 1))
    if re_timepoint == "no_re":
        z = np.ones((len(x), 1))
    clusters = df_encoded["StudyID"]
    y = df_encoded[response_var]

    # Split into training and testing to check quality of model
    x_train, x_test, y_train, y_test = train_test_split(x_for_split, y, test_size=0.2,
                                                        random_state=42)
    print("********TRAINING USING MERF")
    # clusters for each individual
    clusters_train = x_train["StudyID"]
    clusters_test = x_test["StudyID"]

    # remove clusters from features X
    x_train = x_train[feature_list]
    x_test = x_test[feature_list]

    z_train = np.ones((len(x_train), 1))
    z_test = np.ones((len(x_test), 1))

    mrf_training, train_test_plots = train_merf_model(x_train, z_train,
                                                      clusters_train, y_train)

    y_train_estimate = mrf_training.predict(x_train, z_train, clusters_train)
    y_test_estimate = mrf_training.predict(x_test, z_test, clusters_test)

    corr_pearson_train = pearsonr(y_train_estimate, y_train)
    print("R2 training", corr_pearson_train)

    corr_pearson_test = pearsonr(y_test_estimate, y_test)
    print("R2 testing", corr_pearson_test)

    log_string = "Training R2: " + str(corr_pearson_train) + \
                 "\nTesting R2: " + str(corr_pearson_test)

    corr_plot_test = scatter_plot(y_test_estimate, y_test)

    return x, z, clusters, y, feature_list, dummy_dict, log_string, corr_plot_test


def merf_training_stats_plot(mrf, num_clusters):
    # print training stats TODO understand this plot more
    # plot_merf_training_stats(mrf, num_clusters_to_plot=num_clusters)
    training_stats_plot = plt.gcf()
    plt.close()
    return training_stats_plot


def scatter_plot(y_axis, x_axis):
    x_axis = np.array(x_axis)
    y_axis = np.array(y_axis)
    plt.scatter(y_axis, x_axis)
    corr_plot = plt.gcf()
    plt.close()

    return corr_plot


def train_merf_model(x, z, clusters, y):
    # TODO CHANGE THIS later! NEEDS MORE ITERATIONS
    mrf = MERF(max_iterations=max_iters)
    mrf.fit(x, z, clusters, y)
    training_stats_plot = plot_merf_training_stats(mrf, 7)
    plt.close()
    return mrf, training_stats_plot


def train_rf_model(x, y):
    mrf = RandomForestRegressor()
    mrf.fit(x, y)
    plt.plot([1, 2, 3, 4])
    plt.ylabel('place holder plot')
    training_stats_plot = plt.gcf()
    plt.close()

    return mrf, training_stats_plot


def graphic_handling(graphic):
    p = graphic
    plt.close()
    return p


def important_shapley_features(shap_values, x):
    feature_names = list(x.columns.values)
    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                      columns=['col_name',
                                               'feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'],
                                   ascending=False, inplace=True)
    feature_importance['feature_importance_vals'] = \
        feature_importance['feature_importance_vals'].apply(lambda x:
                                                            round(x, 4))
    print(feature_importance)
    return feature_importance


def shap_explainer(trained_forest_model, x):
    explainer = shap.TreeExplainer(trained_forest_model)
    shap_values = explainer.shap_values(x)

    plot_list = []

    # Bee swarm plot
    shap.summary_plot(shap_values, x, show=False, plot_size=(width, height),
                      max_display=max_display)
    plot_list.append(graphic_handling(plt.gcf()))

    # Bar plot, just a different way to look at it
    shap.summary_plot(shap_values, x, plot_type="bar", show=False,
                      plot_size=(width, height), max_display=max_display)
    plot_list.append(graphic_handling(plt.gcf()))

    # Feature Shapley values starting with best at explaining the model
    feature_importance = important_shapley_features(shap_values, x)

    top_features_list = list(feature_importance['col_name'][0:top_features])
    for col in top_features_list:
        shap.dependence_plot(col, shap_values, x, show=False)
        p = plt.gcf()
        p.set_size_inches(width, height)
        plot_list.append(p)
        plt.close()

    # for col in top_features_list:
    #     i = X.columns.get_loc(col)
    #     shap.force_plot(explainer.expected_value, shap_values[i, :],
    #                     X.iloc[i, :], show=False, matplotlib=True)
    #     p = plt.gcf()
    #     p.set_size_inches(width, height)
    #     plot_list.append(p)
    #     plt.close()

    return plot_list, feature_importance, top_features_list


def build_result_pdf(out_file, plot_list):
    """
    Build PDF containing multiple .gcf() objects (get current figure)
    Args:
        pdf_name: PDF name for storing multiple plots
        plot_list: A list of .gcf() objects
    """
    pp = PdfPages(out_file)
    for i in plot_list:
        pp.savefig(i)
    pp.close()


def main(in_file, out_file, random_forest_type, random_effect, sample_ID,
         response_var, delta_flag, join_flag):
    df = pd.read_csv(in_file, sep="\t")
    numeric_column_list = list(df._get_numeric_data().columns)
    column_list = list(df.columns)
    categoric_columns = [i for i in column_list if
                         i not in numeric_column_list]
    # exclude random effect cols and SampleID
    # TODO do I want this in model? Fixed vs random effects?
    to_drop = [random_effect, sample_ID]
    encode = [i for i in categoric_columns if i not in to_drop]

    x, z, clusters, y, feature_list, dummy_dict, log_string, corr_plot_test = \
        setup_df_do_encoding(df_encoded=df, random_effect=random_effect,
                             vars_to_encode=encode, response_var=response_var,
                             delta_flag=delta_flag, join_flag=join_flag)

    if random_forest_type == "mixed":
        mrf, training_stats_plot = train_merf_model(x, z, clusters, y)
        y_predicted = mrf.predict(x, z, clusters)

        rf = mrf.trained_fe_model

        pearson_r = pearsonr(y_predicted, y)
        print("pearson correlation y_predicted, y known full data")
        print(pearson_r)
        log_string += "\nR2 Full: " + str(pearson_r)
    else:
        mrf, training_stats_plot = train_rf_model(x, y)
        y_predicted = mrf.predict(x)

    corr_plot = scatter_plot(y_predicted, y)

    # .pdf is final snakemake check, but rename files with prefix
    out_file_prefix = out_file.split(".pdf")[0]

    f_out = open(out_file_prefix + "-log.txt", "w")
    f_out.write(log_string)
    f_out.close()

    # TODO this only uses the fixed effects RF
    plot_list, feature_importance, top_features_list = shap_explainer(rf, x)
    feature_importance.to_csv(out_file_prefix + "-feature-importance.txt",
                              sep='\t', mode='a')

    # find prefix for encoded values "ENC_Location_is_1_3" decoded is Location
    decoded_top_features_list = []
    for i in top_features_list:
        i_for_dict = i.split("_is_")[0] + "_is"
        try:
            decoded = dummy_dict[i_for_dict]
        except KeyError:
            decoded = i
        decoded_top_features_list.append(''.join(decoded))

    df = pd.DataFrame({'important.features': top_features_list,
                       'decoded.features': decoded_top_features_list})
    df.to_csv(out_file_prefix + "-top-features.txt", index=False, sep="\t")
    plot_partial_dependence(rf, x, features=top_features_list)
    plot_some_partial_dependence = plt.gcf()
    plot_some_partial_dependence.set_size_inches(width, height)
    plot_list.append(plot_some_partial_dependence)
    plot_list.append(training_stats_plot)
    plot_list.append(corr_plot_test)
    plot_list.append(corr_plot)
    build_result_pdf(out_file, plot_list)


main(in_file=in_file, out_file=out_file, random_forest_type=random_forest_type,
     random_effect=random_effect, sample_ID=sample_ID,
     response_var=response_var, delta_flag=delta_flag, join_flag=join_flag)
