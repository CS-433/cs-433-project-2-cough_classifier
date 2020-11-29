from src.utils.get_data import import_data
from src.utils.preprocessing import standard_preprocessing
from src.utils.feature_engineering import feature_engineering
from src.utils.model_helpers import AUC_all_models, homemade_all_models
DATA_PATH = "./data"





if __name__ == "__main__":
    temp = "coarse"

    # load data
    samples, labels = import_data(DATA_PATH, segmentation_type = temp, is_user_features=True)

    # TODO since it doesn hav ethis column, for late
    # TODO make better
    if temp == "no":
        samples.index = samples.index.rename('subject')

    ##### preprocessing
    # TODO -3 because the last three are categorical
    samples, labels = standard_preprocessing(samples, labels, do_smote=False)

    #### feature_engeneering
    samples, labels = feature_engineering(samples, labels)

    #### training
    results = homemade_all_models(samples, labels.Label).rename(columns={'AUC (mean)': "Coarse_AUC"})
    print(results)
