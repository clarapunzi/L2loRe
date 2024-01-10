import numpy as np
from Lib.LoreSA.datamanager import remove_missing_values, one_hot_encoding, get_real_feature_names, get_features_map


def prepare_dataset(df, class_name, encdec):

    df = remove_missing_values(df)

    #numeric_columns = get_numeric_columns(df)
    numeric_columns = list(df.select_dtypes(include='number').columns)
    categorical_columns = list(df.select_dtypes(exclude='number').columns)

    if class_name in numeric_columns:
        numeric_columns.remove(class_name)
    elif class_name in categorical_columns:
        categorical_columns.remove(class_name)

    rdf = df

    if encdec == 'onehot' or encdec == 'none':
        df, feature_names, class_values = one_hot_encoding(df, class_name)

        real_feature_names = get_real_feature_names(rdf, numeric_columns, class_name)

        rdf = rdf[real_feature_names + (class_values if isinstance(class_name, list) else [class_name])]

        features_map = get_features_map(feature_names, real_feature_names)

    elif encdec == 'target':
        feature_names = df.columns.values
        feature_names = np.delete(feature_names, np.where(feature_names == class_name))
        class_values = np.unique(df[class_name]).tolist()
        numeric_columns = list(df._get_numeric_data().columns)
        real_feature_names = [c for c in df.columns if c in numeric_columns and c != class_name]
        real_feature_names += [c for c in df.columns if c not in numeric_columns and c != class_name]
        #print('feat names and real ', len(feature_names), len(real_feature_names))
        features_map = dict()
        for f in range(0, len(real_feature_names)):
            features_map[f] = dict()
            features_map[f][real_feature_names[f]] = np.where(feature_names == real_feature_names[f])[0][0]
    return df, feature_names, class_values, numeric_columns, categorical_columns, rdf, real_feature_names, features_map