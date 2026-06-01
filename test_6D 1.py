from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
import argparse
import re
from pathlib import Path
import uproot
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import ROOT
from tqdm import tqdm
import sys
from scipy import stats
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import joblib
from scipy.stats import norm
import xgboost as xgb
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from sklearn.svm import SVR

 # Import the necessary libraries for neural networks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.losses import Loss
from keras.utils import plot_model
from offset_aware_timing_model import OffsetAwareTimingModel


TEST_6D_COLUMNS = ["Ei", "index_i", "Ej", "index_j", "tdiff", "tdiff_aligned"]


def run_label_from_path(path):
    match = re.search(r"(run[-_]?\d+)", Path(path).name, flags=re.IGNORECASE)
    if match:
        return match.group(1).replace("_", "-")
    stem = Path(path).stem
    if stem.startswith("test_6D_input_"):
        stem = stem[len("test_6D_input_"):]
    return stem


def apply_default_output_paths(args):
    run_label = run_label_from_path(args.input_data)
    if args.model_output is None:
        args.model_output = f"trained_model_{run_label}.joblib"
    if args.root_output is None:
        args.root_output = f"test_6D_output_{run_label}.root"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/evaluate the 6D timing model from extracted pandas input."
    )
    parser.add_argument(
        "input_data",
        nargs="?",
        default="test_6D_input.pkl",
        help=(
            "Extractor output file. Use the .pkl written by "
            "extract_test_6D_input.py, or a CSV with the test_6D columns."
        ),
    )
    parser.add_argument(
        "--model-output",
        help="Path where the trained joblib pipeline will be written. Default includes input run label.",
    )
    parser.add_argument(
        "--root-output",
        help="Path where the output ROOT file will be written. Default includes input run label.",
    )
    parser.add_argument(
        "--model-plot",
        help="Optional path for a Keras model diagram. Omit to skip plotting.",
    )
    return parser.parse_args()


def load_test_6d_input(input_data):
    input_path = Path(input_data)
    if input_path.suffix.lower() == ".pkl":
        loaded = pd.read_pickle(input_path)
        if isinstance(loaded, dict):
            df = loaded.get("df")
            if df is None:
                raise KeyError(f"{input_path} does not contain a 'df' dataframe")
            df_filtered = filter_test_6d_df(df)
        else:
            df = loaded
            df_filtered = filter_test_6d_df(df)
    else:
        df = pd.read_csv(input_path)
        df_filtered = filter_test_6d_df(df)

    missing = [name for name in TEST_6D_COLUMNS if name not in df.columns]
    if missing:
        raise KeyError(f"{input_path} is missing required columns: {missing}")
    if df_filtered.empty:
        raise ValueError(f"{input_path} produced an empty df_filtered dataframe")
    return df[TEST_6D_COLUMNS].copy(), df_filtered[TEST_6D_COLUMNS].copy()


def filter_test_6d_df(df):
    return df[df["tdiff"].between(-200, 200)].copy()


def padded_range(values, fallback=(-1.0, 1.0), padding_fraction=0.05):
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return fallback
    low = float(values.min())
    high = float(values.max())
    if low == high:
        pad = max(abs(low) * padding_fraction, 1.0)
    else:
        pad = (high - low) * padding_fraction
    return low - pad, high + pad


def binned_axis(low, high, bin_width):
    bins = max(1, int(np.ceil((high - low) / bin_width)))
    return bins, low, low + bins * bin_width


args = parse_args()
apply_default_output_paths(args)

# Define a custom loss class
class CustomStdDevLoss(Loss):
    def __init__(self, name='custom_std_dev_loss'):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # Calculate the standard deviation of predictions
        std_dev = tf.math.reduce_std(y_pred, axis=0)  # Compute standard deviation along axis 0
        
        # Return the negative of the standard deviation as a loss
        # Minimizing this loss will encourage predictions with lower standard deviation
        return -std_dev



# Read the extracted dataframes produced by extract_test_6D_input.py.
df, df_filtered = load_test_6d_input(args.input_data)

print("df_filtered")
print(df_filtered)

# # Split the data into training and test sets
# df1, df2 = train_test_split(df_filtered, test_size=1, random_state=42)

# # Assuming you have loaded the training data into numpy arrays or pandas DataFrame
# X_train = df1.iloc[:, :-1]  # Get all columns except the last one as features
# y_train = df1.iloc[:, -1]  # Get the last column as the target

# # Assuming you have loaded the test data into numpy arrays or pandas DataFrame
# X_test = df2.iloc[:, :-1]  # Get all columns except the last one as features
# y_test = df2.iloc[:, -1]  # Get the last column as the target

# # categorical_encoder = ColumnTransformer(
# #     transformers=[#         ('encode_0', OneHotEncoder(categories='auto', sparse=False), [1]),
# #         ('encode_2', OneHotEncoder(categories='auto', sparse=False), [3])
# #     ])



features = df.drop('tdiff', axis=1)
target = df['tdiff']

# Remove outliers using Z-score method
z_scores = (target - target.mean()) / target.std()
outliers = (z_scores.abs() > 5)
outliers_df = df[outliers]
filtered_data = df[~outliers]

# Print the rows that were dropped as outliers
print("Dropped Outliers:")
print(outliers_df)



# numeric_feature = 'tdiff'

# # Create bins (replace this with your chosen binning approach)
# bins = np.linspace(df[numeric_feature].min(), df[numeric_feature].max(), num=100000)

# # Calculate bin frequencies
# bin_counts = pd.cut(df[numeric_feature], bins=bins).value_counts()

# # Set a threshold (for example, 1% of the total data)
# threshold = 1800

# # Identify low-frequency bins
# low_frequency_bins = bin_counts[bin_counts <=threshold].index

# # Remove or group data points within low-frequency bins
# data = df[~df[numeric_feature].isin(df[numeric_feature][pd.cut(df[numeric_feature], bins=bins).isin(low_frequency_bins)])]

# print(data)








# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     df_filtered.drop('tdiff', axis=1),
#     df_filtered['tdiff'],
#     test_size=0.1,
#     random_state=42
# )


# preprocessor = ColumnTransformer(
#     transformers=[
#            ('log_transform', np.log1p, [0,2,4]),
#          ('onehot', OneHotEncoder(categories='auto', sparse=False), [1,3]),
#           ('scale', StandardScaler(), [0, 2, 4])
#      ],
#     remainder='passthrough')



# pipeline = Pipeline([
#     ('preprocessor', preprocessor),
#     ('regressor',LinearRegression())
# ])

X_train, X_test, y_train, y_test = train_test_split(
    df_filtered.drop(['tdiff', 'tdiff_aligned'], axis=1),
    df_filtered['tdiff'],
    test_size=0.1,
    random_state=42,
   # stratify=df_filtered['tdiff']
)
print(y_test)
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer for log1p

class Log1pTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log1p(X)

# Custom transformer for expm1
class Expm1Transformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.expm1(X)

# Define the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('log_transform', Log1pTransformer(), [0, 2]),
         ('onehot', OneHotEncoder(categories='auto', sparse=False), [1, 3]),
        ('scale', StandardScaler(), [0, 2])
    ],
    remainder='passthrough'
)

model = Sequential([
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)  # Output layer for regression
])
if args.model_plot:
    plot_model(model, to_file=args.model_plot, show_shapes=True, show_layer_names=True)
# def custom_std_dev_loss(y_true, y_pred):
#     # Calculate the standard deviation of predictions
#     std_dev = tf.math.reduce_std(y_pred, axis=0)  # Compute standard deviation along axis 0
    
#     # Return the negative of the standard deviation as a loss
#     # Minimizing this loss will encourage predictions with lower standard deviation
#     return -std_dev




# # Create an instance of the custom loss class
# custom_loss = CustomStdDevLoss()
# tf.losses.add_loss(custom_loss)
def custom_std_dev_loss(y_true, y_pred):
    # Calculate the standard deviation of predictions
    std_dev = tf.math.reduce_std(y_pred, axis=0)  # Compute standard deviation along axis 0
    
    # Return the negative of the standard deviation as a loss
    # Minimizing this loss will encourage predictions with lower standard deviation
    return -std_dev


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss=Huber(delta=0.2))



# Create the pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', model)
])





# Now you can use the pipeline for your data
# X_train_transformed = pipeline.fit_transform(X_train, y_train)




# # Define the hyperparameters to search over
# param_grid = {
#     'regressor__n_estimators': [100],
#     'regressor__max_depth': [ 20],
#     'regressor__min_samples_split': [10],
#     'regressor__min_samples_leaf': [4]
# }

# # Perform grid search using cross-validation
# with tqdm(total=len(param_grid), desc="Grid Search Progress") as pbar:
#     grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring='neg_mean_squared_error')
#     grid_search.fit(X_train, y_train)
#     pbar.update(1)




# # Get the best hyperparameters
# best_params = grid_search.best_params_

# # Train the final model using the best hyperparameters
# final_model = grid_search.best_estimator_

# # Evaluate the final model on the test set
# test_score = final_model.score(X_test, y_test)
# print("Best Hyperparameters:", best_params)
# print("Test Score (R^2):", test_score)









# xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)  # You can adjust hyperparameters

# # Create the pipeline with the preprocessor and XGBoost model
# pipeline = Pipeline([
#     ('preprocessor', preprocessor),
#     ('regressor', xgb_model)
# ])

# Split the data into training and test sets

# X_train[:,0,2,4]=np.log1p(X_train[:,0,2,4])
# X_test=np.log1p(X_test)

# # Fit the pipeline to the training data
# pipeline.fit(X_train, y_train)

# # Predict the target variable on the test set
# y_test_predicted = pipeline.predict(X_test)

# # Calculate the mean squared error
# mse = np.mean((y_test - y_test_predicted) ** 2)
# print('Mean Squared Error:', mse)




# Create the XGBoost model
#xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)  # You can adjust hyperparameters

# Fit the model to the training data
pipeline.fit(X_train, y_train, regressor__batch_size=5, regressor__epochs=1)
#pipeline.fit_transform(X_train, y_train)
X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)
scores = model.evaluate(X_test_transformed, y_test)
loss_value = scores[0] if isinstance(scores, (list, tuple)) else scores
print("\n%s: %.2f%%" % (model.metrics_names[0], loss_value * 100))
y_test_predicted = pipeline.predict(X_test, batch_size=5)

df_X_test=pd.DataFrame(X_test)
df_Y_test=pd.DataFrame(y_test)
df_Y_pred=pd.DataFrame(y_test_predicted)
# df_Y_diff=pd.DataFrame(y_diff)
y_test_np = df_Y_test.values
y_pred_np = df_Y_pred.values
# y_diff_np = df_Y_diff.values





print(y_test_predicted)
print('y_test')
y_test_predicted = y_test_predicted.astype(y_test.dtype)
y_test_trans=y_test.iloc[0]
print(y_test)
# Calculate the mean squared error
mse = np.mean((y_test_np - y_test_predicted) ** 2)
print('Mean Squared Error:', mse)

#X_test=np.expm1(X_test)

















# # Create the pipeline with the preprocessor step and RandomForestRegressor
# pipeline = Pipeline([
#      ('preprocessor', preprocessor),
#     ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))  # You can adjust n_estimators and other hyperparameters
# ])









# # # preprocessor = ColumnTransformer(
# # #     transformers=[
# # #         ('poly', PolynomialFeatures(degree=3, include_bias=True), [0,1,2,3]),
# # #         #  ('encode', OneHotEncoder(categories='auto', sparse=False), [0, 2]),
# # #         # ('pca', PCA(n_components=2), [1, 3])  # Apply PCA to the desired features
# # #         ('onehot', OneHotEncoder(categories='auto', sparse=True), [1,3]),
# # #          ('scale', StandardScaler(), [0, 2])  # Scale the desired features
# # #     ])


# # # # Create the pipeline with the preprocessor step
# # # pipeline = Pipeline([
# # #     ('preprocessor', preprocessor),
# # #     ('regressor',LinearRegression())
# # # ])

# # Fit the pipeline to the training data
# pipeline.fit(X_train, y_train)

# # feature_importances = pipeline.feature_importances_

# # # Create a DataFrame to hold feature importances along with their names
# # feature_importances_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

# # # Sort the DataFrame by importance values in descending order
# # feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

# # # Print the sorted feature importances
# # print("Feature Importances:")
# # print(feature_importances_df)


# # Access the RandomForestRegressor within the pipeline
# random_forest_regressor = pipeline.named_steps['regressor']

# # Get feature importances from the RandomForestRegressor
# feature_importances = random_forest_regressor.feature_importances_

# # Print the feature importances
# print("Feature Importances:")
# for feature_name, importance in zip(X_train.columns, feature_importances):
#     print(f"{feature_name}: {importance}")








# # Inverse transform y_test
# y_test_predicted = pipeline.predict(X_test)

# Print the predictions
# mse = np.mean(( y_test-y_test_predicted) ** 2)
# print('Mean Squared Error:', mse)


# residuals = y_test - y_test_predicted

# # Calculate the mean squared error
# mse = np.mean(residuals ** 2)
# print('Mean Squared Error:', mse)

# # Fit a Gaussian distribution to the residuals
# mu, std = norm.fit(residuals)

# print('Standard Deviation of Gaussian Fit:', std)
 # Save the trained model with run-by-run detector-offset detection support.
timing_model = OffsetAwareTimingModel(pipeline)
dump(timing_model, args.model_output)


# Convert the y_diff array to a numpy array
# y_diff_np = np.array(y_diff)



# Assuming you have the 'y_pred' NumPy array
output_file = args.root_output
tree_name = "TreeOutput"
# branch_name = "y_pred"
# branch_name1="y_test"
# branch_name2="y_diff_np"






# Create a ROOT file and a TTree
root_file = ROOT.TFile(output_file, "recreate")
tree = ROOT.TTree(tree_name, tree_name)
Ei = np.zeros(1, dtype=float)
index_i = np.zeros(1, dtype=int)
Ej = np.zeros(1, dtype=float)
index_j = np.zeros(1, dtype=int)
T_Diff_Corrected=np.zeros(1, dtype=float)
T_Diff=np.zeros(1, dtype=float)
T_pred=np.zeros(1, dtype=float)
Dynode=np.zeros(1, dtype=float)
# Disable automatic notifications
# tree.SetNotify(0)










# Create the branches in the TTree
tree.Branch("Ei", Ei, "Ei/D")
tree.Branch("index_i", index_i, "index_i/I")
tree.Branch("Ej", Ej, "Ej/D")
tree.Branch("index_j", index_j, "index_j/I")
tree.Branch("T_Diff_Corrected",T_Diff_Corrected, "T_Diff_Corrected/D")
tree.Branch("T_Diff",T_Diff, "T_Diff/D")
tree.Branch("T_pred",T_pred, "T_pred/D")
tree.Branch("Dynode",Dynode,"Dynode/D")

tree.Fill()





tdiff_corrected_np = y_test_np - y_pred_np
tdiff_corr_min, tdiff_corr_max = padded_range(tdiff_corrected_np)
tdiff_min, tdiff_max = padded_range(y_test_np)
ej_min, ej_max = padded_range(df_X_test["Ej"].to_numpy())
ej_prompt_low, ej_prompt_high = 494.55, 527.45
ei_prompt_low, ei_prompt_high = 494.55, 527.45
time_corr_bins, tdiff_corr_min, tdiff_corr_max = binned_axis(tdiff_corr_min, tdiff_corr_max, 0.01)
time_bins, tdiff_min, tdiff_max = binned_axis(tdiff_min, tdiff_max, 0.01)
energy_bins, ej_min, ej_max = binned_axis(ej_min, ej_max, 10.0)

hist2d = ROOT.TH2D("Ej_Vs_T_Diff_Corrected", "Ej_Vs_T_Diff_Corrected",time_corr_bins,tdiff_corr_min,tdiff_corr_max,energy_bins,ej_min,ej_max)
hist2d1 = ROOT.TH2D("Ej_Vs_T_Diff", "Ej_Vs_T_Diff",time_bins,tdiff_min,tdiff_max,energy_bins,ej_min,ej_max)
hist2d2 = ROOT.TH2D("T_Diff_Counts_Vs_Detector_Pair", "T_Diff_Counts_Vs_Detector_Pair",15,0,15,15,0,15)
hist2d3 = ROOT.TH2D("Average_T_Diff_Vs_Detector_Pair", "T_Diff_Corrected_Couts_Vs_Detector_Pair",15,0,15,15,0,15)
hist2d4 = ROOT.TH1D("Prompt_Response_corrected", "Prompt_Response_Corrected",time_corr_bins,tdiff_corr_min,tdiff_corr_max)
hist2d5 = ROOT.TH1D("Prompt_Response", "Prompt_Response",time_bins,tdiff_min,tdiff_max)


num_entries = len(df_X_test)
for i in tqdm(range(num_entries), desc='Processing Entries'):
# for i in range(num_entries):
    row = df_X_test.iloc[i]
    # row1=df_Y_diff.iloc[i]
    Ei[0] = row[0]
    index_i[0] = row[1]
    Ej[0] = row[2]
    index_j[0] = row[3]
    # Dynode[0]=row[4]
    # Corrected time is the residual: measured tdiff minus predicted tdiff.
    T_Diff_Corrected[0] = y_test_np[i] - y_pred_np[i]
    T_Diff[0] = y_test_np[i]
    T_pred[0] = y_pred_np[i]
    if((row[3]!=15) & (row[1]!=15)):
        hist2d.Fill(T_Diff_Corrected[0], row[2])
        hist2d1.Fill(y_test_np[i],row[2])
    if((row[2]>=ej_prompt_low) & (row[2]<=ej_prompt_high)&(row[3]!=15)&(row[1]!=15)):
      if((row[0]>=ei_prompt_low) & (row[0]<=ei_prompt_high)&(row[3]!=15)&(row[1]!=15)):
        hist2d4.Fill(T_Diff_Corrected[0])
        hist2d5.Fill(y_test_np[i])
    tree.Fill()

count_dict = {}
sum_dict = {}

# for i in tqdm(range(num_entries), desc='Calculating Histograms'):
#     row = X_test.iloc[i]
#     i_val = row[1]
#     j_val = row[3]
#     key = (i_val, j_val)
    
#     # Calculate count and sum for (i, j) combination
#     count_dict[key] = count_dict.get(key, 0) + 1
#     sum_dict[key] = sum_dict.get(key, 0) + y_test_np[i]

# # Create the histograms using the count and sum values
# for i in tqdm(range(15), desc='Creating Histograms'):
#     for j in range(i + 1, 15):
#         key = (i, j)
#         count = count_dict.get(key, 0)
#         sum_val = sum_dict.get(key, 0)
        
#         if count != 0:
#             hist2d2.Fill(i, j, count)
#             hist2d3.Fill(i, j, sum_val / count)
#         else:
#             hist2d2.Fill(i, j, 0)
#             hist2d3.Fill(i, j, 0)
          
hist2d4.Write()
hist2d5.Write()
hist2d2.Write()
hist2d3.Write()
hist2d1.Write()
hist2d.Write()
tree.SetNotify(tree)

    

# Write the TTree to the ROOT file and close the file
tree.Write();
root_file.Close()
print("Processing completed successfully.")
