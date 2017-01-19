# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sklearn.cross_validation as cv
import numpy as np
import tempfile
from six.moves import urllib

import pandas as pd
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "", "Base directory for output models.")
flags.DEFINE_string("model_type", "wide_n_deep",
                    "Valid model types: {'wide', 'deep', 'wide_n_deep'}.")
flags.DEFINE_integer("train_steps", 200, "Number of training steps.")
flags.DEFINE_string(
    "train_data",
    "",
    "Path to the training data.")
flags.DEFINE_string(
    "test_data",
    "",
    "Path to the test data.")

LABEL_COLUMN = "overdue"
CATEGORICAL_COLUMNS = ['gender', 'pro', 'edu', 'marry', 'hukou']
CONTINUOUS_COLUMNS = ['salary_tag_sum', 'time_tradenum', 'income_num_', 'outcome_num_',
                      'volumn_min', 'volumn_max', 'volumn_mean', 'volumn_sum',
                      'bill_num_sum', 'bill_num_min', 'bill_num_mean', 'bill_num_std',
                      'last_pay_min', 'last_pay_max', 'last_pay_mean', 'last_pay_std',
                      'last_pay_lambda', 'cur_min_to_pay_min', 'cur_min_to_pay_max',
                      'cur_min_to_pay_mean', 'cur_min_to_pay_std', 'cur_bill_left_min',
                      'cur_bill_left_max', 'cur_bill_left_mean', 'cur_bill_left_std',
                      'cash_num_min', 'cash_num_max', 'cash_num_mean', 'cash_num_std',
                      'credit_max_max', 'credit_max_min', 'credit_max_mean',
                      'credit_max_std', 'cur_available_min', 'cur_available_max',
                      'cur_available_mean', 'cur_available_std', 'last_to_pay_min',
                      'last_to_pay_max', 'last_to_pay_mean', 'last_to_pay_count',
                      'last_to_pay_std', 'bank_id_count', 'bank_id_lambda',
                      'cur_to_pay_min', 'cur_to_pay_max', 'cur_to_pay_mean',
                      'cur_to_pay_std']


def build_estimator(model_dir):
    """Build an estimator."""
    # Sparse base columns.
    gender = tf.contrib.layers.sparse_column_with_integerized_feature(column_name="gender",
                                                                      bucket_size=3)
    pro = tf.contrib.layers.sparse_column_with_integerized_feature(
        "pro", bucket_size=5)
    edu = tf.contrib.layers.sparse_column_with_integerized_feature(
        "edu", bucket_size=5)
    marry = tf.contrib.layers.sparse_column_with_integerized_feature(
        "marry", bucket_size=6)
    hukou = tf.contrib.layers.sparse_column_with_integerized_feature(
        "hukou", bucket_size=5)

    # Continuous base columns.

    colums = []

    for cname in CONTINUOUS_COLUMNS:
        colums.append(tf.contrib.layers.real_valued_column(cname))

    # Transformations.
    # age_buckets = tf.contrib.layers.bucketized_column(age,
    #                                                   boundaries=[
    #                                                       18, 25, 30, 35, 40, 45,
    #                                                       50, 55, 60, 65
    #                                                   ])

    # Wide columns and deep columns.
    wide_columns = [gender, pro, edu, marry, hukou,
                    tf.contrib.layers.crossed_column([pro, edu],
                                                     hash_bucket_size=int(50)),
                    tf.contrib.layers.crossed_column([gender, hukou],
                                                     hash_bucket_size=int(50)),
                    tf.contrib.layers.crossed_column([gender, edu],
                                                     hash_bucket_size=int(50)),
                    tf.contrib.layers.crossed_column([marry, hukou],
                                                     hash_bucket_size=int(50))]
    deep_columns = [
        tf.contrib.layers.embedding_column(gender, dimension=3),
        tf.contrib.layers.embedding_column(pro, dimension=3),
        tf.contrib.layers.embedding_column(edu, dimension=3),
        tf.contrib.layers.embedding_column(marry, dimension=3),
        tf.contrib.layers.embedding_column(hukou, dimension=3)
    ]

    for c in colums:
        deep_columns.append(c),

    if FLAGS.model_type == "wide":
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                              feature_columns=wide_columns)
    elif FLAGS.model_type == "deep":
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                           feature_columns=deep_columns,
                                           hidden_units=[100, 50])
    else:
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[200,100, 50])
    return m


def input_fn(df):
    """Input builder function."""
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    # Converts the label column into a constant Tensor.
    if LABEL_COLUMN in df.columns:
        label = tf.constant(df[LABEL_COLUMN].values)
    else:
        label = ""
    # Returns the feature columns and the label.
    return feature_cols, label


def train_and_eval():
    """Train and evaluate the model."""
    train_file_name = "/Users/ericzhou.zc/Downloads/credit/train/input.txt"
    test_file_name = "/Users/ericzhou.zc/Downloads/credit/test/input.txt"
    df_train = pd.read_csv(train_file_name)
    df_test = pd.read_csv(test_file_name)
    df_train.columns = [n.replace('<','').replace('>','') for n in df_train.columns]
    df_test.columns = [n.replace('<', '').replace('>', '') for n in df_test.columns]

    global CONTINUOUS_COLUMNS
    tmp = CATEGORICAL_COLUMNS
    tmp.append('id')
    tmp.append('overdue')
    CONTINUOUS_COLUMNS = [x for x in df_train.columns if x not in tmp]
    CONTINUOUS_COLUMNS.remove('Unnamed: 0')
    CATEGORICAL_COLUMNS.remove('id')
    CATEGORICAL_COLUMNS.remove('overdue')
    print(CONTINUOUS_COLUMNS)
    print(CATEGORICAL_COLUMNS)

    X_train, X_test, y_train, y_test = cv.train_test_split(df_train, df_train[LABEL_COLUMN], test_size=0.33, random_state=42)
    pos = X_train.loc[X_train['overdue']==0]
    neg = X_train.loc[X_train['overdue'] == 1]
    # # final = pos.sample(frac = 0.3)
    # final = final.append(neg)
    # final = final.iloc[np.random.permutation(len(final))]
    # X_train = final
    # remove NaN elements
    # df_train = df_train.dropna(how='any', axis=0)
    # df_test = df_test.dropna(how='any', axis=0)

    # df_train[LABEL_COLUMN] = (
    #     df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    # df_test[LABEL_COLUMN] = (
    #     df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

    model_dir = "wnd_model_dir"
    print("model directory = %s" % model_dir)

    m = build_estimator(model_dir)
    m.fit(input_fn=lambda: input_fn(X_train), steps=FLAGS.train_steps)
    # m.export('dl_output_model')
    results = m.evaluate(input_fn=lambda: input_fn(X_test), steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))

def main(_):
    train_and_eval()


if __name__ == "__main__":
    tf.app.run()
