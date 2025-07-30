"""
This is the transform module for breast cancer data
"""
import tensorflow as tf
import tensorflow_transform as tft

# All features except 'id' and 'diagnosis' are numerical features
NUMERICAL_FEATURES = {
    'radius_mean',
    'texture_mean',
    'perimeter_mean',
    'area_mean',
    'smoothness_mean',
    'compactness_mean',
    'concavity_mean',
    'concave points_mean',
    'symmetry_mean',
    'fractal_dimension_mean',
    'radius_se',
    'texture_se',
    'perimeter_se',
    'area_se',
    'smoothness_se',
    'compactness_se',
    'concavity_se',
    'concave points_se',
    'symmetry_se',
    'fractal_dimension_se',
    'radius_worst',
    'texture_worst',
    'perimeter_worst',
    'area_worst',
    'smoothness_worst',
    'compactness_worst',
    'concavity_worst',
    'concave points_worst',
    'symmetry_worst',
    'fractal_dimension_worst'}

# The label is 'diagnosis' - M (Malignant) or B (Benign)
LABEL_KEY = 'diagnosis'


def transformed_feature(key):
    '''Rename transformed features'''
    return key + '_xf'


def preprocessing_fn(inputs):
    '''
    Preprocess input features into transformed features

    Args:
        inputs: map from feature keys to raw features.

    Return:
        outputs: map from feature keys to transformed features.
    '''
    outputs = {}

    # Process numerical features
    for feature in NUMERICAL_FEATURES:
        # Normalize numerical features using z-score normalization
        outputs[transformed_feature(feature)] = tft.scale_to_z_score(
            inputs[feature])

    # Create binary classification label: 1 for Malignant (M), 0 for Benign (B)
    outputs[transformed_feature(LABEL_KEY)] = tf.cast(
        tf.equal(inputs[LABEL_KEY], 'M'), tf.int64
    )

    return outputs
