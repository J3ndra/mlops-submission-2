"""
This is the components TFX module
"""
import os
import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Tuner,
    Trainer,
    Evaluator,
    Pusher
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy)

from modules.transform_module import LABEL_KEY, transformed_feature


def init_components(
    data_dir,
    transform_module,
    training_module,
    tuner_module,
    training_steps,
    eval_steps,
    serving_model_dir,
):
    '''
    Initialize TFX components

    Args:
        data_dir: str, path to the data directory
        transform_module: str, path to the transform module
        training_module: str, path to the training module
        training_steps: int, number of training steps
        eval_steps: int, number of evaluation steps
        serving_model_dir: str, path to the serving model directory
    '''

    # Data Ingestion
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2)
        ]))

    example_gen = CsvExampleGen(
        input_base=data_dir,
        output_config=output
    )

    # Validation data
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs["examples"]
    )

    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"]
    )

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    # Processing Data
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=os.path.abspath(transform_module)
    )

    # Tuner
    tuner = Tuner(
        module_file=os.path.abspath(tuner_module),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(splits=['train'],
                                         num_steps=training_steps),
        eval_args=trainer_pb2.EvalArgs(splits=['eval'],
                                       num_steps=eval_steps)
    )

    # Training
    trainer = Trainer(
        module_file=os.path.abspath(training_module),
        hyperparameters=tuner.outputs['best_hyperparameters'],
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(
            splits=['train'],
            num_steps=training_steps),
        eval_args=trainer_pb2.EvalArgs(
            splits=['eval'],
            num_steps=eval_steps)
    )

    # Resolver
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('Latest_blessed_model_resolver')

    # Evaluation
    slicing_specs = [
        tfma.SlicingSpec(),
    ]

    metrics_specs = [
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(class_name='AUC'),
            tfma.MetricConfig(class_name='Precision'),
            tfma.MetricConfig(class_name='Recall'),
            tfma.MetricConfig(class_name='ExampleCount'),
            tfma.MetricConfig(class_name='BinaryAccuracy',
                              threshold=tfma.MetricThreshold(
                                  value_threshold=tfma.GenericValueThreshold(
                                      lower_bound={'value': 0.5}),
                                  change_threshold=tfma.GenericChangeThreshold(
                                      direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                      absolute={'value': 0.0001})
                              )
                              )
        ])
    ]

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="diagnosis_xf")],
        slicing_specs=[slicing_specs],
        metrics_specs=[metrics_specs])

    evaluator = Evaluator(
        examples=transform.outputs['transformed_examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config)

    # Pusher
    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )

    components = (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher
    )

    return components
