FROM tensorflow/serving:latest

COPY ./outputs/serving_model /models/breast_cancer_wisconsin_model
COPY ./config /model_config

ENV MONITORING_CONFIG='/model_config/prometheus.config'
ENV MODEL_NAME=breast_cancer_wisconsin_model
# ENV PORT=8501
ENV MODEL_BASE_PATH=/models

RUN echo '#!/bin/bash \n\n\
env \n\
tensorflow_model_server --port=8500 --rest_api_port=${PORT} \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
--monitoring_config_file=${MONITORING_CONFIG} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh