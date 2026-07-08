
# compiling prograss
clear
cd ../custom_ops/PagedAttentionMixV3
bash run.sh
cd ..
cd ../test_op

# runtime prograss
#!/bin/bash
rm -rf test_attention_cache/
rm -rf ascend_work/
rm -rf printf
mkdir -p ascend_work
chmod -R 777 ascend_work

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/ascend-toolkit/8.3.RC2/opp/vendors/customize/bin/set_env.bash

source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/8.3.RC2/opp
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/ascend-toolkit/8.3.RC2/opp/vendors/customize
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/8.3.RC2/opp/vendors/customize/op_api/lib:${LD_LIBRARY_PATH}

# export ASCEND_LAUNCH_BLOCKING=1
# export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_WORK_PATH=/root/Ascend_operator/test_op/ascend_work

echo "ASCEND_WORK_PATH=${ASCEND_WORK_PATH}"


rm -rf profiler_attention/
ASCEND_RT_VISIBLE_DEVICES=1 PROFILE_TARGET=both PROFILE_REPEAT=10 python test_attention.py
# ASCEND_RT_VISIBLE_DEVICES=1 python test_attention.py
chmod -R 777 profiler_attention/

# PROFILE_TARGET=mix_v3 PROFILE_REPEAT=10 python test_attention.py
# PROFILE_TARGET=atb PROFILE_REPEAT=10 python test_attention.py
# PROFILE_TARGET=both PROFILE_REPEAT=10 python test_attention.py
# ASCEND_RT_VISIBLE_DEVICES=3 PROFILE_TARGET=both PROFILE_REPEAT=10 python test_attention.py





