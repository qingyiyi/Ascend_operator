rm -rf /root/atc_data/kernel_cache/*
unset ASCEND_CUSTOM_OPP_PATH

source /usr/local/Ascend/ascend-toolkit/set_env.sh

export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/8.3.RC2/opp
unset ASCEND_CUSTOM_OPP_PATH

cd /root/var/custom_ops/PagedAttentionMixV3
bash build.sh

./build_out/custom_opp_ubuntu_aarch64.run --quiet --install-path=/usr/local/Ascend/ascend-toolkit/8.3.RC2/opp

source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf /root/var/custom_ops/PagedAttentionMixV3/build_out
cd /root/var/custom_ops/PagedAttentionMixV3/torch_extension
rm -rf build paged_attention_mix_v3_ext/lib

python setup.py build_ext --force
