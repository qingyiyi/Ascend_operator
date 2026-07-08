################ 部分编译命令（需要全量编译一次后再运行）
################ 同时部分编译需要将test_op/ascend-transformer-boost/src/kernels/configs/build_config.json 文件中对应内容都改成false
# cd /root/var/test_op/ascend-transformer-boost/build

# cmake --build . --target ascendc_cpp_paged_attention_decoder_mask_ascend310p --parallel 1
# cmake --build . --target PagedAttentionOperation --parallel $(nproc)
# cmake --build . --target atb_mixops --parallel $(nproc)
# cmake --build . --target atb --parallel $(nproc)
# cmake --install . --verbose

# cd ..
# source output/atb/set_env.sh --cxx_abi=0


################ 全量编译
bash scripts/build.sh --clean-first --ascendc_dump
source output/atb/set_env.sh --cxx_abi=0


################ 退出dump模式全量编译
## 使用-DUSE_ASCENDC_DUMP=OFF选项没有用，需要重新构建整个项目才能彻底清理
# cd /root/var/test_op/ascend-transformer-boost

# rm -rf 3rdparty/mki
# rm -rf build
# rm -rf output

# bash scripts/build.sh testframework --clean-first --no-pybind --use_cxx11_abi=0
# source output/atb/set_env.sh --cxx_abi=0

