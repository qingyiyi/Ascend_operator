################ 部分编译命令
################ 同时部分编译需要将 test_op/ascend-transformer-boost/src/kernels/configs/mixkernels/op_list.yaml 和 test_op/ascend-transformer-boost/src/kernels/configs/build_config.json 这两个文件中对应内容都改成false
# cd /root/var/test_op/ascend-transformer-boost
# # rm -rf 3rdparty/mki
# rm -rf build
# rm -rf output
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



################ 退出dump模式编译命令，必须全量编译，适用于此前编译开启过dump模式，需要完全清理掉残留才可以关闭dump模式
## 使用-DUSE_ASCENDC_DUMP=OFF选项没有用，需要重新构建整个项目才能彻底清理
# cd /root/var/test_op/ascend-transformer-boost

# rm -rf 3rdparty/mki
# rm -rf build
# rm -rf output

# bash scripts/build.sh testframework --clean-first --no-pybind --use_cxx11_abi=0

# source output/atb/set_env.sh --cxx_abi=0

