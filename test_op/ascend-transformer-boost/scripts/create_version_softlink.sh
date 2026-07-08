#!/bin/bash
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

set -e
script_dir=""
install_dir=""
version_dir=""
torch_atb_flag=""
LOG_PATH=LOG_PATH_PLACEHOLDER
LOG_NAME=LOG_NAME_PLACEHOLDER
MAX_LOG_SIZE=$((1024*1024*50))

if [ "$UID" = "0" ]; then
    log_file=${LOG_PATH}${LOG_NAME}
else
    LOG_PATH="${HOME}${LOG_PATH}"
    log_file=${LOG_PATH}${LOG_NAME}
fi

# 获取上n层目录
function updir()
{
    local dir=$1
    local n=$2
    for ((i=0; i<n; i++)); do
        dir=$(dirname "$dir")
    done
    printf '%s\n' "$dir"
}

# 将日志记录到日志文件并打屏
function print() {
    if [ ! -f "$log_file" ]; then
        if [ ! -d "${LOG_PATH}" ];then
            mkdir -p ${LOG_PATH}
        fi
        touch $log_file
    fi
    if [ x"$log_file" = x ]; then
        echo -e "[cann-atb] [$(date +%Y%m%d-%H:%M:%S)] [$1] $2"
    else
        if [ $(stat -c %s $log_file) -gt $MAX_LOG_SIZE ];then 
            echo -e "[cann-atb] [$(date +%Y%m%d-%H:%M:%S)] [$1] log file is bigger than $MAX_LOG_SIZE, stop write log to file"
            echo -e "[cann-atb] [$(date +%Y%m%d-%H:%M:%S)] [$1] $2"
        else
            echo -e "[cann-atb] [$(date +%Y%m%d-%H:%M:%S)] [$1] $2" | tee -a $log_file
        fi
    fi
}

function init()
{
    if script_path="$(readlink -f -- "${BASH_SOURCE[0]}" 2>/dev/null)"; then
        script_dir="$(dirname -- "$script_path")"
    else
        script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
    fi
    version_dir="$(updir "$script_dir" 2)"
    install_dir="$(updir "$version_dir" 1)"
}

function check_status()
{
    if pip3 show torch_atb > /dev/null 2>&1; then
        torch_atb_flag="1"
    fi
}

function remove_old_version()
{
    cd "${version_dir}"

    if [ -d "${version_dir}/latest" ]; then
        rm -f "${version_dir}/latest"
    fi

    if [ -f "${version_dir}/set_env.sh" ]; then
        chmod 700 "${version_dir}/set_env.sh"
        rm -f "${version_dir}/set_env.sh"
    fi
}

function switch_to_new_version()
{
    current_version_dir="$(updir "$script_dir" 1)"
    ln -s "${current_version_dir}" latest
    cp "${current_version_dir}/atb/set_env.sh" "${version_dir}"
    print "INFO" "Successfully switched latest symlink and set_env."

    py_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    py_major_version=${py_version%%.*}
    py_minor_version=${py_version##*.}
    wheel_file="torch_atb-0.0.1-cp${py_major_version}${py_minor_version}-none-any.whl"
    wheel_path="latest/whl/${wheel_file}"
    if [ -n "$torch_atb_flag" ]; then
        if pip3 install --force-reinstall "$wheel_path" > /dev/null 2>&1; then
            print "INFO" "torch_atb reinstall succeess."
        else
            print "ERROR" "torch_atb reinstallation failed!"
            exit 1
        fi
    fi
}

function main()
{
    init
    check_status
    remove_old_version
    switch_to_new_version
}

main