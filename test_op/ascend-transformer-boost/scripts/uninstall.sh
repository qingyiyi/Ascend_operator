#!/bin/bash
#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

set -e
VERSION=VERSION_PLACEHOLDER
LOG_PATH=LOG_PATH_PLACEHOLDER
LOG_NAME=LOG_NAME_PLACEHOLDER
MAX_LOG_SIZE=$((1024*1024*50))
CUR_DIR=$(dirname $(readlink -f $0))

function exit_solver() {
    exit_code=$?
    if [ ${exit_code} -ne 0 ];then
        print "ERROR" "Uninstall failed, [ERROR] ret code:${exit_code}"
        exit ${exit_code}
    fi
    exit 0
}

trap exit_solver EXIT

if [ "$UID" = "0" ]; then
    log_file=${LOG_PATH}${LOG_NAME}
else
    LOG_PATH="${HOME}${LOG_PATH}"
    log_file=${LOG_PATH}${LOG_NAME}
fi

# 将日志记录到日志文件
function log() {
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
        else
            echo -e "[cann-atb] [$(date +%Y%m%d-%H:%M:%S)] [$1] $2" >>$log_file
        fi
    fi
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

# 创建文件夹
function make_dir() {
    log "INFO" "mkdir ${1}"
    mkdir -p ${1} 2>/dev/null
    if [ $? -ne 0 ]; then
        print "ERROR" "create $1 failed !"
        exit 1
    fi
}

# 创建文件
function make_file() {
    log "INFO" "touch ${1}"
    touch ${1} 2>/dev/null
    if [ $? -ne 0 ]; then
        print "ERROR" "create $1 failed !"
        exit 1
    fi
}

## 日志模块初始化 ##
function log_init() {
    # 判断输入的日志保存路径是否存在，不存在就创建
    if [ ! -d "$LOG_PATH" ]; then
        make_dir "$LOG_PATH"
    fi
    chmod 750 "${LOG_PATH}"
    # 判断日志文件是否存在，如果不存在就创建；存在则判断是否大于50M
    if [ ! -f "$log_file" ]; then
        make_file "$log_file"
        # 安装日志权限
        chmod_recursion ${LOG_PATH} "750" "dir"
        chmod 640 "${log_file}"
    else
        local filesize=$(ls -l $log_file | awk '{ print $5}')
        local maxsize=$((1024*1024*50))
        if [ $filesize -gt $maxsize ]; then
            local log_base_name="${LOG_NAME%.*}"
            local log_extension="${LOG_NAME##*.}"
            if [ -n "${log_extension}" ]; then
                log_extension=".${log_extension}"
            fi
            local log_file_move_name="${log_base_name}_bak${log_extension}"
            mv -f "${log_file}" "${LOG_PATH}${log_file_move_name}"
            chmod 440 "${LOG_PATH}${log_file_move_name}"
            make_file "$log_file"
            log "INFO" "log file > 50M, move ${log_file} to ${LOG_PATH}${log_file_move_name}."
        fi
        chmod 640 "${log_file}"
    fi
    print "INFO" "Install log save in ${log_file}"
}

function chmod_recursion() {
    local parameter2=$2
    local rights="$(echo ${parameter2:0:2})""$(echo ${parameter2:1:1})"
    rights=$([ "${install_for_all_flag}" == "y" ] && echo ${rights} || echo $2)
    if [ "$3" = "dir" ]; then
        find $1 -type d -exec chmod ${rights} {} \; 2>/dev/null
    elif [ "$3" = "file" ]; then
        find $1 -type f -name "$4" -exec chmod ${rights} {} \; 2>/dev/null
    fi
}

function delete_file_with_authority() {
    file_path=$1
    dir_path=$(dirname ${file_path})
    if [ ${dir_path} != "." ];then
        dir_authority=$(stat -c %a ${dir_path})
        chmod 700 "${dir_path}"
        if [ -d ${file_path} ];then
            rm -rf "${file_path}"
        else
            rm -f "${file_path}"
        fi
        chmod ${dir_authority} "${dir_path}"
    else
        chmod 700 "${file_path}"
        if [ -d ${file_path} ];then
            rm -rf "${file_path}"
        else
            rm -f "${file_path}"
        fi
    fi
}

function delete_installed_files() {
    install_dir=$1
    csv_path=$install_dir/scripts/filelist.csv
    is_first_line=true
    cd "$install_dir"
    if [ ! -f $csv_path ];then
        print "INFO" "filelist.csv is not founded, uninstall by delete whole folder."
        [ -n "$1" ] && rm -rf "$1"
        return 0
    fi
    cat ${csv_path} | while read line
    do
        if [ ${is_first_line} == "false" ];then
            file_path=$(echo ${line} | awk '{print $1}')
            if [ ! -f ${file_path} ];then
                continue
            fi
            delete_file_with_authority $file_path
        fi
        is_first_line=false
    done
}

function delete_latest() {
    cd "$1/.."
    if [ -d "latest" -a $(readlink -f $1/../latest) == $1 ];then
        rm -f latest
    fi

    if [ -f "set_env.sh" ];then
        chmod 700 set_env.sh
        rm -f set_env.sh
    fi
}

function delete_empty_recursion() {
    if [ ! -d $1 ];then
        return 0
    fi
    for file in $1/*
    do
        if [ -d $file ];then
            delete_empty_recursion $file
        fi
    done
    if [ -z "$(ls -A $1)" ];then
        delete_file_with_authority $1
    fi
}

function uninstall_process() {
    #检查对应版本目录下的文件是否需要删除，是则进行删除
    if [ ! -d $1 ];then
        print "ERROR" "Ascend-cann-atb dir of $VERSION is not exist, uninstall failed!"
        return 0
    fi
    print "INFO" "Ascend-cann-atb uninstall $(basename $1) start!"
    atb_dir=$(cd "$1/..";pwd)
    delete_latest $1
    delete_installed_files $1
    if [ -d $1 ];then
        delete_empty_recursion $1
    fi
    if [ -z "$(ls $atb_dir)" ];then
        rm -rf "$atb_dir"
    fi
    print "INFO" "Ascend-cann-atb uninstall $(basename $1) success!"
}

function uninstall_torch_atb() {
    PACKAGE_NAME="torch_atb"
    if pip show $PACKAGE_NAME > /dev/null 2>&1; then
        print "INFO" "torch_atb is installed. need uninstall."
        pip uninstall -y torch_atb
    fi
}

install_path=$(cd "${CUR_DIR}/../../${VERSION}";pwd)
log_init
uninstall_process ${install_path}
uninstall_torch_atb
chmod 440 "${log_file}"