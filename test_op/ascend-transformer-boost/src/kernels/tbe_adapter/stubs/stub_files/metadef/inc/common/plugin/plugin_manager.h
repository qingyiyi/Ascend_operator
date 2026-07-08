/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// update this file with metadef, only the include guard macro name is different
#ifndef METADEF_COMMON_GE_PLUGIN_MANAGER_H_
#define METADEF_COMMON_GE_PLUGIN_MANAGER_H_

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>
#include <unordered_map>
#include <unordered_set>


#include "common/ge_common/debug/ge_log.h"
#include "mmpa/mmpa_api.h"

namespace metadef {
class DNNEngine;
class PluginManager {
 public:
  class InvokeFuncPerfRecorder {
   public:
    InvokeFuncPerfRecorder(const std::string &func_name, const std::string &lib_name)
        : func_name_(func_name), lib_name_(lib_name) {
      start_ = std::chrono::high_resolution_clock::now();
    }
    ~InvokeFuncPerfRecorder() {
      const auto end = std::chrono::high_resolution_clock::now();
      GELOGI("[GEPERFTRACE] The time cost of InvokeAll [%s] in [%s] is [%lu] micro seconds.", func_name_.c_str(),
             lib_name_.c_str(), std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count());
    }
   private:
    std::chrono::high_resolution_clock::time_point start_;
    const std::string &func_name_;
    const std::string &lib_name_;
  };
  PluginManager() = default;

  ~PluginManager();

  static void SplitPath(const std::string &mutil_path, std::vector<std::string> &path_vec, const char sep = ':');

  static ge::Status GetOppPath(std::string &opp_path);

  static ge::Status GetUpgradedOppPath(std::string &opp_path);

  static bool IsNewOppPathStruct(const std::string &opp_path);

  static ge::Status GetOppPluginVendors(const std::string &vendors_config, std::vector<std::string> &vendors);

  static ge::Status ReversePathString(std::string &path_str);

  static void GetPluginPathFromCustomOppPath(const std::string &sub_path, std::string &plugin_path);

  static void SetCustomOpLibPath(const std::string &custom_op_Lib_path);

  static ge::Status GetOppPluginPathOld(const std::string &opp_path,
                                    const std::string &path_fmt,
                                    std::string &plugin_path,
                                    const std::string &path_fmt_custom = "");

  static ge::Status GetOppPluginPathNew(const std::string &opp_path,
                                    const std::string &path_fmt,
                                    std::string &plugin_path,
                                    const std::string &old_custom_path,
                                    const std::string &path_fmt_custom = "");

  static bool IsSplitOpp();
  
  static ge::Status GetOpsProtoPath(std::string &opsproto_path);

  static ge::Status GetUpgradedOpsProtoPath(std::string &opsproto_path);

  static ge::Status GetUpgradedOpMasterPath(std::string &op_tiling_path);

  static ge::Status GetCustomOpPath(const std::string &fmk_type, std::string &customop_path);

  static ge::Status GetCustomCaffeProtoPath(std::string &customcaffe_path);

  static ge::Status GetOpTilingPath(std::string &op_tiling_path);

  static ge::Status GetOpTilingForwardOrderPath(std::string &op_tiling_path);

  static ge::Status GetConstantFoldingOpsPath(const std::string &path_base, std::string &constant_folding_ops_path);

  ge::Status LoadSoWithFlags(const std::string &path, const int32_t flags,
      const std::vector<std::string> &func_check_list = std::vector<std::string>());

  ge::Status LoadSo(const std::string &path, const std::vector<std::string> &func_check_list = std::vector<std::string>());

  ge::Status Load(const std::string &path, const std::vector<std::string> &func_check_list = std::vector<std::string>());

  ge::Status LoadWithFlags(const std::string &path, const int32_t flags,
      const std::vector<std::string> &func_check_list = std::vector<std::string>());

  static void GetOppSupportedOsAndCpuType(
      std::unordered_map<std::string, std::unordered_set<std::string>> &opp_supported_os_cpu,
      std::string opp_path = "", std::string os_name = "", uint32_t layer = 0U);

  static void GetCurEnvPackageOsAndCpuType(std::string &host_env_os, std::string &host_env_cpu);

  static bool GetVersionFromPath(const std::string &file_path, std::string &version);

  static void GetFileListWithSuffix(const std::string &path, const std::string &so_suff,
                                    std::vector<std::string> &file_list);

  static bool IsVendorVersionValid(const std::string &opp_version, const std::string &compiler_version);

  static bool IsVendorVersionValid(const std::string &vendor_path);

  static void GetPackageSoPath(std::vector<std::string> &vendors);

  static bool GetVersionFromPathWithName(const std::string &file_path, std::string &version,
                                         const std::string version_name);
  static void FindSoFilesInCustomPassDirs(const std::string &directory,
                                          std::vector<std::string> &so_files);

  static bool IsEndWith(const std::string &path, const std::string &suff);

  static ge::Status GetOpMasterDeviceSoPath(std::string &op_master_device_path);

  static std::string GetSoPackageName(const std::string &path);

  static std::string GetOppPkgPath(const std::string &opp_built_in_path, const std::string &whole_pkg_path,
                                   const std::string &sub_pkg_path, const std::string &os_cpu_type, bool &is_sub_pkg);

  template <typename R, typename... Types>
  ge::Status GetAllFunctions(const std::string &func_name, std::map<std::string, std::function<R(Types... args)>> &funcs) {
    for (const auto &handle : handles_) {
      const auto real_fn = reinterpret_cast<R(*)(Types...)>(mmDlsym(handle.second, func_name.c_str()));
      if (real_fn == nullptr) {
        const char *error = mmDlerror();
        if (error == nullptr) {
          error = "";
        }
        GELOGW("Failed to get function %s in %s! errmsg:%s", func_name.c_str(), handle.first.c_str(), error);
        return ge::GE_PLGMGR_FUNC_NOT_EXIST;
      } else {
        funcs[handle.first] = real_fn;
      }
    }
    return ge::SUCCESS;
  }

  template <typename... Types>
  ge::Status InvokeAll(const std::string &func_name, const Types... args) {
    for (const auto &handle : handles_) {
      InvokeFuncPerfRecorder recorder(func_name, handle.first);
      // If the funcName is existed, signature of realFn can be casted to any type
      const auto real_fn = reinterpret_cast<void (*)(Types...)>(mmDlsym(handle.second, func_name.c_str()));
      if (real_fn == nullptr) {
        const char *error = mmDlerror();
        if (error == nullptr) {
          error = "";
        }
        GELOGW("Failed to invoke function %s in %s! errmsg:%s", func_name.c_str(), handle.first.c_str(), error);
        return ge::GE_PLGMGR_INVOKE_FAILED;
      } else {
        real_fn(args...);
      }
    }
    return ge::SUCCESS;
  }

  template <typename T>
  ge::Status InvokeAll(const std::string &func_name, const T arg) {
    for (const auto &handle : handles_) {
      // If the funcName is existed, signature of realFn can be casted to any type
      InvokeFuncPerfRecorder recorder(func_name, handle.first);
      const auto real_fn = reinterpret_cast<void (*)(T)>(mmDlsym(handle.second, func_name.c_str()));
      if (real_fn == nullptr) {
        const ge::char_t *error = mmDlerror();
        if (error == nullptr) {
          error = "";
        }
        GELOGW("Failed to invoke function %s in %s! errmsg:%s", func_name.c_str(), handle.first.c_str(), error);
        return ge::GE_PLGMGR_INVOKE_FAILED;
      }
      typename std::remove_reference<T>::type arg_temp;
      real_fn(arg_temp);

      if (std::is_same<typename std::remove_reference<T>::type,
                       std::map<std::string, std::shared_ptr<DNNEngine>>>::value) {
        for (const auto &val : arg_temp) {
          if (arg.find(val.first) != arg.end()) {
            GELOGW("FuncName %s in so %s find the same key: %s, will replace it", func_name.c_str(),
                   handle.first.c_str(), val.first.c_str());
            arg[val.first] = val.second;
          }
        }
      }
      arg.insert(arg_temp.begin(), arg_temp.end());
    }
    return ge::SUCCESS;
  }

  template <typename... Args>
  void OptionalInvokeAll(const std::string &func_name, const Args... args) const {
    for (const auto &handle : handles_) {
      // If the funcName is existed, signature of realFn can be casted to any type
      InvokeFuncPerfRecorder recorder(func_name, handle.first);
      const auto real_fn = reinterpret_cast<void (*)(Args...)>(mmDlsym(handle.second, func_name.c_str()));
      if (real_fn == nullptr) {
        GELOGI("func %s does not exist in so %s", func_name.c_str(), handle.first.c_str());
        continue;
      } else {
        GELOGI("func %s exists in so %s", func_name.c_str(), handle.first.c_str());
        real_fn(args...);
      }
    }
  }

  template <typename T1, typename T2>
  ge::Status InvokeAll(const std::string &func_name, const T1 arg) {
    for (const auto &handle : handles_) {
      // If the funcName is existed, signature of realFn can be casted to any type
      InvokeFuncPerfRecorder recorder(func_name, handle.first);
      const auto real_fn = reinterpret_cast<T2(*)(T1)>(mmDlsym(handle.second, func_name.c_str()));
      if (real_fn == nullptr) {
        const ge::char_t *error = mmDlerror();
        if (error == nullptr) {
          error = "";
        }
        GELOGW("Failed to invoke function %s in %s! errmsg:%s", func_name.c_str(), handle.first.c_str(), error);
        return ge::GE_PLGMGR_INVOKE_FAILED;
      } else {
        const T2 res = real_fn(arg);
        if (res != ge::SUCCESS) {
          return ge::FAILED;
        }
      }
    }
    return ge::SUCCESS;
  }

  template <typename T>
  ge::Status InvokeAll(const std::string &func_name) {
    for (const auto &handle : handles_) {
      // If the funcName is existed, signature of realFn can be casted to any type
      InvokeFuncPerfRecorder recorder(func_name, handle.first);
      const auto real_fn = reinterpret_cast<T(*)()>(mmDlsym(handle.second, func_name.c_str()));
      if (real_fn == nullptr) {
        const ge::char_t *error = mmDlerror();
        if (error == nullptr) {
          error = "";
        }
        GELOGW("Failed to invoke function %s in %s! errmsg:%s", func_name.c_str(), handle.first.c_str(), error);
        return ge::GE_PLGMGR_INVOKE_FAILED;
      } else {
        const T res = real_fn();
        if (res != ge::SUCCESS) {
          return ge::FAILED;
        }
      }
    }
    return ge::SUCCESS;
  }

 private:
  void ClearHandles_() noexcept;
  ge::Status ValidateSo(const std::string &file_path, const int64_t size_of_loaded_so, int64_t &file_size) const;
  static bool ParseVersion(std::string &line, std::string &version, const std::string version_name);
  static bool GetRequiredOppAbiVersion(std::vector<std::pair<uint32_t, uint32_t>> &required_opp_abi_version);
  static bool GetEffectiveVersion(const std::string &opp_version, uint32_t &effective_version);
  static bool CheckOppAndCompilerVersions(const std::string &opp_version, const std::string &compiler_version,
                                          const std::vector<std::pair<uint32_t, uint32_t>> &required_version);
  static void GetOppAndCompilerVersion(const std::string &vendor_path, std::string &opp_version,
                                       std::string &compiler_version);
  std::vector<std::string> so_list_;
  std::map<std::string, void *> handles_;
  static std::string custom_op_lib_path_;
};

inline std::string GetSoRealPathByAddr(void *func_ptr) {
  mmDlInfo dl_info{nullptr, nullptr, nullptr, nullptr, 0, 0, 0};
  if ((mmDladdr(func_ptr, &dl_info) != EN_OK) || (dl_info.dli_fname == nullptr)) {
    GELOGW("Failed to read the shared library file path! errmsg:%s", mmDlerror());
    return std::string();
  }

  if (strnlen(dl_info.dli_fname, MMPA_MAX_PATH) >= MMPA_MAX_PATH) {
    GELOGW("The shared library file path is too long!");
    return std::string();
  }

  ge::char_t path[MMPA_MAX_PATH] = {};
  if (mmRealPath(dl_info.dli_fname, &path[0], MMPA_MAX_PATH) != EN_OK) {
    constexpr size_t max_error_strlen = 128U;
    ge::char_t err_buf[max_error_strlen + 1U] = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], max_error_strlen);
    GELOGW("Failed to get realpath of %s, errmsg:%s", dl_info.dli_fname, err_msg);
    return std::string();
  }

  return std::string(path);
}

inline std::string GetModelPathByAddr(void *func_ptr) {
  std::string so_path = GetSoRealPathByAddr(func_ptr);
  if (so_path.empty()) {
    return so_path;
  }
  so_path = so_path.substr(0U, so_path.rfind('/') + 1U);
  return so_path;
}

inline std::string GetModelPath() {
  return GetModelPathByAddr(reinterpret_cast<void *>(&GetModelPath));
}
}  // namespace metadef

#ifdef __cplusplus
extern "C" {
#endif
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void SetMetadefPluginCustomOpLibPathForC(const char* custom_op_Lib_path);
#ifdef __cplusplus
}
#endif

#endif  // METADEF_COMMON_GE_PLUGIN_MANAGER_H_
