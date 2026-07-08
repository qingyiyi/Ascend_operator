import multiprocessing
import os
import shutil
import stat
import subprocess
from distutils.version import LooseVersion

import torch
import torch_npu
from setuptools import find_packages, setup
from setuptools.command.build_clib import build_clib
from setuptools.command.build_ext import build_ext
from torch_npu.utils.cpp_extension import NpuExtension


BUILD_PERMISSION = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def which(thefile):
    path = os.environ.get("PATH", os.defpath).split(os.pathsep)
    for directory in path:
        candidate = os.path.join(directory, thefile)
        if os.access(candidate, os.F_OK | os.X_OK) and not os.path.isdir(candidate):
            return candidate
    return None


def get_cmake_command():
    def get_version(cmd):
        for line in subprocess.check_output([cmd, "--version"]).decode("utf-8").split("\n"):
            if "version" in line:
                return LooseVersion(line.strip().split(" ")[2])
        raise RuntimeError("no cmake version found")

    cmake3 = which("cmake3")
    cmake = which("cmake")
    if cmake3 is not None and get_version(cmake3) >= LooseVersion("3.18.0"):
        return "cmake3"
    if cmake is not None and get_version(cmake) >= LooseVersion("3.18.0"):
        return "cmake"
    raise RuntimeError("no cmake or cmake3 with version >= 3.18.0 found")


class CPPLibBuild(build_clib):
    def run(self):
        cmake = get_cmake_command()
        build_py = self.get_finalized_command("build_py")
        package_dir = build_py.get_package_dir("paged_attention_mix_v3_ext")
        extension_dir = os.path.join(BASE_DIR, build_py.build_lib, package_dir)

        build_dir = os.path.join(BASE_DIR, "build")
        output_lib_path = os.path.join(build_dir, "lib")
        os.makedirs(output_lib_path, exist_ok=True)
        os.chmod(build_dir, mode=BUILD_PERMISSION)
        os.chmod(output_lib_path, mode=BUILD_PERMISSION)

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + os.path.realpath(output_lib_path),
            "-DTORCH_PATH=" + os.path.realpath(os.path.dirname(torch.__file__)),
            "-DTORCH_NPU_PATH=" + os.path.realpath(os.path.dirname(torch_npu.__file__)),
        ]
        cmake_args.append(
            "-DGLIBCXX_USE_CXX11_ABI=1"
            if torch.compiled_with_cxx11_abi()
            else "-DGLIBCXX_USE_CXX11_ABI=0"
        )

        subprocess.check_call([cmake, BASE_DIR] + cmake_args, cwd=build_dir, env=os.environ)
        subprocess.check_call(
            ["make", "-j", os.getenv("MAX_JOBS", str(multiprocessing.cpu_count()))],
            cwd=build_dir,
            env=os.environ,
        )

        dst_dir = os.path.join(extension_dir, "lib")
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        shutil.copytree(output_lib_path, dst_dir)


class Build(build_ext):
    def run(self):
        self.run_command("build_clib")
        self.build_lib = os.path.relpath(os.path.join(BASE_DIR, "build"))
        self.build_temp = os.path.relpath(os.path.join(BASE_DIR, "build", "temp"))
        self.library_dirs.append(os.path.relpath(os.path.join(BASE_DIR, "build", "lib")))
        super().run()


setup(
    name="paged_attention_mix_v3_ext",
    description="PyTorch wrapper for the custom PagedAttentionMixV3 Ascend operator",
    packages=find_packages(),
    ext_modules=[NpuExtension("paged_attention_mix_v3_ext._C", sources=[])],
    cmdclass={
        "build_clib": CPPLibBuild,
        "build_ext": Build,
    },
)