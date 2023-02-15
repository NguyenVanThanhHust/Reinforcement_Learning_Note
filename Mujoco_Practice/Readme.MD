https://mujoco.readthedocs.io/en/latest/overview.html

git config --global --add safe.directory /workspace/Reinforcement_Learning_Projects/mujoco/build/_deps/lodepng-src
git config --global --add safe.directory /workspace/Reinforcement_Learning_Projects/mujoco/build/_deps/qhull-src
git config --global --add safe.directory /workspace/Reinforcement_Learning_Projects/mujoco/build/_deps/tinyxml2-src
git config --global --add safe.directory /workspace/Reinforcement_Learning_Projects/mujoco/build/_deps/tinyobjloader-src
git config --global --add safe.directory /workspace/Reinforcement_Learning_Projects/mujoco/build/_deps/ccd-src
git config --global --add safe.directory /workspace/Reinforcement_Learning_Projects/mujoco/build/_deps/abseil-cpp-src
git config --global --add safe.directory /workspace/Reinforcement_Learning_Projects/mujoco/build/_deps/googletest-src
git config --global --add safe.directory /workspace/Reinforcement_Learning_Projects/mujoco/build/_deps/benchmark-src
git config --global --add safe.directory /workspace/Reinforcement_Learning_Projects/mujoco/build/_deps/eigen3-src
git config --global --add safe.directory /workspace/Reinforcement_Learning_Projects/mujoco/build/_deps/glfw-src


export MUJOCO_GL=osmesa
export MJLIB_PATH=/workspace/Reinforcement_Learning_Projects/mujoco/build/lib/libmujoco.so
export LD_LIBRARY_PATH=/workspace/Reinforcement_Learning_Projects/mujoco/build/lib/:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/Reinforcement_Learning_Projects/mujoco/build/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/nvidia-opengl/
