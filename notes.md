## working notes

```
enroot import docker://nvcr.io#nvidia/tensorflow:22.04-tf2-py3

enroot create --name tf2-ngc-22.04 nvidia+tensorflow+22.04-tf2-py3.sqsh
```

### Add libnvidia-container-tools to bundle

```
enroot start --root --rw tf2-ngc-22.04
```

```
DIST=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$DIST/libnvidia-container.list |

tee /etc/apt/sources.list.d/libnvidia-container.list

apt-get update
apt-get install --yes libnvidia-container-tools
```

### do test run of resnet50

enroot start with /home mounted ...

```
es tf2-ngc-22.04
```

This is from README

```
mpiexec --allow-run-as-root --bind-to socket -np 8 \
  python resnet.py --num_iter=400 --iter_unit=batch \
  --data_dir=/data/imagenet/train-val-tfrecord-480/ \
  --precision=fp16 --display_every=100
```

Single GPU command line,

```
python resnet.py --layers=50 --batch_size=64 --precision=fp32
```

```
(tf2-ngc-22.04)kinghorn@i9:/workspace/nvidia-examples/cnn$ CUDA_VISIBLE_DEVICES=1  python resnet.py  --batch_size=96 --precision=fp32
PY 3.8.10 (default, Mar 15 2022, 12:22:08)
[GCC 9.4.0]
TF 2.8.0
Script arguments:
  --image_width=224
  --image_height=224
  --distort_color=False
  --momentum=0.9
  --loss_scale=128.0
  --image_format=channels_last
  --data_dir=None
  --data_idx_dir=None
  --batch_size=96
  --num_iter=300
  --iter_unit=batch
  --log_dir=None
  --export_dir=None
  --tensorboard_dir=None
  --display_every=10
  --precision=fp32
  --dali_mode=None
  --use_xla=False
  --predict=False
  --dali_threads=4
2022-05-11 16:57:28.511029: I tensorflow/core/platform/cpu_feature_guard.cc:152] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE3 SSE4.1 SSE4.2 AVX
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-05-11 16:57:28.877114: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 9030 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 2080 Ti, pci bus id: 0000:65:00.0, compute capability: 7.5
2022-05-11 16:57:36.287518: I tensorflow/stream_executor/cuda/cuda_dnn.cc:379] Loaded cuDNN version 8400
global_step: 10 images_per_sec: 60.0
global_step: 20 images_per_sec: 259.6
global_step: 30 images_per_sec: 259.3
global_step: 40 images_per_sec: 256.8
global_step: 50 images_per_sec: 257.7
global_step: 60 images_per_sec: 257.4
global_step: 70 images_per_sec: 257.2
global_step: 80 images_per_sec: 255.6
global_step: 90 images_per_sec: 253.8
global_step: 100 images_per_sec: 255.8
global_step: 110 images_per_sec: 255.4
global_step: 120 images_per_sec: 255.7
global_step: 130 images_per_sec: 248.2
global_step: 140 images_per_sec: 255.5
global_step: 150 images_per_sec: 255.4
global_step: 160 images_per_sec: 255.8
global_step: 170 images_per_sec: 254.9
global_step: 180 images_per_sec: 254.6
global_step: 190 images_per_sec: 255.0
global_step: 200 images_per_sec: 253.3
global_step: 210 images_per_sec: 251.8
global_step: 220 images_per_sec: 251.0
global_step: 230 images_per_sec: 250.8
global_step: 240 images_per_sec: 251.2
global_step: 250 images_per_sec: 248.1
global_step: 260 images_per_sec: 252.0
global_step: 270 images_per_sec: 253.1
global_step: 280 images_per_sec: 249.8
global_step: 290 images_per_sec: 253.2
global_step: 300 images_per_sec: 253.8
epoch: 0 time_taken: 125.5
300/300 - 126s - loss: 8.8514 - top1: 0.8011 - top5: 0.8217 - 126s/epoch - 418ms/step
```

#### 2 x GPU:

```
mpiexec -np 2 python resnet.py --batch_size=96 --precision=fp32
```

```
(tf2-ngc-22.04)kinghorn@i9:/workspace/nvidia-examples/cnn$ mpiexec -np 2 python resnet.py --batch_size=96 --precision=fp32
PY 3.8.10 (default, Mar 15 2022, 12:22:08)
[GCC 9.4.0]
TF 2.8.0
Script arguments:
  --image_width=224
  --image_height=224
  --distort_color=False
  --momentum=0.9
  --loss_scale=128.0
  --image_format=channels_last
  --data_dir=None
  --data_idx_dir=None
  --batch_size=96
  --num_iter=300
  --iter_unit=batch
  --log_dir=None
  --export_dir=None
  --tensorboard_dir=None
  --display_every=10
  --precision=fp32
  --dali_mode=None
  --use_xla=False
  --predict=False
  --dali_threads=4
2022-05-11 17:04:04.011113: I tensorflow/core/platform/cpu_feature_guard.cc:152] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE3 SSE4.1 SSE4.2 AVX
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-05-11 17:04:04.048795: I tensorflow/core/platform/cpu_feature_guard.cc:152] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE3 SSE4.1 SSE4.2 AVX
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-05-11 17:04:04.444509: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 10550 MB memory:  -> device: 0, name: NVIDIA TITAN V, pci bus id: 0000:b3:00.0, compute capability: 7.0
2022-05-11 17:04:04.456197: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 9043 MB memory:  -> device: 1, name: NVIDIA GeForce RTX 2080 Ti, pci bus id: 0000:65:00.0, compute capability: 7.5
2022-05-11 17:04:12.182168: I tensorflow/stream_executor/cuda/cuda_dnn.cc:379] Loaded cuDNN version 8400
2022-05-11 17:04:12.420946: I tensorflow/stream_executor/cuda/cuda_dnn.cc:379] Loaded cuDNN version 8400
2022-05-11 17:04:13.931395: W tensorflow/core/common_runtime/bfc_allocator.cc:275] Allocator (GPU_0_bfc) ran out of memory trying to allocate 601.19MiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2022-05-11 17:04:13.931442: W tensorflow/core/common_runtime/bfc_allocator.cc:275] Allocator (GPU_0_bfc) ran out of memory trying to allocate 601.19MiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2022-05-11 17:04:13.931456: W tensorflow/core/common_runtime/bfc_allocator.cc:275] Allocator (GPU_0_bfc) ran out of memory trying to allocate 601.19MiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2022-05-11 17:04:14.196163: W tensorflow/core/common_runtime/bfc_allocator.cc:275] Allocator (GPU_0_bfc) ran out of memory trying to allocate 601.19MiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2022-05-11 17:04:14.196214: W tensorflow/core/common_runtime/bfc_allocator.cc:275] Allocator (GPU_0_bfc) ran out of memory trying to allocate 601.19MiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2022-05-11 17:04:14.196233: W tensorflow/core/common_runtime/bfc_allocator.cc:275] Allocator (GPU_0_bfc) ran out of memory trying to allocate 601.19MiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2022-05-11 17:04:18.029385: W tensorflow/core/common_runtime/bfc_allocator.cc:275] Allocator (GPU_0_bfc) ran out of memory trying to allocate 400.00MiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2022-05-11 17:04:18.055323: W tensorflow/core/common_runtime/bfc_allocator.cc:275] Allocator (GPU_0_bfc) ran out of memory trying to allocate 400.00MiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
global_step: 10 images_per_sec: 113.5
global_step: 20 images_per_sec: 526.2
global_step: 30 images_per_sec: 523.4
global_step: 40 images_per_sec: 518.1
global_step: 50 images_per_sec: 517.5
global_step: 60 images_per_sec: 522.5
global_step: 70 images_per_sec: 520.4
global_step: 80 images_per_sec: 516.1
global_step: 90 images_per_sec: 515.5
global_step: 100 images_per_sec: 511.0
global_step: 110 images_per_sec: 517.3
global_step: 120 images_per_sec: 518.9
global_step: 130 images_per_sec: 518.2
global_step: 140 images_per_sec: 515.4
global_step: 150 images_per_sec: 512.7
global_step: 160 images_per_sec: 516.2
global_step: 170 images_per_sec: 515.9
global_step: 180 images_per_sec: 514.3
global_step: 190 images_per_sec: 511.6
global_step: 200 images_per_sec: 511.5
global_step: 210 images_per_sec: 512.8
global_step: 220 images_per_sec: 511.7
global_step: 230 images_per_sec: 511.6
global_step: 240 images_per_sec: 512.0
global_step: 250 images_per_sec: 512.8
global_step: 260 images_per_sec: 511.2
global_step: 270 images_per_sec: 512.7
global_step: 280 images_per_sec: 511.0
global_step: 290 images_per_sec: 511.2
global_step: 300 images_per_sec: 510.3
epoch: 0 time_taken: 125.0
300/300 - 125s - loss: 8.5620 - top1: 0.8067 - top5: 0.8601 - 125s/epoch - 417ms/step
```

### running container direct

```
enroot start --env CUDA_VISIBLE_DEVICES=1  tf2-ngc-22.04 python nvidia-examples/cnn/resnet.py  --batch_size=96 --precision=fp32
```

### Create container bundle

```
enroot export tf2-ngc-22.04
enroot bundle tf2-ngc-22.04.sqsh
...
Self-extractable archive "/home/kinghorn/git/DL-bench/containers/tf2-ngc-22.04.run" successfully created.
```

## Query GPU metrics

Cool line to grab info every 5 seconds for logging.

```
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 5
```

`nvidia-smi --help-query-gpu` to see full list of options, LOTS of them!

## enroot

You can import container image and run it directly if you have
fuse-overlayfs
squashfuse

```
enroot import docker://ubuntu:22.04
enroot start ubuntu+22.04.sqsh
```

Create a bundle

```
enroot bundle -o u2204-1.run ubuntu+22.04.sqsh
```

can run the resulting .run file or do --keep

```
u2204-1.run --keep
```

that gives the extracted container directory by default ./u2204.1/

You can run that where it is with enroot

```
ENROOT_DATA_PATH=. enroot start u2204-1
```

Cool, but you have to have enroot installed locally.

### maybe make a portable runnable enroot???

- import ubuntu container image
- create a container from it
- install enroot and deps in that container
- export the container to an image
- make a bundle from the image
- portable-enroot.run ENROOT_DATA_PATH=. enroot start ./ubuntu ???
  IT WORKED!!
  had to cd to /mnt and run ENROOT_DATA_PATH=. enroot start u2204-1

```
./enubuntu.run --mount .:/mnt --rw
cd /mnt
ENROOT_DATA_PATH=. enroot start u2204-1

cat /etc/lsb-release
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=22.04
DISTRIB_CODENAME=jammy
DISTRIB_DESCRIPTION="Ubuntu 22.04 LTS"
```

So yes I can create an enroot.run container bundle and the run an extracted container with it.
