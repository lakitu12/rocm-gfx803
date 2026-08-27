# ROCm gfx803 (Polaris, RX 590 GME)

Port of **therock-7.14** (ROCm 7.14) to `gfx803` — `ChipID 0x6fdf`, `DoorbellType 1`, `RX 590 GME`.

Verified on `RX 590 GME` with `rocminfo 1.21`, `hipcc --offload-arch=gfx803`, `ComfyUI venv_gfx803` (SD1.5 512², 2 steps, `euler`).

## Hardware
- `AMD Radeon RX 590 GME` / `gfx803` / `36 CU` / `ChipID 0x6fdf` / `DoorbellType 1`
- `rocminfo` Runtime `1.21`

## What was ported
- **rocr-runtime** (trap_handler_gfx12.s rebuilt with `rocm714` LLVM23 at `/home/lakitu/rocm714/rocm/lib/llvm/bin/clang`)
  - `amd_aql_queue`: legacy doorbell `StoreRelaxed/Cas`, `ComputeRingBufferMin/Max`, `AllocRegisteredRingBuffer` keeps `AllocateDoubleMap` flag only (no `*2`), relies on `libhsakmt/fmm.c` 2× VA
  - `amd_gpu_agent`: allow `DoorbellType 0/1/2`, `IMAGE_*` fallback to `16384` on `hsa_amd_image_get_info_max_dim` failure (keeps `6.4 CLR` happy)
  - `amd_kfd_driver`: `AllocateDoubleMap → AQLQueueMemory` forwarding
- **clr (hipamd/rocclr)** rebuilt for `7.14` (`libamdhip64.so.7.14.60850-2b22ab01`, `libhiprtc`) with `amd_comgr 3.3`
- **HIP headers** synced `rocm714/include/hip → /usr/include/hip`, `/opt/rocm/include/hip` (fixes `__AMDGCN_WAVEFRONT_SIZE`)
- `rocBLAS` skipped — use `vkblas` (`col-major` / `NN` 3.68T) instead

## Install
```bash
# ROCR
cmake -DCMAKE_PREFIX_PATH=/home/lakitu/rocm714/rocm -DCMAKE_INSTALL_PREFIX=/opt/rocm -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc) && sudo make install
sudo cp rocr/lib/libhsa-runtime64.so.1.21.0 /opt/rocm/lib/ /opt/rocm-6.4.3/lib/ ~/rocm714/rocm/lib/
# CLR
cmake -DHIP_COMMON_DIR=../hip -DCMAKE_PREFIX_PATH=/home/lakitu/rocm714/rocm;/opt/rocm -DCMAKE_INSTALL_PREFIX=/opt/rocm -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIP_PLATFORM=amd -Damd_comgr_DIR=/home/lakitu/rocm714/rocm/lib/cmake/amd_comgr ..
LD_LIBRARY_PATH=/home/lakitu/rocm714/rocm/lib/rocm_sysdeps/lib:/home/lakitu/rocm714/rocm/lib make -j$(nproc) && sudo make install
```

## Verification
- `rocminfo | grep -A 5 "Agent 2"` → `gfx803`
- `hipcc --offload-arch=gfx803` → `hipGetDeviceCount 1`, `vecAdd 3 3 3 PASS` (both `/opt/rocm/bin/hipcc` 6.4 and `rocm714/bin/hipcc` 7.14)
- `hsa_qtest` → `hsa_queue_create size=64 → 0 SUCCESS`
- `ComfyUI venv_gfx803` (`torch 2.8.0+git8d1791e`) → `matmul 1024²`, `SDPA`, `conv 512²` PASS; `main.py` → `Kornia` fix (`@torch.jit.ignore` on `gfx803_linalg.py` `wrapped`), workflow `ckpt v1-5-pruned-emaonly-fp16 → KSampler 2 steps → VAEDecode` → `comfyui_test_gfx803_00001_.png` `324K` `success`

## Notes
- `comfy/venv_gfx803`: `hipfix.so` (`LD_PRELOAD`, `98` → `libtorch_hip.so` gfx803) + `gfx803_linalg.py` (`eig`/`eigvals`/`cholesky` → `numpy` fallback) remain required for full `torch` coverage (`nonzero`/`argwhere`/`masked_select` etc. via `librocsparse` fatbin gap)
- `llvm-21` cannot build `trap_handler_gfx12.s` (`hwreg TRAPSTS`), must use `rocm714` bundled `LLVM23`

Original base: `ROCm/rocm-systems` `therock-7.14` (`2b22ab01`).
