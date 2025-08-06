#!/bin/bash
set -e

echo "[1/3] build-riscv.sh"
sleep 1

cd ../../../ || { echo "[Fail to move /llamma.cpp]"; exit 1; }
if ./build-riscv.sh; then
        echo "[Complete build]"
else
        echo "[Fail to build llamma.cpp]"
        exit 1
fi
sleep 1

echo "[2/3] update_rootfs.sh"
sleep 1

cd ../../ || { echo "[Fail to move /deploy]"; exit 1; }
if ./update_rootfs.sh; then
        echo "[Complete update bin]"
else
        echo "[Fail to update bin]"
        exit 1
fi
sleep 1

echo "[3/3] launch_sim.sh"
sleep 1

./launch_sim.sh || { echo "[Fail to launch QEMU]"; exit 1; }