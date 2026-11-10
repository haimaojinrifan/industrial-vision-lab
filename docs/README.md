# 工业视觉检测与尺寸测量实验

C++17 / OpenCV 4.x 实现的工业视觉链路练习：**相机标定 → 畸变校正 → 光照归一化 → 缺陷检测 → 尺寸测量**。
每一步都给出可量化的误差，而不是只写"跑通了"。

```
day4_calib  ──►  day4_check / day4_straight  ──►  day5_detect  ──►  day6_measure
棋盘格标定        畸变校正是否可信                缺陷检测          像素 → 毫米
```

---

## 目录结构

```
cv_1/
├── CMakeLists.txt
├── cmake/copy_runtime_dlls.cmake   构建后自动把 OpenCV 运行库拷到输出目录
├── src/                            主线程序（每个对应一个阶段）
│   ├── day1_hello.cpp              环境自检
│   ├── day2_mat.cpp                cv::Mat 内存模型实验
│   ├── day3_cap.cpp                VideoCapture / VideoWriter 取流与帧率
│   ├── day4_calib.cpp              棋盘格标定（含 --synth 合成自检）
│   ├── day4_check.cpp              畸变校正的位移可视化
│   ├── day4_straight.cpp           直线度检验（判断标定能不能用于测量）
│   ├── day5_detect.cpp             缺陷检测（黑帽 + 顶帽 + 形态学）
│   └── day6_measure.cpp            尺寸测量（像素当量 + 矩形拟合）
├── tools/                          生成器 + 调试探针
│   ├── gen_workpiece.cpp           合成带缺陷的工件图
│   ├── gen_circle.cpp              合成圆点标定图
│   └── probe_*.cpp                 15 个一次性探针，每个对应一次问题定位
├── data/
│   ├── calib_final.yaml            定稿的相机内参
│   ├── calib_alt_simple.yaml       备选模型（k1,k2 两参数，未采用）
│   └── samples/                    真实采集样本（棋盘格 21 张）
└── docs/
    ├── README.md
    ├── CALIB.md                    标定的完整记录与已知风险
    └── NOTES.md                    尚未完全想清楚的问题
```

`tools/probe_*.cpp` 不是垃圾文件：每一次数值异常都是先写一个探针把中间量打出来才定位到的。它们记录了调试过程。

---

## 环境

| 项 | 版本 |
|---|---|
| 系统 | Windows 10 / 11 |
| 编译器 | MinGW-w64 g++ (x86_64)，`E:\mingw64\bin` |
| 构建 | CMake ≥ 3.21，生成器 `MinGW Makefiles` |
| OpenCV | 4.12.0，vcpkg `x64-mingw-dynamic` |

> 注意：vcpkg 自带的 Debug 版 OpenCV（`libopencv_*4d.dll`）与 Release 可执行文件混用会在 `cvtColor` 里崩溃。本项目用 `CMAKE_MAP_IMPORTED_CONFIG_DEBUG RELEASE` 强制 OpenCV 始终使用 Release 库，并把运行库直接拷到输出目录，不依赖 PATH。

## 构建

```powershell
cmake -S . -B build -G "MinGW Makefiles" `
      -DOpenCV_DIR=C:/Users/<you>/vcpkg/installed/x64-mingw-dynamic/share/opencv4 `
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

需要调试探针时加 `-DBUILD_PROBES=ON`（默认关闭，避免拖慢常规构建）。

---

## 快速验证（不需要相机）

全部在 `build` 目录下执行：

| 程序 | 命令 | 期望输出 |
|---|---|---|
| 环境自检 | `.\day1_hello.exe` | 弹出测试图案窗口，标题带 OpenCV 版本 |
| 标定自检 | `.\day4_calib.exe --synth` | 12/12 视角检出，`fx≈604`，`RMS≈0.12 px` |
| 缺陷检测 | `.\gen_workpiece.exe workpiece.png`<br>`.\day5_detect.exe workpiece.png` | 画出 5 个框（4 暗 + 1 亮），0 误检 |
| 尺寸测量 | `.\day6_measure.exe --self-rect` | `long 601.5 / short 380.0`，`err +0.18 %` |
| 真实标定复算 | `.\day4_calib.exe --solve 25 ..\data\samples\calib_imgs` | `fx≈550.9`，`RMS≈1.07 px` |

---

## 实测结果

### 1. 相机标定（`data/calib_final.yaml`）

| 参数 | 值 |
|---|---|
| fx / fy | 550.91 / 552.76 px |
| cx / cy | 360.94 / 217.12 px |
| k1, k2, k3 | −0.0897, −0.0869, 0.1180 |
| p1, p2 | 0.00044, −0.00045 |
| 重投影 RMS | **1.07 px** |
| 有效视角 | 21 张，棋盘格 spanX 26.7 % ~ 71.1 % |

自检：换成两参数模型（k1,k2）后 fx 只变 0.36 %（稳定）；直线度检验校正后弯曲量 0.317 px。
**已知风险**：主点偏离画面中心 40.9 px（6.4 % 画幅），测量前必须用标准件对标，详见 `docs/CALIB.md`。

### 2. 缺陷检测（`day5_detect`）

合成工件上 **5/5 命中、0 误检**（4 处暗缺陷 + 1 处亮点 + 1 条划痕）。

黑帽（BlackHat）负责暗缺陷，顶帽（TopHat）负责亮点，本质是把"绝对灰度"换成"相对邻域差"，从而消除 90 个灰阶的光照梯度。
弱对比测试（缺陷幅值降到 0.45×）：阈值 12 仍 4/4，阈值 16 掉到 3/4 —— **当前阈值离失效只有 4 个灰度级**，这是下一步要换成基于良品图统计的相对阈值的原因。

### 3. 尺寸测量（`day6_measure`）

| 场景 | 结果 |
|---|---|
| 合成矩形自检 | long 601.467 px（+1.467）、short 379.977 px（+1.613）；当量 0.142319 mm/px，短边预测误差 **+0.18 %** |
| 真实银行卡 | long 259.40 / short 134.58 px，ratio 1.927；短边预测 44.41 mm vs 真值 53.98 mm，误差 **−17.7 %** |

真实场景的 −17.7 % **不是算法误差，是相机不垂直**：卡片投影比例 1.927 而真卡是 1.586，反推镜头轴与卡片法线差约 35°，短边被透视压缩了 18 %。程序里的 `[geom]` 告警会直接拦下这种照片。

---

## 完成度

| 模块 | 状态 | 说明 |
|---|---|---|
| 环境与构建 | 完成 | 一键构建，运行库自动部署 |
| cv::Mat 内存模型 | 完成 | 浅拷贝 / ROI / 连续性的实测数据在 `day2_mat` 输出里 |
| 相机取流与曝光 | 完成 | DSHOW 640×480，手动曝光 −3（max 192 不削顶），8 fps |
| 相机标定 | 完成 | RMS 1.07 px，主点偏移属已知风险 |
| 缺陷检测 | 完成 | 5/5 命中 0 误检；阈值鲁棒性待改进 |
| 尺寸测量 | **未完成** | 合成数据 +0.18 %；真实数据受透视影响 −17.7 %，需四点单应校正 |
| 重复性统计 | **未完成** | 尚未取得可信的重复测量标准差 |
| Qt 界面 | 未开始 | 计划第 3 阶段 |

---

## 踩坑记录

1. **标准件在画面里占比太小，RMS 好看但标定是错的**
   20 张里只有 13 张检出，棋盘格 spanX 最大 46.9 %，从来没碰到画面边缘。重投影 RMS 0.3114 px 很低，但四角畸变位移在 0.94 ~ 14.48 px 之间乱跳（物理上不可能），换成 4 种畸变模型 fx 在 674 ~ 708 之间摆动 —— 典型的**欠约束**。

2. **模型会为了拟合而"折叠"**
   覆盖度提到 57 % 后，k3 解出 −19.99，径向函数 g(r)=1+k1r²+k2r⁴+k3r⁶ 在 r≈0.68 处变负，角落 `undistortPoints` 直接不收敛、返回原值。**"角落位移 0.00 px" 不是没有畸变，是求解器放弃了。**

3. **距离没有变化时，fx 可以错 3 倍**
   19 张全部拍在 26~28 cm，fx 解出 359.5（真实约 550）。透视投影 u = fx·X/Z 中 fx 与 Z 只以比值出现，距离恒定时两者可以同比缩放而残差几乎不变。所以标定必须**远近都拍**。

4. **`std` 接近 0 不是精度高，是在量同一个东西**
   批量测量 10 张，`w_px mean=1279 std=0.0001`。1280×720 的画面，1279×719 正好是"图像边框"—— 分割把背景当成了工件，而边框在每张图里像素级一致，方差自然是 0。

5. **`RotatedRect` 只交换 width/height 会画出旋转 90° 的框**
   `size` 和 `angle` 是一组。只把 `size.width/height` 换个位置而不给 `angle` 加 90°，`points()` 给出的四个角点是错的 —— 数字全对，框是歪的。画框必须用原始的 `RotatedRect`。

6. **最小外接矩形有 +1.5 px 的系统性外扩**
   同一批点，`minAreaRect` 比四点直线拟合稳定偏大 1.5 px，因为它是由最外侧那一个像素点决定的。

7. **实测到帧率被曝光时间卡死**
   DSHOW 下 `CAP_PROP_EXPOSURE = log2(秒)`：−4 → 1/16 s = 62.5 ms → 16 fps；−5 → 1/32 s → 30 fps 上限。帧间隔必须 ≥ 曝光时间，所以"想更亮"和"想更快"在硬件上直接冲突。

---

## 下一步

1. **四点单应校正**：用工件四角做 `getPerspectiveTransform` + `warpPerspective`，从数学上消除相机姿态影响（当前 −17.7 % 误差的根因）
2. **亚像素边缘拟合**：用四点直线拟合替代 `minAreaRect`，消掉 +1.5 px 外扩
3. **相对阈值**：缺陷阈值改为良品图响应的 99.9 分位数 ×3，替代硬编码的 12
4. **重复性统计**：固定相机与光照，同一工件重摆 30 次，给出标准差
5. **自建数据集** + Qt 界面

## 参考

- OpenCV 官方文档：Camera Calibration and 3D Reconstruction
- ISO/IEC 7810 ID-1：银行卡尺寸 85.60 × 53.98 mm（用作长度标准件）
