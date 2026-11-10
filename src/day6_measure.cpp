// day6_measure.cpp —— 像素当量标定 + 矩形工件尺寸测量 V0
//
// 用法:
//   day6_measure.exe --self-rect                         自检（合成 600 x 378.36 矩形）
//   day6_measure.exe --capture <dir> <n> [exposure]      用相机连拍 n 张（默认曝光 -3）
//   day6_measure.exe --single  <img> --mode 3            单张测量 + 可视化
//   day6_measure.exe --calib   <img> --mode 3            单张标定，输出 mm/px
//   day6_measure.exe --batch-fixed <dir> --mode 3 --scale <s>   批量重复性统计
//
// mode: 3 = Otsu 全局阈值（测大工件，默认）
//       0 = adaptiveThreshold（演示用）
//       1 = 黑帽+顶帽（测缺陷用）
//
// 约定：卡片 = 长 85.60 mm / 短 53.98 mm（ISO/IEC 7810 ID-1）
#include <opencv2/opencv.hpp>
#include <vector>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cmath>
#include <algorithm>

namespace fs = std::filesystem;

static const double REAL_LONG_MM  = 85.60;
static const double REAL_SHORT_MM = 53.98;
static const double EXPECT_RATIO  = REAL_LONG_MM / REAL_SHORT_MM;   // 1.5858

// ---------------------------------------------------------------- 分割 + 拟合

struct SegResult
{
    cv::RotatedRect rr;      // 长边一定是 rr.size.width
    cv::Mat         vis;     // 带红框的可视化图
    bool            ok = false;
};

// inv = true  : 暗目标 / 亮背景（蓝卡放白纸上）
// inv = false : 亮目标 / 暗背景
static bool segmentRect(const cv::Mat& src, SegResult& out, int mode, bool inv)
{
    cv::Mat gray, blur, bin;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);

    int tflag = inv ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY;

    if (mode == 0)
    {
        cv::adaptiveThreshold(blur, bin, 255, cv::ADAPTIVE_THRESH_MEAN_C, tflag, 31, 5);
    }
    else if (mode == 1)
    {
        cv::Mat blackhat, tophat, combine;
        cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
        cv::morphologyEx(gray, blackhat, cv::MORPH_BLACKHAT, k);
        cv::morphologyEx(gray, tophat, cv::MORPH_TOPHAT, k);
        cv::add(blackhat, tophat, combine);
        cv::threshold(combine, bin, 10, 255, cv::THRESH_BINARY);
    }
    else
    {
        // Otsu：阈值由直方图自动决定，能适应亮度变化
        cv::threshold(blur, bin, 0, 255, tflag | cv::THRESH_OTSU);
    }

    const double total = (double)src.total();
    const double fgPct = 100.0 * cv::countNonZero(bin) / total;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    double bestArea = 0;
    cv::RotatedRect bestRR;
    bool found = false;

    for (const auto& c : contours)
    {
        double area = cv::contourArea(c);
        if (area < 0.01 * total) continue;   // 太小：噪声
        if (area > 0.60 * total) continue;   // 太大：这是背景 / 画面边框

        // 形状闸门：卡片的已知长宽比是 1.586，细长条（阴影、桌缝）直接被踢掉
        cv::RotatedRect cand = cv::minAreaRect(c);
        double cw = std::max(cand.size.width, cand.size.height);
        double ch = std::min(cand.size.width, cand.size.height);
        if (ch <= 1.0) continue;
        double cratio = cw / ch;
        if (cratio < 1.15 || cratio > 2.60) continue;

        if (area > bestArea)
        {
            bestArea = area;
            bestRR = cand;
            found = true;
        }
    }

    std::cout << "[seg] mode=" << mode << "  fg=" << std::fixed << std::setprecision(1)
              << fgPct << "%  contours=" << contours.size();
    if (found)
        std::cout << "  accepted area=" << std::setprecision(0) << bestArea
                  << " (" << std::setprecision(1) << 100.0 * bestArea / total << "% of frame)";
    std::cout << "\n";

    if (!found)
    {
        if (fgPct > 60.0)
            std::cout << "[hint] foreground covers " << std::setprecision(0) << fgPct
                      << "% -> polarity looks INVERTED. ";
        if (fgPct > 60.0)
            std::cout << "try --bright-target\n";
        else
            std::cout << "[hint] nothing big enough: move the card fully inside the frame\n";
        return false;
    }

    // 画框必须用"原始"的 RotatedRect：直接交换 size.width/height 会让
    // points() 画出一个旋转 90° 的框（长宽对调，角度没跟着转）。
    cv::RotatedRect drawRR  = bestRR;
    double longPx  = std::max(bestRR.size.width, bestRR.size.height);
    double shortPx = std::min(bestRR.size.width, bestRR.size.height);

    out.rr = bestRR;
    out.rr.size.width  = (float)longPx;    // 只影响报告，不影响画框
    out.rr.size.height = (float)shortPx;

    out.vis = src.clone();
    if (out.vis.channels() == 1) cv::cvtColor(out.vis, out.vis, cv::COLOR_GRAY2BGR);

    cv::Point2f p[4];
    drawRR.points(p);
    for (int i = 0; i < 4; ++i)
        cv::line(out.vis, p[i], p[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);
    cv::putText(out.vis,
                cv::format("long %.1f  short %.1f", longPx, shortPx),
                cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

    out.ok = true;

    double gotRatio = out.rr.size.width / out.rr.size.height;
    if (std::fabs(gotRatio - EXPECT_RATIO) / EXPECT_RATIO > 0.03)
    {
        std::cout << "[geom] bbox ratio " << std::fixed << std::setprecision(3) << gotRatio
                  << " vs card " << EXPECT_RATIO << "  = "
                  << std::showpos << (gotRatio - EXPECT_RATIO) / EXPECT_RATIO * 100.0
                  << std::noshowpos << " %\n"
                     "       -> camera is NOT perpendicular to the card: the short edge is\n"
                     "          foreshortened. Re-shoot from straight above, or the short\n"
                     "          edge measurement is meaningless.\n";
    }

    return true;
}

// ---------------------------------------------------------------- 各子命令

static void reportRect(const std::string& tag, const SegResult& r, double scale = 0.0)
{
    double w = r.rr.size.width;
    double h = r.rr.size.height;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[" << tag << "] long=" << w << " px  short=" << h << " px"
              << "  ratio=" << std::setprecision(4) << w / h << "\n";

    if (scale > 0.0)
    {
        std::cout << std::setprecision(3)
                  << "        scale=" << std::setprecision(6) << scale << " mm/px"
                  << "  long=" << w * scale << " mm  short=" << h * scale << " mm\n";
    }
}

static void runSingle(const std::string& imgPath, int mode, bool inv)
{
    cv::Mat src = cv::imread(imgPath);
    if (src.empty()) { std::cerr << "read fail: " << imgPath << "\n"; return; }

    SegResult r;
    if (!segmentRect(src, r, mode, inv))
    {
        std::cerr << "no valid rectangle\n";
        return;
    }

    double scale   = REAL_LONG_MM / r.rr.size.width;
    double predH   = r.rr.size.height * scale;
    double errPct  = (predH - REAL_SHORT_MM) / REAL_SHORT_MM * 100.0;

    reportRect("SINGLE", r, scale);
    std::cout << std::setprecision(3)
              << "        short edge predicted = " << predH << " mm"
              << "  (truth " << REAL_SHORT_MM << ")  err = "
              << std::showpos << errPct << std::noshowpos << " %\n";

    cv::imwrite("measure_vis.png", r.vis);
    std::cout << "[vis] measure_vis.png\n";
}

static void runCalib(const std::string& imgPath, int mode, bool inv)
{
    cv::Mat src = cv::imread(imgPath);
    if (src.empty()) { std::cerr << "read fail: " << imgPath << "\n"; return; }

    SegResult r;
    if (!segmentRect(src, r, mode, inv))
    {
        std::cerr << "calib: no valid rectangle\n";
        return;
    }

    double scale  = REAL_LONG_MM / r.rr.size.width;
    double predH  = r.rr.size.height * scale;
    double errPct = (predH - REAL_SHORT_MM) / REAL_SHORT_MM * 100.0;

    reportRect("CALIB", r, scale);
    std::cout << std::setprecision(6)
              << "[CALIB RECT] scale = " << scale << " mm/px   (from long edge "
              << REAL_LONG_MM << " mm)\n";
    std::cout << std::setprecision(3)
              << "        short edge predicted = " << predH << " mm   err = "
              << std::showpos << errPct << std::noshowpos << " %\n";

    cv::imwrite("calib_vis.png", r.vis);
    std::cout << "[vis] calib_vis.png\n";
}

static void runBatch(const std::string& folder, int mode, bool inv, double scale)
{
    std::vector<double> wList, hList;
    int idx = 0, saved = 0;

    for (const auto& entry : fs::directory_iterator(folder))
    {
        auto p = entry.path();
        auto ext = p.extension().string();
        if (ext != ".png" && ext != ".jpg" && ext != ".jpeg") continue;

        cv::Mat src = cv::imread(p.string());
        if (src.empty()) continue;

        SegResult r;
        if (!segmentRect(src, r, mode, inv))
        {
            std::cout << "  skip " << p.filename().string() << " (no rect)\n";
            continue;
        }
        wList.push_back(r.rr.size.width);
        hList.push_back(r.rr.size.height);
        if (saved < 3)
        {
            cv::imwrite(cv::format("batch_vis_%02d.png", saved), r.vis);
            ++saved;
        }
        ++idx;
        if (idx <= 3)
            std::cout << "  ok " << p.filename().string()
                      << "  long=" << std::fixed << std::setprecision(2) << r.rr.size.width
                      << "  short=" << r.rr.size.height << "\n";
    }

    if (wList.size() < 2)
    {
        std::cerr << "need at least 2 valid images, got " << wList.size() << "\n";
        return;
    }

    auto stat = [](const std::vector<double>& v, double& mean, double& sd) {
        mean = 0; for (double x : v) mean += x; mean /= v.size();
        double s2 = 0; for (double x : v) s2 += (x - mean) * (x - mean);
        sd = std::sqrt(s2 / v.size());
    };

    double mw, sw, mh, sh;
    stat(wList, mw, sw);
    stat(hList, mh, sh);

    std::cout << "\n[RECT BATCH SUMMARY]  valid = " << wList.size() << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  long_px  mean = " << mw << "  std = " << sw
              << "  cv = " << (100.0 * sw / mw) << " %\n";
    std::cout << "  short_px mean = " << mh << "  std = " << sh
              << "  cv = " << (100.0 * sh / mh) << " %\n";

    if (scale > 0)
    {
        double predH = mh * scale;
        double err   = (predH - REAL_SHORT_MM) / REAL_SHORT_MM * 100.0;
        std::cout << "  scale = " << std::setprecision(6) << scale << " mm/px\n";
        std::cout << std::setprecision(3)
                  << "  long_mm  = " << mw * scale << "  +/- " << sw * scale << "\n"
                  << "  short_mm = " << mh * scale << "  +/- " << sh * scale
                  << "   (truth " << REAL_SHORT_MM << ", err "
                  << std::showpos << err << std::noshowpos << " %)\n";
    }

    if (sw < 1e-3 && sh < 1e-3)
        std::cout << "\n[warn] std is ~0 -> every image produced the SAME rectangle.\n"
                     "       That usually means the detector locked onto the image border.\n"
                     "       Open batch_vis_*.png and check the red box.\n";
}

static void runCapture(const std::string& folder, int n, double exposure)
{
    fs::create_directories(folder);

    cv::VideoCapture cap(0, cv::CAP_DSHOW);
    if (!cap.isOpened()) { std::cerr << "camera open fail\n"; return; }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0);      // 手动曝光
    cap.set(cv::CAP_PROP_EXPOSURE, exposure);

    // 前几帧常常还是旧参数，丢掉
    for (int i = 0; i < 10; ++i) { cv::Mat t; cap >> t; }

    for (int i = 0; i < n; ++i)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Scalar mu, sd;
        cv::meanStdDev(gray, mu, sd);
        double mx = 0;
        cv::minMaxLoc(gray, nullptr, &mx);

        std::string name = folder + "/cap_" + std::to_string(i) + ".png";
        cv::imwrite(name, frame);

        std::cout << "saved " << name
                  << "  " << frame.cols << "x" << frame.rows
                  << "  mean=" << std::fixed << std::setprecision(1) << mu[0]
                  << "  max=" << std::setprecision(0) << mx
                  << (mx >= 253 ? "   <-- CLIPPED, lower exposure" : "") << "\n";

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

// 合成一个精确已知尺寸的实心矩形，用来验证测量链路
static void runSelfRect()
{
    const double trueLong  = 600.0;
    const double trueShort = trueLong * REAL_SHORT_MM / REAL_LONG_MM;

    cv::Mat img(1000, 1000, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::RotatedRect rr(cv::Point2f(500, 500),
                       cv::Size2f((float)trueLong, (float)trueShort), 8.0f);
    cv::Point2f p[4]; rr.points(p);
    std::vector<cv::Point> ip;
    for (auto& q : p) ip.emplace_back(cvRound(q.x), cvRound(q.y));
    std::vector<std::vector<cv::Point>> polys{ ip };
    cv::fillPoly(img, polys, cv::Scalar(40, 40, 40), cv::LINE_AA);
    cv::imwrite("self_rect.png", img);

    SegResult r;
    if (!segmentRect(img, r, 3, true))
    {
        std::cout << "[SELF] FAILED to detect the synthetic rectangle\n";
        return;
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n==== SELF TEST ====\n";
    std::cout << "true  long  = " << trueLong  << " px\n";
    std::cout << "true  short = " << trueShort << " px\n";
    std::cout << "meas  long  = " << r.rr.size.width  << " px  (err "
              << std::showpos << r.rr.size.width - trueLong << std::noshowpos << ")\n";
    std::cout << "meas  short = " << r.rr.size.height << " px  (err "
              << std::showpos << r.rr.size.height - trueShort << std::noshowpos << ")\n";

    double scale = REAL_LONG_MM / r.rr.size.width;
    double predH = r.rr.size.height * scale;
    std::cout << "simulated calib: scale = " << std::setprecision(6) << scale
              << " mm/px, pred short = " << std::setprecision(3) << predH
              << " mm, err = " << std::showpos
              << (predH - REAL_SHORT_MM) / REAL_SHORT_MM * 100.0
              << std::noshowpos << " %\n";
}

static void showHelp()
{
    std::cout << R"(
usage:
  --self-rect                           synthetic self test
  --capture <dir> <n> [exposure]        grab n stills (default exposure -3)
  --single  <img> [--mode 3]            measure one image, writes measure_vis.png
  --calib   <img> [--mode 3]            print scale (mm/px), writes calib_vis.png
  --batch-fixed <dir> [--mode 3] [--scale s]   repeatability stats
options:
  --mode 0|1|3    0 adaptive  1 black/top hat  3 otsu (default, for a big card)
  --scale <mm/px>
  --bright-target target is BRIGHTER than background (default: darker)
)";
}

int main(int argc, char** argv)
{
    std::string action, imgPath, folderPath;
    int    mode    = 3;
    int    capN    = 0;
    double scale   = 0.0;
    double exposure = -3.0;
    bool   inv     = true;

    for (int i = 1; i < argc; ++i)
    {
        std::string s(argv[i]);
        if (s == "--single" && i + 1 < argc)      { action = "single"; imgPath = argv[++i]; }
        else if (s == "--calib" && i + 1 < argc)  { action = "calib";  imgPath = argv[++i]; }
        else if (s == "--batch-fixed" && i + 1 < argc) { action = "batch"; folderPath = argv[++i]; }
        else if (s == "--capture" && i + 2 < argc)
        {
            action = "capture"; folderPath = argv[++i]; capN = std::stoi(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') exposure = std::stod(argv[++i]);
        }
        else if (s == "--self-rect")              { action = "self"; }
        else if (s == "--mode" && i + 1 < argc)   { mode = std::stoi(argv[++i]); }
        else if (s == "--scale" && i + 1 < argc)  { scale = std::stod(argv[++i]); }
        else if (s == "--exposure" && i + 1 < argc) { exposure = std::stod(argv[++i]); }
        else if (s == "--bright-target")          { inv = false; }
    }

    if (action == "self")    { runSelfRect(); return 0; }
    if (action == "capture") { runCapture(folderPath, capN, exposure); return 0; }
    if (action == "single")  { runSingle(imgPath, mode, inv); return 0; }
    if (action == "calib")   { runCalib(imgPath, mode, inv); return 0; }
    if (action == "batch")   { runBatch(folderPath, mode, inv, scale); return 0; }

    showHelp();
    return 0;
}
