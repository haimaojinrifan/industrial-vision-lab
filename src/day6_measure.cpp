// day6_measure.cpp —— 像素当量标定 + 矩形工件尺寸测量 V0
//
// 用法:
//   day6_measure.exe --self-rect                         自检（合成 600 x 378.36 矩形）
//   day6_measure.exe --capture <dir> <n> [exposure]      用相机连拍 n 张（默认曝光 -3）
//   day6_measure.exe --single  <img> --mode 3            单张测量 + 可视化
//   day6_measure.exe --calib   <img> --mode 3            单张标定，输出 mm/px
//   day6_measure.exe --batch-fixed <dir> --mode 3 --scale <s>   批量重复性统计
//   day6_measure.exe --rectify <img> --mode 3            四点单应校正到 856x540 正视图
//   day6_measure.exe --self-homography                   单应链路自检（合成斜拍卡片）
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

// 单应校正的目标坐标系：卡片真实尺寸 × 10 -> 856 × 540 px（53.98 × 10 = 539.8，取整 540，
// 长宽比因此差 0.04 %，可以忽略）。校正后 1 px 恒等于 0.1 mm，当量由目标尺寸直接给出。
static const double RECT_PX_PER_MM = 10.0;
static const int    RECT_W = (int)std::lround(REAL_LONG_MM  * RECT_PX_PER_MM);   // 856
static const int    RECT_H = (int)std::lround(REAL_SHORT_MM * RECT_PX_PER_MM);   // 540

// ---------------------------------------------------------------- 分割 + 拟合

struct SegResult
{
    cv::RotatedRect rr;      // 长边一定是 rr.size.width
    cv::Mat         vis;     // 带红框的可视化图
    bool            ok = false;
};

// 二值化：segmentRect / segmentQuad 共用。返回前景占比（%）
static double binarize(const cv::Mat& src, cv::Mat& bin, int mode, bool inv)
{
    cv::Mat gray, blur;
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

    return 100.0 * cv::countNonZero(bin) / (double)src.total();
}

// inv = true  : 暗目标 / 亮背景（蓝卡放白纸上）
// inv = false : 亮目标 / 暗背景
static bool segmentRect(const cv::Mat& src, SegResult& out, int mode, bool inv)
{
    cv::Mat bin;
    const double total = (double)src.total();
    const double fgPct = binarize(src, bin, mode, inv);

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

// ---------------------------------------------------------------- 四点单应

struct QuadResult
{
    cv::Point2f pt[4];      // 左上 → 右上 → 右下 → 左下
    cv::Mat     vis;        // 带四点编号的可视化图
    bool        ok = false;
};

// 四个角点排序。atan2 只保证"绕质心一圈"的先后（图像坐标下方向固定），
// 起点是任意的；再把 x+y 最小的点（最靠左上）转到首位。
static void orderQuad(std::vector<cv::Point>& q)
{
    cv::Point2f c(0.f, 0.f);
    for (const auto& p : q) c += cv::Point2f(p);
    c *= 1.0f / (float)q.size();

    std::sort(q.begin(), q.end(), [&](const cv::Point& a, const cv::Point& b) {
        return std::atan2(a.y - c.y, a.x - c.x) < std::atan2(b.y - c.y, b.x - c.x);
    });

    auto it = std::min_element(q.begin(), q.end(),
                               [](const cv::Point& a, const cv::Point& b) {
                                   return (a.x + a.y) < (b.x + b.y);
                               });
    std::rotate(q.begin(), it, q.end());
}

// 多边形逼近取真实四角。没有 SVD 拟合，四角误差在 1 px 量级，
// 但足够让单应把"透视压缩"从测量误差里去掉。
static bool segmentQuad(const cv::Mat& src, QuadResult& out, int mode, bool inv)
{
    cv::Mat bin;
    const double total = (double)src.total();
    const double fgPct = binarize(src, bin, mode, inv);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    double bestArea = 0;
    std::vector<cv::Point> best;

    for (const auto& c : contours)
    {
        double area = cv::contourArea(c);
        if (area < 0.01 * total) continue;   // 太小：噪声
        if (area > 0.60 * total) continue;   // 太大：这是背景 / 画面边框

        std::vector<cv::Point> approx;
        double peri = cv::arcLength(c, true);
        cv::approxPolyDP(c, approx, 0.02 * peri, true);
        if (approx.size() != 4) continue;            // 不是四边形，跳过
        if (!cv::isContourConvex(approx)) continue;  // 凹的说明分割溢出了

        orderQuad(approx);

        // 透视会把长宽比从 1.586 拉开，闸门放宽到和 minAreaRect 那条一致
        double w = (cv::norm(approx[1] - approx[0]) + cv::norm(approx[2] - approx[3])) * 0.5;
        double h = (cv::norm(approx[2] - approx[1]) + cv::norm(approx[3] - approx[0])) * 0.5;
        if (std::min(w, h) <= 1.0) continue;
        double qr = std::max(w, h) / std::min(w, h);
        if (qr < 1.15 || qr > 2.60) continue;

        if (area > bestArea) { bestArea = area; best = approx; }
    }

    std::cout << "[quad] mode=" << mode << "  fg=" << std::fixed << std::setprecision(1)
              << fgPct << "%  contours=" << contours.size();
    if (best.empty())
    {
        std::cout << "\n[hint] no 4-point contour: keep the whole card inside the frame,\n"
                     "       or raise the contrast (the gate is area 1%~60%, aspect 1.15~2.60)\n";
        return false;
    }
    std::cout << "  area=" << std::setprecision(0) << bestArea << " ("
              << std::setprecision(1) << 100.0 * bestArea / total << "% of frame)\n";

    for (int i = 0; i < 4; ++i) out.pt[i] = cv::Point2f(best[i]);

    out.vis = src.clone();
    if (out.vis.channels() == 1) cv::cvtColor(out.vis, out.vis, cv::COLOR_GRAY2BGR);
    std::vector<cv::Point> poly(best.begin(), best.end());
    cv::polylines(out.vis, poly, true, cv::Scalar(0, 255, 0), 2);
    const char* name[4] = { "TL", "TR", "BR", "BL" };
    for (int i = 0; i < 4; ++i)
    {
        cv::circle(out.vis, best[i], 5, cv::Scalar(0, 0, 255), -1);
        cv::putText(out.vis, name[i], best[i] + cv::Point(8, -8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    }

    out.ok = true;
    return true;
}

// 检测 → 排序 → 单应。rect = 正视图（RECT_W × RECT_H），H = 像素→像素的 3x3 矩阵
//
// rotateSrc 由用户给（--card-portrait）：卡片竖着放时，检测到的"第一条边"其实是
// 卡片的短边，直接映射到 856 会把画面横向拉伸 1.586 倍。旋转一位源角点即可。
// 这件事不能自动判断：一个矩形转 90° 和它被斜着拍，外轮廓完全可能一模一样，
// 单靠四条边看不出区别（自检 --self-homography 的 extreme-yaw 就是这个例子）。
static bool rectifyImage(const cv::Mat& src, cv::Mat& rect, cv::Mat& H,
                         QuadResult& q, int mode, bool inv, bool rotateSrc = false)
{
    if (!segmentQuad(src, q, mode, inv)) return false;

    const int shift = rotateSrc ? 1 : 0;
    cv::Point2f src4[4];
    for (int i = 0; i < 4; ++i) src4[i] = q.pt[(i + shift) % 4];

    cv::Point2f dst4[4] = { {0, 0},
                            {(float)RECT_W, 0},
                            {(float)RECT_W, (float)RECT_H},
                            {0, (float)RECT_H} };

    H = cv::getPerspectiveTransform(src4, dst4);
    cv::warpPerspective(src, rect, H, cv::Size(RECT_W, RECT_H),
                        cv::INTER_LINEAR, cv::BORDER_REPLICATE);
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

// 单应校正：输出正视图，并把当量固定成 0.1 mm/px
static bool runRectify(const std::string& imgPath, int mode, bool inv, bool portrait)
{
    cv::Mat src = cv::imread(imgPath);
    if (src.empty()) { std::cerr << "read fail: " << imgPath << "\n"; return false; }

    QuadResult q;
    cv::Mat rect, H;
    if (!rectifyImage(src, rect, H, q, mode, inv, portrait))
    {
        std::cerr << "rectify: no valid quad\n";
        return false;
    }

    double e[4];                                    // top right bottom left
    for (int i = 0; i < 4; ++i) e[i] = cv::norm(q.pt[(i + 1) % 4] - q.pt[i]);

    const char* name[4] = { "TL", "TR", "BR", "BL" };
    std::cout << std::fixed;
    std::cout << "[quad] " << name[0] << " (" << std::setprecision(1) << q.pt[0].x
              << ", " << q.pt[0].y << ")  " << name[1] << " (" << q.pt[1].x
              << ", " << q.pt[1].y << ")  " << name[2] << " (" << q.pt[2].x
              << ", " << q.pt[2].y << ")  " << name[3] << " (" << q.pt[3].x
              << ", " << q.pt[3].y << ")\n";
    std::cout << std::setprecision(2)
              << "[quad] edges  top=" << e[0] << "  right=" << e[1]
              << "  bottom=" << e[2] << "  left=" << e[3] << " px\n"
              << std::setprecision(1)
              << "[quad] opposite edges differ by " << std::fabs(e[0] - e[2]) / e[0] * 100.0
              << " % (top/bottom) and " << std::fabs(e[1] - e[3]) / e[1] * 100.0
              << " % (left/right)  (non-zero = perspective, removed by the homography)\n";

    if (portrait)
        std::cout << "[note] --card-portrait: source corners rotated one step so the long\n"
                     "       physical edge lands on the 856 axis (result may be upside down)\n";
    else if (e[1] > e[0])
        std::cout << "[hint] the 856-axis edge is the SHORTER one: either the card is just\n"
                     "       tilted (the homography handles it) or it lies in portrait --\n"
                     "       the outline alone cannot tell those apart, so if the rectified\n"
                     "       image looks stretched, re-run with --card-portrait\n";

    std::cout << "[H] source px -> rectified px\n";
    for (int r = 0; r < 3; ++r)
        std::cout << cv::format("      [%10.5f %10.5f %10.5f]\n",
                                H.at<double>(r, 0), H.at<double>(r, 1), H.at<double>(r, 2));

    const double scale = REAL_LONG_MM / RECT_W;     // 目标尺寸是真实尺寸 ×10
    std::cout << "[warp] " << RECT_W << " x " << RECT_H << " px  =  "
              << std::setprecision(2) << REAL_LONG_MM << " x " << REAL_SHORT_MM
              << " mm  ->  " << std::setprecision(6) << scale
              << " mm/px  (constant everywhere, no calibration needed)\n";

    cv::imwrite("rectified.png", rect);
    cv::imwrite("rectify_vis.png", q.vis);
    std::cout << "[vis] rectify_vis.png  +  rectified.png\n";
    return true;
}

// 合成"斜拍"的卡片，验证 检测 → 排序 → 单应 → 当量 整条链路
static bool runSelfHomography()
{
    std::cout << "\n==== SELF TEST: homography ====\n";

    // 卡片坐标系（就是校正后的目标坐标系）：深色卡面 + 两个大小不同的白标记
    cv::Mat card(RECT_H, RECT_W, CV_8UC3, cv::Scalar(60, 60, 60));
    const cv::Rect markA(40, 40, 80, 80);                          // 大：靠左上
    const cv::Rect markB(RECT_W - 120, RECT_H - 120, 40, 40);      // 小：靠右下
    cv::rectangle(card, markA, cv::Scalar(230, 230, 230), -1);
    cv::rectangle(card, markB, cv::Scalar(230, 230, 230), -1);

    cv::Point2f cardCorner[4] = { {0, 0},
                                  {(float)RECT_W, 0},
                                  {(float)RECT_W, (float)RECT_H},
                                  {0, (float)RECT_H} };

    // 把卡片投到画布上：对应关系由 corr 给出（必须是顺时针一圈，否则是镜像）
    auto makeCanvas = [&](const cv::Point2f quad[4], const int corr[4]) {
        cv::Point2f src4[4];
        for (int i = 0; i < 4; ++i) src4[i] = cardCorner[corr[i]];
        cv::Mat H = cv::getPerspectiveTransform(src4, quad);
        cv::Mat canvas(750, 1000, CV_8UC3, cv::Scalar(245, 245, 245));
        cv::warpPerspective(card, canvas, H, canvas.size(), cv::INTER_LINEAR,
                            cv::BORDER_CONSTANT, cv::Scalar(245, 245, 245));
        cv::GaussianBlur(canvas, canvas, cv::Size(3, 3), 0);

        // 加一点零均值噪声，别让自检只在"完美合成图"上成立
        cv::Mat noise(canvas.size(), CV_16SC3);
        cv::randn(noise, 0, 5);
        cv::Mat f;
        canvas.convertTo(f, CV_16SC3);
        f += noise;
        f.convertTo(canvas, CV_8UC3);
        return canvas;
    };

    const cv::Point2f markAc = { markA.x + markA.width * 0.5f, markA.y + markA.height * 0.5f };
    const cv::Point2f markBc = { markB.x + markB.width * 0.5f, markB.y + markB.height * 0.5f };
    const double trueDist  = cv::norm(markBc - markAc);
    const double trueAreaA = (double)markA.area();
    const double trueAreaB = (double)markB.area();

    auto runCase = [&](const char* tag, const cv::Point2f quad[4], const int corr[4],
                       bool portrait, bool expectUpright) {
        cv::Mat canvas = makeCanvas(quad, corr);
        cv::Mat rect, H;
        QuadResult q;
        std::cout << "\n-- " << tag << " --\n";
        if (!rectifyImage(canvas, rect, H, q, 3, true, portrait))
        {
            std::cout << "[FAIL] " << tag << ": quad not detected\n";
            return false;
        }

        bool ok = true;

        // 角点：和真值做无序最近点比较（顺序对不对由下面的标记位置单独判）
        double maxCornerErr = 0;
        for (int i = 0; i < 4; ++i)
        {
            double best = 1e9;
            for (int j = 0; j < 4; ++j)
                best = std::min(best, (double)cv::norm(q.pt[i] - quad[j]));
            maxCornerErr = std::max(maxCornerErr, best);
        }
        std::cout << std::fixed << std::setprecision(2)
                  << "  corner error (max) = " << maxCornerErr << " px\n";
        if (maxCornerErr > 3.0) { std::cout << "  [FAIL] corners off by > 3 px\n"; ok = false; }

        // 标记：两斑的面积和间距都是旋转不变的，被拉伸/镜像都会露馅
        cv::Mat gray, mark;
        cv::cvtColor(rect, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, mark, 160, 255, cv::THRESH_BINARY);

        cv::Mat lbl, stats, cent;
        int n = cv::connectedComponentsWithStats(mark, lbl, stats, cent, 8, CV_32S);
        struct Blob { double area; cv::Point2f c; };
        std::vector<Blob> blobs;
        for (int i = 1; i < n; ++i)
        {
            int x = stats.at<int>(i, cv::CC_STAT_LEFT);
            int y = stats.at<int>(i, cv::CC_STAT_TOP);
            int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
            int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            double a = (double)stats.at<int>(i, cv::CC_STAT_AREA);
            if (a < 600.0) continue;                       // 噪声
            if (x < 6 || y < 6 || x + w > RECT_W - 6 || y + h > RECT_H - 6)
                continue;                                  // 画面边框的反光
            blobs.push_back({ a, cv::Point2f((float)cent.at<double>(i, 0),
                                            (float)cent.at<double>(i, 1)) });
        }
        std::sort(blobs.begin(), blobs.end(),
                  [](const Blob& a, const Blob& b) { return a.area > b.area; });

        if (blobs.size() < 2)
        {
            std::cout << "  [FAIL] found " << blobs.size() << " marker(s), expected 2\n";
            ok = false;
        }
        else
        {
            double dist = cv::norm(blobs[1].c - blobs[0].c);
            std::cout << "  markers: area " << std::setprecision(0) << blobs[0].area
                      << " / " << blobs[1].area << " px2  (truth "
                      << trueAreaA << " / " << trueAreaB << ")\n"
                      << std::setprecision(2)
                      << "           distance " << dist << " px  (truth " << trueDist
                      << ", err " << std::showpos << (dist - trueDist) / trueDist * 100.0
                      << std::noshowpos << " %)\n";

            if (std::fabs(dist - trueDist) / trueDist > 0.02)
            {
                std::cout << "  [FAIL] marker distance off by > 2 % -> image is stretched\n";
                ok = false;
            }
            if (std::fabs(blobs[0].area - trueAreaA) / trueAreaA > 0.20)
            {
                std::cout << "  [FAIL] big marker area off by > 20 % -> image is stretched\n";
                ok = false;
            }

            if (expectUpright)
            {
                // 角点顺序错了（镜像/转 90°）会直接表现在标记位置上
                double eA = cv::norm(blobs[0].c - markAc);
                double eB = cv::norm(blobs[1].c - markBc);
                std::cout << "           marker offset " << eA << " / " << eB << " px\n";
                if (eA > 15.0 || eB > 15.0)
                {
                    std::cout << "  [FAIL] markers not at their original corners "
                                 "-> corner order is wrong\n";
                    ok = false;
                }
            }
        }

        std::cout << (ok ? "  [PASS] " : "  [FAIL] ") << tag << "\n";
        cv::imwrite(std::string("self_homography_") + tag + "_canvas.png", canvas);
        cv::imwrite(std::string("self_homography_") + tag + "_rect.png", rect);
        return ok;
    };

    // 1) 普通斜拍：卡片长边仍然朝画面水平方向
    const cv::Point2f shotTilt[4] = { {150, 120}, {830, 90}, {880, 640}, {110, 690} };
    // 2) 大偏航：透视把长边压到比短边还短。边长比已经无法判断朝向，
    //    只能靠单应把真实的长宽比找回来（这正是"不能自动判断竖拍"的证据）
    const cv::Point2f shotYaw[4]  = { {400, 80}, {700, 60}, {720, 660}, {380, 680} };
    // 3) 卡片竖着放在画面里：长边落在竖直方向，必须显式给 --card-portrait
    const cv::Point2f shotPort[4] = { {400, 80}, {720, 60}, {700, 660}, {380, 680} };
    const int corrIdentity[4]  = { 0, 1, 2, 3 };
    // 卡 TL -> 画面右上、TR -> 右下、BR -> 左下、BL -> 左上（顺时针一圈，不是镜像）
    const int corrRotated90[4] = { 3, 0, 1, 2 };

    bool ok = true;
    ok = runCase("tilt",        shotTilt, corrIdentity,  false, true) && ok;
    ok = runCase("extreme-yaw", shotYaw,  corrIdentity,  false, true) && ok;
    ok = runCase("portrait",    shotPort, corrRotated90, true,  true) && ok;

    std::cout << "\n[vis] self_homography_*_canvas.png / _rect.png\n";
    std::cout << (ok ? "[SELF] PASS\n" : "[SELF] FAIL\n");
    return ok;
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
  --self-rect                           synthetic self test (rect measure)
  --self-homography                     synthetic self test (quad + homography)
  --capture <dir> <n> [exposure]        grab n stills (default exposure -3)
  --single  <img> [--mode 3]            measure one image, writes measure_vis.png
  --calib   <img> [--mode 3]            print scale (mm/px), writes calib_vis.png
  --batch-fixed <dir> [--mode 3] [--scale s]   repeatability stats
  --rectify <img> [--mode 3]            warp the card to a 856x540 front view
                                        (10 px/mm), writes rectified.png
options:
  --mode 0|1|3    0 adaptive  1 black/top hat  3 otsu (default, for a big card)
  --scale <mm/px>
  --bright-target target is BRIGHTER than background (default: darker)
  --card-portrait the card is lying in portrait in the image (rotate the 4
                  source corners one step; needed when the rectified image
                  comes out stretched)
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
    bool   portrait = false;

    for (int i = 1; i < argc; ++i)
    {
        std::string s(argv[i]);
        if (s == "--single" && i + 1 < argc)      { action = "single"; imgPath = argv[++i]; }
        else if (s == "--calib" && i + 1 < argc)  { action = "calib";  imgPath = argv[++i]; }
        else if (s == "--batch-fixed" && i + 1 < argc) { action = "batch"; folderPath = argv[++i]; }
        else if (s == "--rectify" && i + 1 < argc) { action = "rectify"; imgPath = argv[++i]; }
        else if (s == "--capture" && i + 2 < argc)
        {
            action = "capture"; folderPath = argv[++i]; capN = std::stoi(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') exposure = std::stod(argv[++i]);
        }
        else if (s == "--self-rect")              { action = "self"; }
        else if (s == "--self-homography")        { action = "self-homography"; }
        else if (s == "--mode" && i + 1 < argc)   { mode = std::stoi(argv[++i]); }
        else if (s == "--scale" && i + 1 < argc)  { scale = std::stod(argv[++i]); }
        else if (s == "--exposure" && i + 1 < argc) { exposure = std::stod(argv[++i]); }
        else if (s == "--bright-target")          { inv = false; }
        else if (s == "--card-portrait")          { portrait = true; }
    }

    if (action == "self")    { runSelfRect(); return 0; }
    if (action == "self-homography") { return runSelfHomography() ? 0 : 1; }
    if (action == "capture") { runCapture(folderPath, capN, exposure); return 0; }
    if (action == "single")  { runSingle(imgPath, mode, inv); return 0; }
    if (action == "calib")   { runCalib(imgPath, mode, inv); return 0; }
    if (action == "batch")   { runBatch(folderPath, mode, inv, scale); return 0; }
    if (action == "rectify") { return runRectify(imgPath, mode, inv, portrait) ? 0 : 1; }

    showHelp();
    return 0;
}
