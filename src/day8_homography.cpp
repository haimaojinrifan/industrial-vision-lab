// day8_homography.cpp —— 四点单应校正：把倾斜拍摄的平面工件映射回正面视图
//
// 用法:
//   day8_homography.exe <img 或 目录> [--canvas 1400x900] [--offset 300]
//
// 原理:
//   卡片真实尺寸 85.60 x 53.98 mm，放大 10 倍当作目标坐标系 -> 856 x 540 单位，
//   于是校正后 1 单位 = 0.1 mm，当量是常数。
//   注意：不能把目标画布就设成 856x540，那样会把卡片以外的物体全部裁掉。
//   这里把卡片映射到画布内的一块区域（默认从 (300,200) 开始），四周留出余量，
//   同一平面上比卡片大的物体也能被量到。
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <cmath>

namespace fs = std::filesystem;

static const double CARD_W_MM = 85.60;
static const double CARD_H_MM = 53.98;
static const double UNITS_PER_MM = 10.0;                 // 1 unit = 0.1 mm
static const int    CARD_W_U = (int)std::lround(CARD_W_MM * UNITS_PER_MM);  // 856
static const int    CARD_H_U = (int)std::lround(CARD_H_MM * UNITS_PER_MM);  // 540

// ---------------------------------------------------------------------------
// 找到卡片轮廓 -> 多边形逼近成 4 个角
// ---------------------------------------------------------------------------
static bool detectQuad(const cv::Mat& src, std::vector<cv::Point2f>& quad)
{
    cv::Mat gray, edges;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
    cv::Canny(gray, edges, 40, 110);

    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(edges, edges, k);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    const double total = (double)src.total();
    double bestScore = 0;
    std::vector<cv::Point> best;

    for (const auto& c : contours)
    {
        double area = cv::contourArea(c);
        if (area < 0.02 * total || area > 0.60 * total) continue;

        double peri = cv::arcLength(c, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(c, approx, 0.03 * peri, true);
        if (approx.size() != 4 || !cv::isContourConvex(approx)) continue;

        cv::Rect br = cv::boundingRect(approx);
        double rect = area / std::max(1.0, (double)br.width * br.height);
        if (rect < 0.60) continue;

        // 长宽比先验：卡片 1.586，A4 纸 1.414 —— 两者都可能通过，
        // 所以再乘上面积，让"占画面更大且更像矩形"的胜出
        double score = area * rect;
        if (score > bestScore) { bestScore = score; best = approx; }
    }

    if (best.empty()) return false;

    quad.clear();
    for (auto& p : best) quad.emplace_back((float)p.x, (float)p.y);
    return true;
}
// 四个角排序成 左上 / 右上 / 右下 / 左下（图像坐标 y 向下）
static void orderQuad(std::vector<cv::Point2f>& q)
{
    std::vector<cv::Point2f> out(4);
    // 左上：x+y 最小；右下：x+y 最大；右上：x-y 最大；左下：x-y 最小
    auto bySum  = [](const cv::Point2f& a, const cv::Point2f& b) { return a.x + a.y < b.x + b.y; };
    auto byDiff = [](const cv::Point2f& a, const cv::Point2f& b) { return a.x - a.y > b.x - b.y; };

    auto tl = std::min_element(q.begin(), q.end(), bySum);
    out[0] = *tl;
    auto br = std::max_element(q.begin(), q.end(), bySum);
    out[2] = *br;
    auto tr = std::max_element(q.begin(), q.end(), byDiff);
    out[1] = *tr;
    auto bl = std::min_element(q.begin(), q.end(), byDiff);
    out[3] = *bl;

    q = out;
}

// 在校正后的画布上找"卡片以外"的四边形物体
struct Measured
{
    double w_mm = 0, h_mm = 0;
    cv::RotatedRect rr;
    bool ok = false;
};

static std::vector<Measured> measureOthers(const cv::Mat& rectified, cv::Rect cardBox,
                                           cv::Mat& vis)
{
    std::vector<Measured> out;
    vis = rectified.clone();

    cv::Mat gray;
    cv::cvtColor(rectified, gray, cv::COLOR_BGR2GRAY);
    cv::Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);

    cv::Mat bin;
    cv::adaptiveThreshold(blur, bin, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                          cv::THRESH_BINARY_INV, 31, 5);

    // 铅笔线常常断成几段，先膨胀把它们连成一个闭合轮廓
    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(bin, bin, k, cv::Point(-1, -1), 2);

    // RETR_CCOMP: 只保留"有内孔"的闭合轮廓 —— 画出来的矩形框是空心环，
    // 而实心暗条、阴影带没有内孔，正好被排除。
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hier;
    cv::findContours(bin, contours, hier, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);

    const double total = (double)rectified.total();

    for (size_t ci = 0; ci < contours.size(); ++ci)
    {
        const auto& c = contours[ci];
        if (hier[ci][2] < 0) continue;            // 没有内孔 -> 不是空心框
        double area = cv::contourArea(c);
        if (area < 0.005 * total) continue;

        cv::Rect br = cv::boundingRect(c);
        // 跳过卡片本体（以及和它大幅重叠的轮廓）
        cv::Rect inter = br & cardBox;
        double interArea = (double)inter.area();
        if (interArea > 0.5 * br.area()) continue;
        if (interArea > 0.5 * cardBox.area()) continue;

        cv::RotatedRect rr = cv::minAreaRect(c);
        if (rr.size.height < 30) continue;        // 小于 3 mm 的不算目标
        double w = std::max(rr.size.width, rr.size.height);
        double h = std::min(rr.size.width, rr.size.height);
        if (h < 20) continue;

        Measured m;
        m.rr = rr;
        m.w_mm = w / UNITS_PER_MM;
        m.h_mm = h / UNITS_PER_MM;
        m.ok = true;
        out.push_back(m);

        cv::Point2f p[4];
        rr.points(p);
        for (int i = 0; i < 4; ++i)
            cv::line(vis, p[i], p[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);
        cv::putText(vis, cv::format("%.1f x %.1f mm", m.w_mm, m.h_mm),
                    cv::Point((int)rr.center.x - 60, (int)rr.center.y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    }

    std::sort(out.begin(), out.end(),
              [](const Measured& a, const Measured& b) { return a.w_mm * a.h_mm > b.w_mm * b.h_mm; });
    return out;
}

static void processOne(const std::string& path, int canvasW, int canvasH,
                       int ox, int oy, const std::string& tag)
{
    cv::Mat src = cv::imread(path);
    if (src.empty()) { std::cout << "read fail: " << path << "\n"; return; }

    std::vector<cv::Point2f> quad;
    if (!detectQuad(src, quad))
    {
        std::cout << "[" << tag << "] " << fs::path(path).filename().string()
                  << "  -> card quad NOT found\n";
        return;
    }
    orderQuad(quad);

    cv::Point2f dst4[4] = {
        {(float)ox,          (float)oy},
        {(float)(ox + CARD_W_U), (float)oy},
        {(float)(ox + CARD_W_U), (float)(oy + CARD_H_U)},
        {(float)ox,          (float)(oy + CARD_H_U)}
    };

    cv::Mat H = cv::getPerspectiveTransform(quad.data(), dst4);
    cv::Mat rectified;
    cv::warpPerspective(src, rectified, H, cv::Size(canvasW, canvasH));

    cv::Rect cardBox(ox, oy, CARD_W_U, CARD_H_U);
    cv::Mat vis;
    auto objs = measureOthers(rectified, cardBox, vis);

    cv::rectangle(vis, cardBox, cv::Scalar(0, 255, 0), 2);

    std::cout << "[" << tag << "] " << fs::path(path).filename().string() << "\n";
    std::cout << "        card corners: ";
    for (auto& p : quad) std::cout << "(" << std::setprecision(1) << p.x << "," << p.y << ") ";
    std::cout << "\n";

    // 卡片自身的比例：恒等于 856/540，只是证明"这是定义不是验证"
    std::cout << std::fixed << std::setprecision(4)
              << "        card ratio in rectified = " << (double)CARD_W_U / CARD_H_U
              << "  (by construction, NOT a check)\n";

    if (objs.empty())
    {
        std::cout << "        other objects: none found\n";
    }
    else
    {
        for (size_t i = 0; i < objs.size(); ++i)
            std::cout << std::fixed << std::setprecision(2)
                      << "        obj[" << i << "]  " << objs[i].w_mm << " x "
                      << objs[i].h_mm << " mm   ratio "
                      << (objs[i].w_mm / objs[i].h_mm) << "\n";
    }

    std::string stem = fs::path(path).stem().string();
    cv::imwrite("homo_vis_" + stem + ".png", vis);
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cout << "usage: day8_homography <img|dir> [--canvas WxH] [--offset X,Y]\n";
        return 0;
    }

    std::string target = argv[1];
    int cw = 2400, chh = 1400, ox = 200, oy = 200;

    for (int i = 2; i < argc; ++i)
    {
        std::string s(argv[i]);
        if (s == "--canvas" && i + 1 < argc)
        {
            std::string v(argv[++i]);
            auto pos = v.find('x');
            if (pos != std::string::npos) { cw = std::stoi(v.substr(0, pos)); chh = std::stoi(v.substr(pos + 1)); }
        }
        else if (s == "--offset" && i + 1 < argc)
        {
            std::string v(argv[++i]);
            auto pos = v.find(',');
            if (pos != std::string::npos) { ox = std::stoi(v.substr(0, pos)); oy = std::stoi(v.substr(pos + 1)); }
        }
    }

    if (fs::is_directory(target))
    {
        std::vector<std::string> files;
        for (const auto& e : fs::directory_iterator(target))
        {
            auto ext = e.path().extension().string();
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg")
                files.push_back(e.path().string());
        }
        std::sort(files.begin(), files.end());
        for (auto& f : files)
            processOne(f, cw, chh, ox, oy, "DIR");
    }
    else
    {
        processOne(target, cw, chh, ox, oy, "IMG");
    }
    return 0;
}
