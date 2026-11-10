// day4_straight.cpp — Day 4 直线度检验
//
// 用途：拿一把尺子 / 桌沿 / 门框边，斜穿画面拍一张照片，
//       量一下这条"本来笔直"的边，在 undistort 前后到底有多弯。
//
// 原理：畸变会把世界上的直线弯成曲线（一阶近似是抛物线）。
//       对一个"残差 ~ 沿边位置 t 的二次函数"做最小二乘拟合，
//       抛物线的矢高 sagitta = |c| * L^2 / 4 就是这条边的弯曲量。
//       校正得对，这个数应该掉到 2px 以下；校正得不对，它会一直留着，
//       并且直接变成 Day 6 尺寸测量的系统误差。
//
// 用法：day4_straight.exe <image> [calib.yaml]
// 输出：原始图 / 校正图各自的弯曲量，以及 straight_check.png

#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cstdio>

struct Straightness {
    bool ok = false;
    double sagitta = 0;  // 边的弯曲量（矢高，px）
    double bowRms = 0;   // 点到这条抛物线的均方根残差（px），太大小心不是干净边
    double len = 0;      // 参与拟合的边有多长（px）
    int    nPts = 0;     // 参与拟合的边缘点数
    cv::Point2f a, b;    // 用于显示的弦
};

// 在灰度图上找最长的一条近直线边缘，并量出它的弯曲量
static Straightness measure(const cv::Mat& gray)
{
    Straightness r;
    cv::Mat edges;
    double meanV = cv::mean(gray)[0];
    int minLen = (int)(0.20 * std::max(gray.cols, gray.rows));

    for (int attempt = 0; attempt < 3 && !r.ok; ++attempt) {
        double k = 1.0 / (1 << attempt);          // 逐步放宽阈值
        double lo = std::max(6.0, meanV * 0.40 * k);
        double hi = std::max(18.0, meanV * 1.00 * k);
        cv::Canny(gray, edges, lo, hi, 3);

        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(edges, lines, 1, CV_PI / 180.0, 40, minLen, 15);
        if (lines.empty()) continue;

        // 取最长的一段作为"主干"（只用来定方向，不用来定形状）
        size_t best = 0;
        double bestLen = 0;
        for (size_t i = 0; i < lines.size(); ++i) {
            double L = std::hypot((double)lines[i][2] - lines[i][0],
                                  (double)lines[i][3] - lines[i][1]);
            if (L > bestLen) { bestLen = L; best = i; }
        }
        cv::Point2f p0((float)lines[best][0], (float)lines[best][1]);
        cv::Point2f p1((float)lines[best][2], (float)lines[best][3]);
        cv::Point2f dir = p1 - p0;
        double n = std::hypot(dir.x, dir.y);
        if (n < 1) continue;
        dir *= (float)(1.0 / n);

        // 收集这条线附近的边缘点（带宽 12px，够把典型畸变弓形整个包进来）
        std::vector<double> T, D;
        for (int y = 0; y < edges.rows; ++y) {
            const uchar* row = edges.ptr<uchar>(y);
            for (int x = 0; x < edges.cols; ++x) {
                if (!row[x]) continue;
                double dx = x - p0.x, dy = y - p0.y;
                double d = dx * (-dir.y) + dy * dir.x;   // 垂直距离
                if (std::fabs(d) >= 12.0) continue;
                T.push_back(dx * dir.x + dy * dir.y);    // 沿边位置
                D.push_back(d);
            }
        }
        if (T.size() < 50) continue;

        // 迭代拟合 d = a + b*t + c*t^2，剔除离群点
        double a = 0, b = 0, c = 0;
        std::vector<size_t> idx(T.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;

        for (int it = 0; it < 3; ++it) {
            double S0 = 0, S1 = 0, S2 = 0, S3 = 0, S4 = 0;
            double R0 = 0, R1 = 0, R2 = 0;
            for (size_t k2 = 0; k2 < idx.size(); ++k2) {
                double t = T[idx[k2]], d = D[idx[k2]], t2 = t * t;
                S0 += 1; S1 += t; S2 += t2; S3 += t2 * t; S4 += t2 * t2;
                R0 += d; R1 += t * d; R2 += t2 * d;
            }
            if (S0 < 10) return r;
            cv::Mat A = (cv::Mat_<double>(3, 3) << S0, S1, S2,
                                                   S1, S2, S3,
                                                   S2, S3, S4);
            cv::Mat rhs = (cv::Mat_<double>(3, 1) << R0, R1, R2);
            cv::Mat x;
            if (!cv::solve(A, rhs, x, cv::DECOMP_SVD)) break;
            a = x.at<double>(0); b = x.at<double>(1); c = x.at<double>(2);

            if (it == 2) break;
            std::vector<size_t> next;
            for (size_t k2 = 0; k2 < idx.size(); ++k2) {
                double t = T[idx[k2]];
                if (std::fabs(D[idx[k2]] - (a + b * t + c * t * t)) < 1.5)
                    next.push_back(idx[k2]);
            }
            if (next.size() < 30) break;
            idx.swap(next);
        }

        double tmin = 1e9, tmax = -1e9, s2 = 0;
        for (size_t k2 = 0; k2 < idx.size(); ++k2) {
            double t = T[idx[k2]];
            double res = D[idx[k2]] - (a + b * t + c * t * t);
            s2 += res * res;
            tmin = std::min(tmin, t);
            tmax = std::max(tmax, t);
        }
        double L = tmax - tmin;
        if (L < 50) continue;

        r.ok = true;
        r.len = L;
        r.sagitta = std::fabs(c) * L * L / 4.0;   // 抛物线的矢高
        r.bowRms = std::sqrt(s2 / idx.size());
        r.nPts = (int)idx.size();
        r.a = p0 + dir * (float)tmin;
        r.b = p0 + dir * (float)tmax;
    }
    return r;
}

static void label(cv::Mat& img, const cv::Mat& src, const std::string& title,
                  const Straightness& s)
{
    if (src.channels() == 1) cv::cvtColor(src, img, cv::COLOR_GRAY2BGR);
    else img = src.clone();
    if (s.ok) {
        cv::line(img, s.a, s.b, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        cv::circle(img, s.a, 4, cv::Scalar(0, 255, 0), 1);
        cv::circle(img, s.b, 4, cv::Scalar(0, 255, 0), 1);
    }
    cv::putText(img, title, cv::Point(8, 24), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(img, title, cv::Point(8, 24), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    if (s.ok) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "bow %.2f px   residual %.2f px",
                      s.sagitta, s.bowRms);
        cv::putText(img, buf, cv::Point(8, 48), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(img, buf, cv::Point(8, 48), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    } else {
        cv::putText(img, "no straight edge found", cv::Point(8, 48),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "usage: day4_straight <image> [calib.yaml]\n"
                     "  拍一张长直边（尺子/桌沿/门框）斜穿画面的照片，\n"
                     "  边缘要清晰、和背景反差大、别过曝。\n";
        return 1;
    }
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty()) { std::cout << "cannot read " << argv[1] << "\n"; return 1; }

    std::string calib = (argc > 2) ? argv[2] : "calib.yaml";
    cv::FileStorage fs(calib, cv::FileStorage::READ);
    if (!fs.isOpened()) { std::cout << "cannot open " << calib << "\n"; return 1; }
    cv::Mat K, D;
    fs["camera_matrix"] >> K;
    fs["distortion_coefficients"] >> D;

    // ---- 分辨率 / 方向一致性检查 ----
    // 内参是"像素坐标系"里的量：换了分辨率就必须按比例缩放，换了方向就要交换，
    // 换了镜头则完全无效。不检查就会得到"看着挺像"的假结论。
    int w0 = 0, h0 = 0;
    fs["image_width"] >> w0;
    fs["image_height"] >> h0;
    if (w0 > 0 && h0 > 0 && (img.cols != w0 || img.rows != h0)) {
        double sx = (double)img.cols / w0, sy = (double)img.rows / h0;
        char msg[512];
        std::snprintf(msg, sizeof(msg),
            "\n[FATAL] image is %dx%d but the calibration (image_width/image_height) "
            "is %dx%d.\n"
            "        The K in that yaml only describes a %dx%d frame.\n"
            "        Either\n"
            "          (a) shoot at %dx%d with THE SAME camera, or\n"
            "          (b) pass a yaml that was calibrated at %dx%d.\n"
            "        Refusing to print numbers that would only look plausible.\n",
            img.cols, img.rows, w0, h0, w0, h0, w0, h0, img.cols, img.rows);
        std::cout << msg;
        if (std::fabs(sx - sy) > 0.02 * sx)
            std::cout << "[note] scale differs per axis (" << sx << " vs " << sy
                      << ") -> different aspect ratio, almost certainly a different "
                         "camera or a rotated shot.\n";
        return 2;
    }
    std::cout << "image: " << img.cols << "x" << img.rows
              << "  (matches calibration)\n";

    cv::Mat gray, und, undGray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::undistort(img, und, K, D);
    cv::cvtColor(und, undGray, cv::COLOR_BGR2GRAY);

    Straightness s0 = measure(gray);
    Straightness s1 = measure(undGray);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "calib: " << calib << "\n";
    std::cout << "fx=" << K.at<double>(0, 0) << "  fy=" << K.at<double>(1, 1)
              << "  cx=" << K.at<double>(0, 2) << "  cy=" << K.at<double>(1, 2) << "\n\n";
    std::cout << "                  bow(px)   resid   edge len   points\n";
    if (s0.ok)
        std::cout << "original       " << std::setw(8) << s0.sagitta << " "
                  << std::setw(7) << s0.bowRms << " " << std::setw(10) << s0.len
                  << " " << std::setw(8) << s0.nPts << "\n";
    else
        std::cout << "original       (no straight edge detected)\n";
    if (s1.ok)
        std::cout << "undistorted    " << std::setw(8) << s1.sagitta << " "
                  << std::setw(7) << s1.bowRms << " " << std::setw(10) << s1.len
                  << " " << std::setw(8) << s1.nPts << "\n";
    else
        std::cout << "undistorted    (no straight edge detected)\n";

    if (s0.ok && s1.ok) {
        double before = s0.sagitta, after = s1.sagitta;
        double diag = std::hypot((double)img.cols, (double)img.rows);
        std::cout << "\n[bow] " << before << " px -> " << after << " px\n";

        bool invalid = false;
        if (before < 5.0) {
            std::cout << "[check] raw bow is only " << before
                      << " px -> this edge stays in the low-distortion zone.\n"
                         "        Near the centre the correction is ~0, so this shot cannot\n"
                         "        tell a good model from a bad one. Re-shoot with the edge\n"
                         "        running CORNER TO CORNER (expect raw bow 10~25 px on this lens).\n";
            invalid = true;
        }
        if (s0.len < 0.60 * diag || s1.len < 0.60 * diag) {
            std::cout << "[check] edge shorter than 60% of the diagonal ("
                      << s0.len << " / " << s1.len << " of " << diag
                      << " px) -> not a full-frame test\n";
            invalid = true;
        }
        if (std::fabs(s0.len - s1.len) > 0.15 * std::max(s0.len, s1.len)) {
            std::cout << "[check] the two runs locked onto DIFFERENT edge segments ("
                      << s0.len << " vs " << s1.len
                      << " px) -> the two bow numbers are not comparable,\n"
                         "        do not use them to rank two calibrations.\n";
            invalid = true;
        }

        if (invalid)
            std::cout << "[verdict] INCONCLUSIVE  fix the shot (or the edge) and re-run\n";
        else if (after < 2.0)
            std::cout << "[verdict] PASS  residual bow " << after
                      << " px < 2 px, calibration is usable for measurement\n";
        else
            std::cout << "[verdict] FAIL  residual bow still " << after
                      << " px -> distortion model does not match the lens\n";
    }

    cv::Mat left, right, canvas;
    label(left, img, "ORIGINAL", s0);
    label(right, und, "UNDISTORTED", s1);
    if (left.size() != right.size()) cv::resize(right, right, left.size());
    cv::hconcat(left, right, canvas);
    cv::imwrite("straight_check.png", canvas);
    std::cout << "\n[save] straight_check.png\n";
    return 0;
}
