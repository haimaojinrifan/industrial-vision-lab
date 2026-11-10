// probe_radial.cpp —— 检查畸变模型是否"物理可用"
// 径向因子 g(r) = 1 + k1 r^2 + k2 r^4 + k3 r^6
//   判据1：g(r) 在 r in [0,1] 上没有拐点（方向一致）——桶形畸变本来就单调递减，递减是对的
//   判据2：g(r) > 0 —— 否则图像被"翻面"，物理上不可能
//   判据3：undistortPoints 在四角能收敛 —— 返回原值说明求解器发散了

#include <opencv2/opencv.hpp>

#include <iomanip>
#include <iostream>
#include <string>

static double gAt(double k1, double k2, double k3, double r) {
    const double r2 = r * r;
    return 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
}

int main(int argc, char** argv) {
    const char* yaml = argc > 1 ? argv[1] : "calib.yaml";
    cv::Mat K, dist;
    cv::FileStorage fs(yaml, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "cannot open " << yaml << "\n";
        return 1;
    }
    fs["camera_matrix"] >> K;
    fs["distortion_coefficients"] >> dist;
    fs.release();

    const double fx = K.at<double>(0, 0), fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2), cy = K.at<double>(1, 2);
    const double k1 = dist.at<double>(0), k2 = dist.at<double>(1), k3 = dist.at<double>(4);
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "k1=" << k1 << " k2=" << k2 << " k3=" << k3 << "\n\n";

    std::cout << "  r_norm    g(r)\n";
    for (int i = 0; i <= 16; ++i) {
        const double r = i * 0.05;
        const double g = gAt(k1, k2, k3, r);
        std::cout << std::setw(8) << r << "  " << std::setw(10) << g << "\n";
    }

    // 判据1 与 判据2：在 [0,1] 上细采样，看方向是否一致、是否穿零
    bool negative = false;
    int inc = 0, dec = 0;
    double prev = gAt(k1, k2, k3, 0.0);
    for (int i = 1; i <= 200; ++i) {
        const double g = gAt(k1, k2, k3, i * 0.005);
        if (g > prev + 1e-12) ++inc;
        else if (g < prev - 1e-12) ++dec;
        if (g <= 0) negative = true;
        prev = g;
    }
    const bool hasExtremum = (inc > 0 && dec > 0);
    std::cout << "\n[1] 方向一致(无拐点)? " << (hasExtremum ? "NO  <-- 有拐点，模型折叠"
                                                          : "yes（单调递减的桶形畸变是正常的）")
              << "\n    (递增步数 " << inc << " / 递减步数 " << dec << ")"
              << "\n[2] 出现 g(r)<=0? " << (negative ? "YES <-- 物理上不可能" : "no")
              << "\n\n";

    // 判据3：角落处 undistortPoints 是否真的收敛
    struct P { const char* n; cv::Point2f p; };
    const P pts[] = {{"TL", cv::Point2f(0.f, 0.f)},
                     {"TR", cv::Point2f(639.f, 0.f)},
                     {"BL", cv::Point2f(0.f, 479.f)},
                     {"BR", cv::Point2f(639.f, 479.f)},
                     {"CENTER", cv::Point2f(320.f, 240.f)}};
    std::cout << "[3] 角落反畸变\n";
    std::cout << "  corner   r_norm   g(r)      undistorted pixel        shift\n";
    bool diverged = false;
    for (int i = 0; i < 5; ++i) {
        const cv::Point2f& q = pts[i].p;
        const double dx = q.x - cx, dy = q.y - cy;
        const double rn = std::sqrt(dx * dx / (fx * fx) + dy * dy / (fy * fy));
        std::vector<cv::Point2f> one(1, q), und;
        cv::undistortPoints(one, und, K, dist);
        const cv::Point2f o((float)(fx * und[0].x + cx), (float)(fy * und[0].y + cy));
        const double shift = cv::norm(o - q);
        if (i < 4 && shift < 0.01) diverged = true;  // 一点没动 = 求解器吐回原值
        std::cout << std::setw(7) << pts[i].n << "  " << std::setw(7) << rn << "  "
                  << std::setw(9) << gAt(k1, k2, k3, rn) << "   (" << std::setw(8) << o.x
                  << "," << std::setw(8) << o.y << ")   " << std::setw(8) << shift << " px\n";
    }
    std::cout << "\n[3] 是否发散? " << (diverged ? "YES <-- 有角点位移为 0，求解器放弃了" : "no")
              << "\n";
    return 0;
}
