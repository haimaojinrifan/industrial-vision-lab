// 生成一张"工件"测试图：光照不均 + 纹理 + 5 处缺陷
// 用法: gen_workpiece.exe [out.png]
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <random>

int main(int argc, char** argv)
{
    const std::string out = argc > 1 ? argv[1] : "workpiece.png";
    const double ampScale = argc > 2 ? atof(argv[2]) : 1.0;
    const int W = 640, H = 480;

    // ---------- 1. 光照场：左上亮、右下暗，外加轻微暗角 ----------
    cv::Mat illum(H, W, CV_32F);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float gx = (float)x / W, gy = (float)y / H;
            float lin = 1.00f - 0.35f * gx - 0.15f * gy;             // 线性梯度
            float u = gx - 0.5f, v = gy - 0.5f;
            float vig = 1.0f - 0.45f * (u * u * 2.2f + v * v * 2.2f); // 暗角
            illum.at<float>(y, x) = std::max(0.15f, lin * vig);
        }
    }

    // ---------- 2. 工件区域（带倒角的平板） ----------
    cv::Mat plateMask(H, W, CV_8U, cv::Scalar(0));
    cv::Rect part(60, 50, 520, 380);
    cv::rectangle(plateMask, part, cv::Scalar(255), cv::FILLED);
    cv::Mat bevel;
    {
        cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
        cv::morphologyEx(plateMask, bevel, cv::MORPH_GRADIENT, k);
    }

    cv::Mat img(H, W, CV_32F);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float v = illum.at<float>(y, x) * 205.0f;      // 工件表面
            if (!plateMask.at<uchar>(y, x)) v *= 0.42f;    // 背景更暗
            img.at<float>(y, x) = v;
        }
    }

    // ---------- 3. 低频纹理（磨砂表面） ----------
    cv::Mat lowFreq(H, W, CV_32F);
    cv::randn(lowFreq, 0.0, 14.0);
    cv::GaussianBlur(lowFreq, lowFreq, cv::Size(0, 0), 9.0);
    cv::Mat lowNorm;
    cv::normalize(lowFreq, lowNorm, -1.0, 1.0, cv::NORM_MINMAX);
    img += lowNorm * 5.0f;

    // ---------- 4. 缺陷（遮罩加法叠加，边缘保持抗锯齿） ----------
    struct Defect { int kind; int a, b, c, d, e; };  // kind:0=暗斑 1=划痕 2=亮点
    std::vector<Defect> defs;
    defs.push_back({0, 150, 140,  6, 0, 0});   // 小暗斑
    defs.push_back({0, 330, 300, 11, 0, 0});   // 大暗斑
    defs.push_back({0, 480, 110,  4, 0, 0});   // 极小暗斑
    defs.push_back({1, 200, 380, 420, 350, 2}); // 斜向划痕
    defs.push_back({2, 500, 380,  7, 0, 0});   // 亮点（极性相反）
    for (size_t i = 0; i < defs.size(); ++i) {
        const Defect& d = defs[i];
        cv::Mat m(H, W, CV_8U, cv::Scalar(0));
        double amp;
        if (d.kind == 0) {
            cv::circle(m, cv::Point(d.a, d.b), d.c, cv::Scalar(255), cv::FILLED, cv::LINE_AA);
            amp = -55.0;
        } else if (d.kind == 2) {
            cv::circle(m, cv::Point(d.a, d.b), d.c, cv::Scalar(255), cv::FILLED, cv::LINE_AA);
            amp = +50.0;
        } else {
            cv::line(m, cv::Point(d.a, d.b), cv::Point(d.c, d.d),
                     cv::Scalar(255), d.e, cv::LINE_AA);
            amp = -45.0;
        }
        cv::Mat md;
        m.convertTo(md, CV_32F, 1.0 / 255.0);
        img += md * (amp * ampScale);   // 叠加，而不是覆盖
    }

    // 倒角压暗（放在缺陷之后，保证边缘真实）
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            if (bevel.at<uchar>(y, x)) img.at<float>(y, x) -= 45.0f;

    // ---------- 5. 传感器噪声 ----------
    cv::Mat noise(H, W, CV_32F);
    cv::randn(noise, 0.0, 3.5);
    img += noise;
    cv::max(img, 0.0f, img);
    cv::min(img, 255.0f, img);

    cv::Mat out8;
    img.convertTo(out8, CV_8U);
    cv::imwrite(out, out8);

    // ---------- 6. 真值 ----------
    std::ofstream gt("workpiece_gt.txt");
    gt << "# kind cx cy r   (scratch: x1 y1 x2 y2 thick)\n";
    for (size_t i = 0; i < defs.size(); ++i) {
        const Defect& d = defs[i];
        if (d.kind == 1) gt << "scratch " << d.a << " " << d.b << " " << d.c << " " << d.d << " " << d.e << "\n";
        else             gt << (d.kind == 0 ? "dark " : "bright ") << d.a << " " << d.b << " " << d.c << "\n";
    }
    std::cout << "[gen] wrote " << out << "  " << out8.cols << "x" << out8.rows
              << "   defects=" << defs.size() << "\n"
              << "[gen] wrote workpiece_gt.txt\n";
    return 0;
}
