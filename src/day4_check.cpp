// day4_check.cpp —— 标定结果体检：把"畸变有多大"变成数字和图
//
// 用法: day4_check <image>   （默认 calib_imgs/view_00.png）
// 输出: 位移统计表 + compare.png（原图/校正/红绿叠加）

#include <opencv2/opencv.hpp>

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static cv::Point2f proj(const cv::Point2f& n, const cv::Mat& K) {
    return cv::Point2f((float)(K.at<double>(0, 0) * n.x + K.at<double>(0, 2)),
                       (float)(K.at<double>(1, 1) * n.y + K.at<double>(1, 2)));
}

int main(int argc, char** argv) {
    std::string imgPath = argc > 1 ? argv[1] : "calib_imgs/view_00.png";

    cv::Mat K, dist;
    cv::FileStorage fs("calib.yaml", cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "cannot open calib.yaml (run in the folder that has it)\n";
        return 1;
    }
    fs["camera_matrix"] >> K;
    fs["distortion_coefficients"] >> dist;
    fs.release();

    cv::Mat src = cv::imread(imgPath, cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "cannot read " << imgPath << "\n";
        return 1;
    }
    cv::Mat undist;
    cv::undistort(src, undist, K, dist);

    const int W = src.cols, H = src.rows;

    // ---- 1. 全图位移统计（原始像素 -> 校正后位置）----
    std::vector<cv::Point2f> pts, und;
    for (int y = 0; y < H; y += 8)
        for (int x = 0; x < W; x += 8) pts.push_back(cv::Point2f((float)x, (float)y));
    cv::undistortPoints(pts, und, K, dist);

    double maxd = 0, sum = 0;
    int over2 = 0;
    cv::Point2f maxp(0, 0);
    for (size_t i = 0; i < pts.size(); ++i) {
        cv::Point2f q = proj(und[i], K);
        double d = cv::norm(q - pts[i]);
        sum += d;
        if (d > 2.0) ++over2;
        if (d > maxd) { maxd = d; maxp = pts[i]; }
    }
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "== 畸变位移（校正后相对原来的移动量，单位 px）==\n";
    std::cout << "  平均位移 = " << sum / (double)pts.size() << " px\n";
    std::cout << "  最大位移 = " << maxd << " px  @ (" << maxp.x << ", " << maxp.y << ")\n";
    std::cout << "  位移 > 2px 的采样点占比 = " << 100.0 * over2 / (double)pts.size() << " %\n\n";

    // ---- 2. 关键位置逐点 ----
    struct NamedPoint { const char* name; cv::Point2f p; };
    NamedPoint probes[] = {
        {"左上角", cv::Point2f(0.f, 0.f)},
        {"右上角", cv::Point2f((float)(W - 1), 0.f)},
        {"左下角", cv::Point2f(0.f, (float)(H - 1))},
        {"右下角", cv::Point2f((float)(W - 1), (float)(H - 1))},
        {"中心", cv::Point2f(W * 0.5f, H * 0.5f)},
        {"画面中心 1/2 半径处", cv::Point2f(W * 0.75f, H * 0.5f)},
    };
    std::cout << "== 关键位置 ==\n";
    for (auto& np : probes) {
        std::vector<cv::Point2f> one(1, np.p), oned;
        cv::undistortPoints(one, oned, K, dist);
        cv::Point2f q = proj(oned[0], K);
        std::cout << "  " << std::setw(20) << std::left << np.name << std::right
                  << "  (" << np.p.x << "," << np.p.y << ") -> (" << q.x << "," << q.y
                  << ")  位移 " << std::setw(7) << cv::norm(q - np.p) << " px\n";
    }
    std::cout << std::defaultfloat;

    // ---- 3. 拼图：原图 / 校正 / 红绿叠加 ----
    cv::Mat fullTop;
    cv::hconcat(src, undist, fullTop);

    // 左上角放大 4 倍：160x120 -> 640x480
    cv::Rect cr(0, 0, 160, 120);
    cv::Mat cs = src(cr).clone(), cu = undist(cr).clone();
    cv::Mat cs4, cu4;
    cv::resize(cs, cs4, cv::Size(640, 480), 0, 0, cv::INTER_NEAREST);
    cv::resize(cu, cu4, cv::Size(640, 480), 0, 0, cv::INTER_NEAREST);
    cv::Mat zoomRow;
    cv::hconcat(cs4, cu4, zoomRow);

    // 红绿叠加：原图->R，校正->G。对齐处为黄色，错位处出现红/绿边
    cv::Mat gs, gu;
    cv::cvtColor(src, gs, cv::COLOR_BGR2GRAY);
    cv::cvtColor(undist, gu, cv::COLOR_BGR2GRAY);
    std::vector<cv::Mat> ch(3);
    ch[0] = cv::Mat::zeros(gs.size(), CV_8U);  // B
    ch[1] = gu;                                // G
    ch[2] = gs;                                // R
    cv::Mat overlay;
    cv::merge(ch, overlay);

    cv::Mat ovz;
    cv::resize(overlay(cr), ovz, cv::Size(640, 480), 0, 0, cv::INTER_NEAREST);
    cv::Mat ovRow;
    cv::hconcat(overlay, ovz, ovRow);

    cv::Mat canvas;
    cv::vconcat(fullTop, zoomRow, canvas);
    cv::vconcat(canvas, ovRow, canvas);

    cv::putText(canvas, "ORIGINAL", cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 0), 2);
    cv::putText(canvas, "UNDISTORTED", cv::Point(650, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 0), 2);
    cv::putText(canvas, "TL corner x4 (orig)", cv::Point(10, 505), cv::FONT_HERSHEY_SIMPLEX,
                0.6, cv::Scalar(0, 255, 0), 2);
    cv::putText(canvas, "TL corner x4 (undist)", cv::Point(650, 505), cv::FONT_HERSHEY_SIMPLEX,
                0.6, cv::Scalar(0, 255, 0), 2);
    cv::putText(canvas, "R=orig G=undist  (yellow=aligned)", cv::Point(10, 985),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    cv::putText(canvas, "same, TL x4", cv::Point(650, 985), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 0), 2);

    cv::imwrite("compare.png", canvas);
    std::cout << "\n[out] wrote compare.png (" << canvas.cols << "x" << canvas.rows << ")\n";
    return 0;
}
