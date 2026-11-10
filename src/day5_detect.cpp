#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

// 亮斑检测：圆形度 + 面积 + ROI，细长条纹不再无条件放行
static void detectAndDrawBright(const cv::Mat& resp, cv::Mat& dst, cv::Scalar color,
    double minA, double maxA, double minCirc, int margin = 10)
{
    cv::Mat bin, morph;
    cv::threshold(resp, bin, 20, 255, cv::THRESH_BINARY);
    cv::Mat k_small = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(bin, morph, cv::MORPH_OPEN, k_small);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(morph, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (auto& c : contours)
    {
        double area = cv::contourArea(c);
        if (area < minA || area > maxA) continue;

        double peri = cv::arcLength(c, true);
        double circ = (peri < 1e-6) ? 0.0 : (4 * CV_PI * area) / (peri * peri);
        if (circ < minCirc) continue;

        cv::Rect r = cv::boundingRect(c);
        if (r.x < margin || r.y < margin ||
            (r.x + r.width) > resp.cols - margin ||
            (r.y + r.height) > resp.rows - margin)
            continue;

        double w = r.width;
        double h = r.height;
        double ratio = std::max(w, h) / (std::min(w, h) + 1e-6);
        // 限制长宽比，不允许无限细长
        if (ratio > 4.0) continue;

        cv::rectangle(dst, r, color, 2);
    }
}

int main(int argc, char** argv)
{
    const std::string path = (argc > 1) ? argv[1] : "workpiece.png";
    cv::Mat src = cv::imread(path);
    if (src.empty())
    {
        printf("image read fail: %s\n", path.c_str());
        return -1;
    }
    cv::Mat gray, blur, dst;
    src.copyTo(dst);

    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    // 改成 3x3，保护细划痕幅值
    cv::GaussianBlur(gray, blur, cv::Size(3, 3), 0);

    // ROI，之后用来清零外围
    cv::Rect inner_roi(65, 65, gray.cols - 130, gray.rows - 130);
    cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
    mask(inner_roi).setTo(255);

    // ========== 暗缺陷：BlackHat，核31，阈值12 ==========
    cv::Mat k_blackhat = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(31, 31));
    cv::Mat blackhat;
    cv::morphologyEx(blur, blackhat, cv::MORPH_BLACKHAT, k_blackhat);

    cv::Mat bin_dark;
    cv::threshold(blackhat, bin_dark, 12, 255, cv::THRESH_BINARY);
    bin_dark.setTo(0, ~mask);   // 先全图二值，再抹掉 ROI 外，避免邻域掺黑

    cv::Mat k_morph = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(bin_dark, bin_dark, cv::MORPH_OPEN, k_morph);

    std::vector<std::vector<cv::Point>> contours_dark;
    cv::findContours(bin_dark, contours_dark, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    const double min_area_dark = 25.0;
    const double max_area = 8000.0;
    const int margin = 10;

    for (auto& c : contours_dark)
    {
        double area = cv::contourArea(c);
        if (area < min_area_dark || area > max_area)
            continue;

        cv::Rect r = cv::boundingRect(c);
        if (r.x < margin || r.y < margin ||
            (r.x + r.width) > bin_dark.cols - margin ||
            (r.y + r.height) > bin_dark.rows - margin)
            continue;

        double w = r.width;
        double h = r.height;
        double ratio = std::max(w, h) / (std::min(w, h) + 1e-6);
        if (ratio > 12.0) continue; // 划痕允许更长，但限制极端长条噪声

        cv::rectangle(dst, r, cv::Scalar(0, 0, 255), 2);
    }

    // ========== 亮缺陷：TopHat，去掉 CLAHE，核31，阈值20 ==========
    cv::Mat k_tophat = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(31, 31));
    cv::Mat tophat;
    cv::morphologyEx(blur, tophat, cv::MORPH_TOPHAT, k_tophat);
    tophat.setTo(0, ~mask);

    const double min_area_bright = 25.0;
    const double min_circularity = 0.45;
    detectAndDrawBright(tophat, dst, cv::Scalar(0, 255, 0),
        min_area_bright, max_area, min_circularity, margin);

    cv::imshow("src", src);
    cv::imshow("mask", mask);
    cv::imshow("bin_dark", bin_dark);
    cv::imshow("blackhat", blackhat);
    cv::imshow("tophat_bright", tophat);
    cv::imshow("result", dst);
    cv::waitKey(0);
    return 0;
}
