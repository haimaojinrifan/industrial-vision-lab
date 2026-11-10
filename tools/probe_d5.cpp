// 复现 day5_.cpp 的流水线，把每一级的中间量打印出来，定位"框不到"的原因
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <vector>

static void dumpContours(const char* tag, const cv::Mat& bin, double minA, double maxA)
{
    std::vector<std::vector<cv::Point>> cs;
    std::vector<cv::Vec4i> h;
    cv::findContours(bin, cs, h, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    printf("\n[%s] contours=%d  (minA=%.0f maxA=%.0f)\n", tag, (int)cs.size(), minA, maxA);
    int kept = 0;
    for (size_t i = 0; i < cs.size(); ++i) {
        double a = cv::contourArea(cs[i]);
        cv::Rect r = cv::boundingRect(cs[i]);
        double solid = a / (double)(r.width * r.height + 1);
        bool pass = (a >= minA && a <= maxA);
        if (pass) ++kept;
        printf("  #%02d area=%8.1f  bbox=(%3d,%3d %3dx%3d)  solid=%.2f  %s\n",
               (int)i, a, r.x, r.y, r.width, r.height, solid,
               pass ? "KEEP" : "drop(area)");
    }
    printf("[%s] kept %d / %d\n", tag, kept, (int)cs.size());
}

int main()
{
    cv::Mat src = cv::imread("workpiece.png");
    if (src.empty()) { printf("cannot read workpiece.png\n"); return 1; }
    printf("src %dx%d ch=%d\n", src.cols, src.rows, src.channels());

    cv::Mat gray, blur;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);

    cv::Rect inner_roi(65, 65, gray.cols - 130, gray.rows - 130);
    cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
    mask(inner_roi).setTo(255);

    cv::Mat blur_roi;
    blur.copyTo(blur_roi, mask);
    printf("blur_roi mean=%.1f  zeros=%d / %d  <-- masked-out area is BLACK\n",
           cv::mean(blur_roi)[0], (int)cv::countNonZero(blur_roi == 0), blur_roi.total());

    cv::Mat bin;
    cv::adaptiveThreshold(blur_roi, bin, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                          cv::THRESH_BINARY_INV, 31, 5);
    printf("bin foreground = %.1f%% of frame\n", 100.0 * cv::countNonZero(bin) / bin.total());

    cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat opened, closed;
    cv::morphologyEx(bin, opened, cv::MORPH_OPEN, k);
    cv::morphologyEx(opened, closed, cv::MORPH_CLOSE, k);

    dumpContours("dark: adaptiveThreshold+open+close", closed, 25.0, 8000.0);

    // ---- 亮缺陷分支 ----
    cv::Mat clahe_roi;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(blur_roi, clahe_roi);
    cv::Mat k_hat = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(21, 21));
    cv::Mat tophat;
    cv::morphologyEx(clahe_roi, tophat, cv::MORPH_TOPHAT, k_hat);
    double mn, mx;
    cv::minMaxLoc(tophat, &mn, &mx);
    printf("\n[tophat] min=%.1f max=%.1f  (threshold used in day5_.cpp is 9)\n", mn, mx);
    printf("[tophat] pixels >9: %d\n", (int)cv::countNonZero(tophat > 9));

    cv::Mat bin2, m2;
    cv::threshold(tophat, bin2, 9, 255, cv::THRESH_BINARY);
    cv::Mat k3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(bin2, m2, cv::MORPH_OPEN, k3);
    dumpContours("bright: tophat>9", m2, 25.0, 8000.0);

    cv::imwrite("probe_bin_dark.png", closed);
    cv::imwrite("probe_tophat.png", tophat);
    return 0;
}
