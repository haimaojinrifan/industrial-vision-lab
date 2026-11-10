#include <opencv2/opencv.hpp>
#include <cstdio>
#include <vector>

int main()
{
    cv::Mat src = cv::imread("workpiece.png", cv::IMREAD_GRAYSCALE);
    cv::Mat blur;
    cv::GaussianBlur(src, blur, cv::Size(5, 5), 0);
    printf("src  (500,380)=%d   (480,380)=%d  (520,380)=%d\n",
           src.at<uchar>(380, 500), src.at<uchar>(380, 480), src.at<uchar>(380, 520));

    cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(31, 31));
    cv::Mat th;
    cv::morphologyEx(blur, th, cv::MORPH_TOPHAT, k);
    printf("tophat(500,380)=%d  tophat(480,380)=%d\n",
           th.at<uchar>(380, 500), th.at<uchar>(380, 480));

    double mx; cv::Point loc;
    cv::minMaxLoc(th, 0, &mx, 0, &loc);
    printf("tophat max=%.0f at (%d,%d)\n", mx, loc.x, loc.y);

    cv::Mat bin;
    cv::threshold(th, bin, 20, 255, cv::THRESH_BINARY);
    cv::Mat k5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(bin, bin, cv::MORPH_OPEN, k5);
    std::vector<std::vector<cv::Point>> cs;
    cv::findContours(bin, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    printf("contours at thr=20: %d\n", (int)cs.size());
    for (size_t i = 0; i < cs.size() && i < 20; ++i) {
        cv::Rect r = cv::boundingRect(cs[i]);
        printf("  (%3d,%3d %3dx%3d) area=%.0f\n", r.x, r.y, r.width, r.height,
               cv::contourArea(cs[i]));
    }
    return 0;
}
