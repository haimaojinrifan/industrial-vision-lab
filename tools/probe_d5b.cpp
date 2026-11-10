// 验证"正确写法"能否稳定找到 5 处缺陷
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <vector>
#include <algorithm>

struct Out { int x, y, w, h; double area; };

static std::vector<Out> extract(const cv::Mat& resp, int openSz, double minA, double maxA)
{
    cv::Mat bin;
    cv::threshold(resp, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(openSz, openSz));
    cv::morphologyEx(bin, bin, cv::MORPH_OPEN, k);

    std::vector<std::vector<cv::Point>> cs;
    cv::findContours(bin, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<Out> res;
    for (size_t i = 0; i < cs.size(); ++i) {
        double a = cv::contourArea(cs[i]);
        if (a < minA || a > maxA) continue;
        cv::Rect r = cv::boundingRect(cs[i]);
        Out o; o.x = r.x; o.y = r.y; o.w = r.width; o.h = r.height; o.area = a;
        res.push_back(o);
    }
    return res;
}

int main()
{
    cv::Mat src = cv::imread("workpiece.png", cv::IMREAD_GRAYSCALE);
    if (src.empty()) { printf("no image\n"); return 1; }
    cv::Mat blur;
    cv::GaussianBlur(src, blur, cv::Size(5, 5), 0);

    cv::Mat k21 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(21, 21));
    cv::Mat blackhat, tophat;
    cv::morphologyEx(blur, blackhat, cv::MORPH_BLACKHAT, k21);
    cv::morphologyEx(blur, tophat, cv::MORPH_TOPHAT, k21);

    double mn, mx;
    cv::minMaxLoc(blackhat, &mn, &mx); printf("blackhat  min=%.0f max=%.0f\n", mn, mx);
    cv::minMaxLoc(tophat, &mn, &mx);   printf("tophat    min=%.0f max=%.0f\n", mn, mx);

    std::vector<Out> dark = extract(blackhat, 5, 30.0, 5000.0);
    std::vector<Out> bright = extract(tophat, 5, 30.0, 5000.0);

    printf("\n--- DARK (blackhat + Otsu) %d ---\n", (int)dark.size());
    for (size_t i = 0; i < dark.size(); ++i)
        printf("  (%3d,%3d %3dx%3d) area=%.0f  cx=%d cy=%d\n",
               dark[i].x, dark[i].y, dark[i].w, dark[i].h, dark[i].area,
               dark[i].x + dark[i].w / 2, dark[i].y + dark[i].h / 2);

    printf("\n--- BRIGHT (tophat + Otsu) %d ---\n", (int)bright.size());
    for (size_t i = 0; i < bright.size(); ++i)
        printf("  (%3d,%3d %3dx%3d) area=%.0f  cx=%d cy=%d\n",
               bright[i].x, bright[i].y, bright[i].w, bright[i].h, bright[i].area,
               bright[i].x + bright[i].w / 2, bright[i].y + bright[i].h / 2);

    printf("\n--- GT ---\n");
    printf("  dark    (150,140) r6    (330,300) r11   (480,110) r4\n");
    printf("  scratch (200,380)-(420,350) thick2\n");
    printf("  bright  (500,380) r7\n");
    return 0;
}
