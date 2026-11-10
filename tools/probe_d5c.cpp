// 亮缺陷：不同核大小 / 阈值策略的检出情况
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <vector>

static int count(const cv::Mat& resp, double thr, int openSz,
                 double minA, bool& hit, const cv::Point& gt, int& cx, int& cy)
{
    cv::Mat bin;
    cv::threshold(resp, bin, thr, 255, cv::THRESH_BINARY);
    cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(openSz, openSz));
    cv::morphologyEx(bin, bin, cv::MORPH_OPEN, k);
    std::vector<std::vector<cv::Point>> cs;
    cv::findContours(bin, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    int n = 0; hit = false; cx = cy = -1;
    for (size_t i = 0; i < cs.size(); ++i) {
        double a = cv::contourArea(cs[i]);
        if (a < minA) continue;
        ++n;
        cv::Rect r = cv::boundingRect(cs[i]);
        if (r.contains(gt)) { hit = true; cx = r.x + r.width / 2; cy = r.y + r.height / 2; }
    }
    return n;
}

int main()
{
    cv::Mat src = cv::imread("workpiece.png", cv::IMREAD_GRAYSCALE);
    cv::Mat blur;
    cv::GaussianBlur(src, blur, cv::Size(5, 5), 0);
    const cv::Point gt(500, 380);

    for (int ks = 15; ks <= 41; ks += 8) {
        cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ks, ks));
        cv::Mat th;
        cv::morphologyEx(blur, th, cv::MORPH_TOPHAT, k);
        cv::Scalar mu, sd;
        cv::meanStdDev(th, mu, sd);
        double mx; cv::minMaxLoc(th, 0, &mx);
        printf("kernel=%2d  mean=%.1f sd=%.1f max=%.1f\n", (int)ks, mu[0], sd[0], mx);
        double cand[] = { mu[0] + 3 * sd[0], mu[0] + 5 * sd[0], 25.0, 35.0, 0.4 * mx };
        const char* name[] = { "mean+3sd", "mean+5sd", "fixed 25", "fixed 35", "0.4*max" };
        for (int i = 0; i < 5; ++i) {
            bool hit; int cx, cy;
            int n = count(th, cand[i], 5, 30.0, hit, gt, cx, cy);
            printf("    %-8s thr=%5.1f -> %3d ge   bright:%s(%d,%d)\n",
                   name[i], cand[i], n, hit ? "YES" : "no ", cx, cy);
        }
    }
    printf("\nGT bright = (500,380) r7, bbox about (493,373 15x15)\n");
    return 0;
}
