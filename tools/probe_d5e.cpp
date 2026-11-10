#include <opencv2/opencv.hpp>
#include <cstdio>
#include <vector>

static int countHits(const cv::Mat& resp, double thr, int openSz,
                     const std::vector<cv::Point>& gts, bool verbose)
{
    cv::Mat bin;
    cv::threshold(resp, bin, thr, 255, cv::THRESH_BINARY);
    cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(openSz, openSz));
    cv::morphologyEx(bin, bin, cv::MORPH_OPEN, k);
    std::vector<std::vector<cv::Point>> cs;
    cv::findContours(bin, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    int hits = 0;
    for (size_t g = 0; g < gts.size(); ++g) {
        bool found = false;
        for (size_t i = 0; i < cs.size(); ++i) {
            if (cv::contourArea(cs[i]) < 25) continue;
            if (cv::boundingRect(cs[i]).contains(gts[g])) { found = true; break; }
        }
        if (found) ++hits;
    }
    if (verbose) {
        printf("      contours=%d ", (int)cs.size());
        for (size_t g = 0; g < gts.size(); ++g) {
            bool found = false;
            for (size_t i = 0; i < cs.size(); ++i) {
                if (cv::contourArea(cs[i]) < 25) continue;
                if (cv::boundingRect(cs[i]).contains(gts[g])) { found = true; break; }
            }
            printf("%c", found ? 'Y' : '.');
        }
    }
    return hits;
}

int main(int argc, char** argv)
{
    const char* file = argc > 1 ? argv[1] : "workpiece.png";
    cv::Mat src = cv::imread(file, cv::IMREAD_GRAYSCALE);
    if (src.empty()) { printf("cannot read %s\n", file); return 1; }
    printf("image: %s\n", file);
    cv::Mat blur;
    cv::GaussianBlur(src, blur, cv::Size(5, 5), 0);

    std::vector<cv::Point> darkGt;
    darkGt.push_back(cv::Point(150, 140));
    darkGt.push_back(cv::Point(330, 300));
    darkGt.push_back(cv::Point(480, 110));
    darkGt.push_back(cv::Point(310, 365));   // scratch midpoint
    std::vector<cv::Point> brightGt;
    brightGt.push_back(cv::Point(500, 380));

    int ks[] = {21, 31, 41, 51};
    double ths[] = {12, 16, 20, 25, 30};

    printf("== BLACKHAT (dark defects), 4 GT: small, big, tiny, scratch ==\n");
    for (int i = 0; i < 4; ++i) {
        cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ks[i], ks[i]));
        cv::Mat bh;
        cv::morphologyEx(blur, bh, cv::MORPH_BLACKHAT, k);
        double mx; cv::minMaxLoc(bh, 0, &mx);
        printf("  kernel=%2d max=%.0f\n", ks[i], mx);
        for (int j = 0; j < 5; ++j) {
            int h = countHits(bh, ths[j], 5, darkGt, true);
            printf("   thr=%2.0f  hits=%d/4\n", ths[j], h);
        }
    }

    printf("\n== TOPHAT (bright defect), 1 GT ==\n");
    for (int i = 0; i < 4; ++i) {
        cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ks[i], ks[i]));
        cv::Mat th;
        cv::morphologyEx(blur, th, cv::MORPH_TOPHAT, k);
        double mx; cv::minMaxLoc(th, 0, &mx);
        printf("  kernel=%2d max=%.0f\n", ks[i], mx);
        for (int j = 0; j < 5; ++j) {
            int h = countHits(th, ths[j], 5, brightGt, true);
            printf("   thr=%2.0f  hits=%d/1\n", ths[j], h);
        }
    }
    return 0;
}
