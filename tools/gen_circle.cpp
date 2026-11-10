// gen_circle.cpp —— 生成一张直径精确已知的圆，用来做算法自检（A 级实验）
// 用法: gen_circle.exe [out.png] [radius]
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv)
{
    const char* out = argc > 1 ? argv[1] : "circle_test.png";
    const int R = argc > 2 ? atoi(argv[2]) : 100;

    cv::Mat img(480, 640, CV_8UC1, cv::Scalar(210));      // 亮背景
    cv::circle(img, cv::Point(320, 240), R, cv::Scalar(35), -1, cv::LINE_AA);  // 暗圆

    cv::Mat noise(img.size(), CV_16SC1);
    cv::randn(noise, 0, 3);
    cv::Mat tmp;
    img.convertTo(tmp, CV_16S);
    tmp += noise;
    tmp.convertTo(img, CV_8U);

    cv::imwrite(out, img);
    printf("[gen] %s   circle r=%d px -> true diameter = %.1f px\n", out, R, 2.0 * R);
    return 0;
}
