// probe_exp3.cpp —— 给"静态测量拍摄"找合适的曝光值
// 输出每个曝光档的亮度均值/最大值/削顶比例/帧率
#include <opencv2/opencv.hpp>
#include <cstdio>

int main()
{
    cv::VideoCapture cap(0, cv::CAP_DSHOW);
    if (!cap.isOpened()) { printf("cam open fail\n"); return 1; }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);   // manual

    printf("%-8s %8s %8s %10s %8s\n", "exposure", "mean", "max", "clip%", "fps");
    const double exps[] = { -2.0, -3.0, -4.0, -5.0, -6.0 };

    for (double e : exps)
    {
        cap.set(cv::CAP_PROP_EXPOSURE, e);
        for (int i = 0; i < 8; ++i) { cv::Mat t; cap >> t; }   // 稳定

        cv::Mat f;
        double t0 = (double)cv::getTickCount();
        int n = 0;
        for (int i = 0; i < 12; ++i) { if (cap.read(f)) ++n; }
        double dt = ((double)cv::getTickCount() - t0) / cv::getTickFrequency();

        cv::Mat gray; cv::cvtColor(f, gray, cv::COLOR_BGR2GRAY);
        cv::Scalar mu, sd; cv::meanStdDev(gray, mu, sd);
        double mx; cv::minMaxLoc(gray, 0, &mx);
        int clipped = cv::countNonZero(gray >= 253);

        printf("%-8.1f %8.1f %8.0f %9.2f%% %8.2f\n",
               e, mu[0], mx, 100.0 * clipped / gray.total(), n / dt);
    }
    return 0;
}
