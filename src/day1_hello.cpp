// day1_hello.cpp —— 环境自检：确认 OpenCV 能链接、能读图、能开窗口
//
// 用法:
//   day1_hello              生成一张测试图案并显示
//   day1_hello <image>      显示指定图片
#include <opencv2/opencv.hpp>
#include <cstdio>

int main(int argc, char** argv)
{
    std::printf("OpenCV version: %s\n", CV_VERSION);

    cv::Mat img;

    if (argc > 1)
    {
        img = cv::imread(argv[1], cv::IMREAD_COLOR);
        if (img.empty())
        {
            std::printf("image read fail: %s\n", argv[1]);
            return -1;
        }
        std::printf("loaded %s  (%d x %d, %d channels)\n",
                    argv[1], img.cols, img.rows, img.channels());
    }
    else
    {
        // 没有给图就自己造一张：横向渐变 + 几个方块，用来确认显示链路
        img.create(480, 640, CV_8UC3);
        for (int y = 0; y < img.rows; ++y)
        {
            cv::Vec3b* p = img.ptr<cv::Vec3b>(y);
            for (int x = 0; x < img.cols; ++x)
            {
                const uchar v = (uchar)(255 * x / (img.cols - 1));
                p[x] = cv::Vec3b(v / 2, v, 255 - v);
            }
        }
        cv::rectangle(img, cv::Rect(40, 40, 120, 80),   cv::Scalar(0, 0, 255), -1);
        cv::rectangle(img, cv::Rect(200, 40, 120, 80),  cv::Scalar(0, 255, 0), -1);
        cv::rectangle(img, cv::Rect(360, 40, 120, 80),  cv::Scalar(255, 0, 0), -1);
        cv::putText(img, "OpenCV " CV_VERSION, cv::Point(40, 300),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 255, 255), 2);
        std::printf("generated a 640x480 test pattern (no image argument given)\n");
    }

    cv::imshow("day1_hello  (press any key to quit)", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
