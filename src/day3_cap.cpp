#include <opencv2/opencv.hpp>
#include <iostream>

void measure_camera_fps() {
    cv::VideoCapture cap(0, cv::CAP_DSHOW);
    if (!cap.isOpened()) { std::cerr << "camera open failed\n"; return; }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0);   // 0 = 手动曝光（DSHOW）
    cap.set(cv::CAP_PROP_EXPOSURE, -6);       // 曝光值（DSHOW 的刻度，越负越短）

    cv::Mat f;
    for (int i = 0; i < 10; ++i) cap.read(f);   // 等自动曝光稳定

    double t0 = (double)cv::getTickCount();
    int n = 0;
    while (n < 150 && cap.read(f)) ++n;          // 纯采集，不 imshow
    double dt = ((double)cv::getTickCount() - t0) / cv::getTickFrequency();
    std::cout << "camera capture-only FPS = " << n / dt << "\n";
}

int main()
{
    const int width = 640;
    const int height = 480;
    const int total_frames = 200;
    const std::string video_name = "test_circle.avi";

    measure_camera_fps();

    // ========== 第一步：生成测试视频 ==========
    cv::VideoWriter writer(
        video_name,
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        30.0,
        cv::Size(width, height)
    );
    if (!writer.isOpened())
    {
        std::cerr << "VideoWriter open failed\n";
        return -1;
    }

    cv::Mat img(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i < total_frames; ++i)
    {
        img.setTo(cv::Scalar(0, 0, 0));
        int cx = 50 + i * 2;
        int cy = height / 2;
        cv::circle(img, cv::Point(cx, cy), 20, cv::Scalar(0, 255, 0), -1);
        cv::putText(img, "Frame: " + std::to_string(i), cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        writer.write(img);
    }
    writer.release();
    std::cout << "Generated " << total_frames << " frames -> " << video_name << "\n";

    // ========== 第二步：读取并计时 ==========
    cv::VideoCapture cap(video_name);
    if (!cap.isOpened())
    {
        std::cerr << "open generated video failed\n";
        return -1;
    }

    cv::Mat frame;
    double t0 = cv::getTickCount();
    int cnt = 0;
    while (cap.read(frame))
    {
        cv::imshow("play", frame);
        int k = cv::waitKey(1);
        if (k == 27) break;
        cnt++;
    }
    double dt = (cv::getTickCount() - t0) / cv::getTickFrequency();
    double fps = cnt / dt;

    std::cout << "Played " << cnt << " frames, time: " << dt << " s\n";
    std::cout << "Read+imshow+waitKey FPS = " << fps << "\n";

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
