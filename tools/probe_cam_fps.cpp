#include <opencv2/opencv.hpp>
#include <iostream>
int main(){
    cv::VideoCapture cap(0, cv::CAP_DSHOW);
    if(!cap.isOpened()){ std::cout << "open failed\n"; return 1; }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cv::Mat f;
    for(int i=0;i<10;i++) cap.read(f);              // warmup / auto-exposure
    double t0 = (double)cv::getTickCount();
    int n = 0;
    while(n < 150){ if(!cap.read(f)) break; ++n; }   // capture-only, no imshow
    double dt = ((double)cv::getTickCount() - t0) / cv::getTickFrequency();
    std::cout << "camera capture-only: frames=" << n << " time=" << dt << "s fps=" << (n/dt) << "\n";
    std::cout << "size=" << f.cols << "x" << f.rows << "\n";
    return 0;
}
