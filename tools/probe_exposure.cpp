#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>

static double measure(cv::VideoCapture& cap, int n=150){
    cv::Mat f;
    for(int i=0;i<10;i++) cap.read(f);
    double t0=(double)cv::getTickCount();
    int c=0;
    while(c<n && cap.read(f)) ++c;
    double dt=((double)cv::getTickCount()-t0)/cv::getTickFrequency();
    return c/dt;
}
static double brightness(cv::VideoCapture& cap){
    cv::Mat f; cap.read(f);
    if(f.empty()) return -1;
    cv::Mat g; cv::cvtColor(f,g,cv::COLOR_BGR2GRAY);
    return cv::mean(g)[0];
}
int main(){
    cv::VideoCapture cap(0, cv::CAP_DSHOW);
    if(!cap.isOpened()){ std::cout<<"open failed\n"; return 1; }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT,480);
    std::cout<<std::fixed<<std::setprecision(2);
    std::cout<<"[auto] auto_exp="<<cap.get(cv::CAP_PROP_AUTO_EXPOSURE)<<" exp="<<cap.get(cv::CAP_PROP_EXPOSURE)<<"\n";
    std::cout<<"[auto]   fps="<<measure(cap)<<" mean_brightness="<<brightness(cap)<<"\n";
    double exps[] = {-4,-5,-6,-7,-8};
    for(double e: exps){
        cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);  // 0.25 = manual (DSHOW)
        cap.set(cv::CAP_PROP_EXPOSURE, e);
        double fps = measure(cap);
        std::cout<<"[manual exp="<<e<<"] auto_exp="<<cap.get(cv::CAP_PROP_AUTO_EXPOSURE)
                 <<" fps="<<fps<<" mean_brightness="<<brightness(cap)<<"\n";
    }
    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.75); // restore auto
    return 0;
}
