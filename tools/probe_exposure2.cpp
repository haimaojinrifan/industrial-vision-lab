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
int main(){
    std::cout<<std::fixed<<std::setprecision(2);
    cv::VideoCapture cap(0, cv::CAP_DSHOW);
    if(!cap.isOpened()){ std::cout<<"open failed\n"; return 1; }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT,480);

    auto show=[&](const char* tag){
        std::cout<<tag<<" auto_exp_prop="<<cap.get(cv::CAP_PROP_AUTO_EXPOSURE)
                 <<" exp_prop="<<cap.get(cv::CAP_PROP_EXPOSURE)
                 <<" fps="<<measure(cap)<<"\n";
    };
    std::cout<<"== A. ??(??) ==\n";  show("A");
    std::cout<<"== B. set AUTO_EXPOSURE=0, EXPOSURE=-6 ==\n";
    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0);
    cap.set(cv::CAP_PROP_EXPOSURE, -6);
    show("B");
    std::cout<<"== C. ?? EXPOSURE=-4 ==\n";
    cap.set(cv::CAP_PROP_EXPOSURE, -4);
    show("C");
    std::cout<<"== D. ???? AUTO_EXPOSURE=0.75 ==\n";
    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.75);
    show("D");
    return 0;
}
