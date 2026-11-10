#include <opencv2/opencv.hpp>
#include <iostream>
int main(){
    for(int i=0;i<3;i++){
        cv::VideoCapture cap(i, cv::CAP_DSHOW);
        std::cout << "index " << i << " opened=" << cap.isOpened();
        if(cap.isOpened()){
            std::cout << " w=" << cap.get(cv::CAP_PROP_FRAME_WIDTH)
                      << " h=" << cap.get(cv::CAP_PROP_FRAME_HEIGHT)
                      << " fps=" << cap.get(cv::CAP_PROP_FPS);
        }
        std::cout << "\n";
    }
    return 0;
}
