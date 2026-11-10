#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
int main(int argc, char** argv){
    std::string dir = argc > 1 ? argv[1] : "calib_imgs";
    std::vector<cv::String> files; cv::glob(dir + "/*.png", files, false);
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  view   mean_bright  sharpness(varLap)  white_clip%%\n";
    for(size_t i=0;i<files.size();++i){
        cv::Mat g = cv::imread(files[i], cv::IMREAD_GRAYSCALE);
        if(g.empty()) continue;
        double m = cv::mean(g)[0];
        cv::Mat lap; cv::Laplacian(g, lap, CV_64F);
        cv::Scalar mu, sd; cv::meanStdDev(lap, mu, sd);
        double clip = 100.0 * cv::countNonZero(g == 255) / (double)(g.rows*g.cols);
        std::cout << std::setw(6) << i << "   " << std::setw(10) << m << "   "
                  << std::setw(15) << sd[0]*sd[0] << "   " << std::setw(10) << clip << "\n";
    }
    return 0;
}
