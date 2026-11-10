// probe_model.cpp —— 诊断：这套标定数据到底能约束住什么
// 1) 每张图棋盘格占画面的比例（覆盖度）
// 2) 四种畸变模型下的系数/RMS/角落位移对比

#include <opencv2/opencv.hpp>

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static const int kCols = 9;
static const int kRows = 6;
static const double kSquare = 25.0;

static bool findCorners(const cv::Mat& gray, std::vector<cv::Point2f>& c) {
    bool ok = cv::findChessboardCorners(
        gray, cv::Size(kCols, kRows), c,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
    if (!ok) return false;
    cv::cornerSubPix(gray, c, cv::Size(11, 11), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
    return true;
}

struct Fit {
    cv::Mat K, dist;
    double rms;
    double dTL, dTR, dBL, dBR;
};

static double shiftAt(const cv::Mat& K, const cv::Mat& dist, cv::Point2f p) {
    std::vector<cv::Point2f> one(1, p), und;
    cv::undistortPoints(one, und, K, dist);
    cv::Point2f q((float)(K.at<double>(0, 0) * und[0].x + K.at<double>(0, 2)),
                  (float)(K.at<double>(1, 1) * und[0].y + K.at<double>(1, 2)));
    return cv::norm(q - p);
}

static Fit fit(const std::vector<std::vector<cv::Point2f> >& pts, cv::Size sz, int flags) {
    std::vector<cv::Point3f> tmpl;
    for (int y = 0; y < kRows; ++y)
        for (int x = 0; x < kCols; ++x)
            tmpl.push_back(cv::Point3f((float)(x * kSquare), (float)(y * kSquare), 0.f));
    std::vector<std::vector<cv::Point3f> > obj(pts.size(), tmpl);
    Fit f;
    f.K = cv::Mat::eye(3, 3, CV_64F);
    f.dist = cv::Mat::zeros(5, 1, CV_64F);
    std::vector<cv::Mat> rv, tv;
    f.rms = cv::calibrateCamera(obj, pts, sz, f.K, f.dist, rv, tv, flags,
                                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 1e-6));
    const int W = sz.width, H = sz.height;
    f.dTL = shiftAt(f.K, f.dist, cv::Point2f(0.f, 0.f));
    f.dTR = shiftAt(f.K, f.dist, cv::Point2f((float)(W - 1), 0.f));
    f.dBL = shiftAt(f.K, f.dist, cv::Point2f(0.f, (float)(H - 1)));
    f.dBR = shiftAt(f.K, f.dist, cv::Point2f((float)(W - 1), (float)(H - 1)));
    return f;
}

int main() {
    std::vector<cv::String> files;
    cv::glob("calib_imgs/*.png", files, false);
    if (files.empty()) {
        std::cerr << "no images\n";
        return 1;
    }

    std::vector<std::vector<cv::Point2f> > pts;
    cv::Size imgSize;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "== 1. 棋盘格覆盖度（bounding box 占画面比例）==\n";
    for (size_t i = 0; i < files.size(); ++i) {
        cv::Mat g = cv::imread(files[i], cv::IMREAD_GRAYSCALE);
        if (g.empty()) continue;
        imgSize = g.size();
        std::vector<cv::Point2f> c;
        if (!findCorners(g, c)) {
            std::cout << "  " << files[i] << "  UNDETECTED\n";
            continue;
        }
        pts.push_back(c);
        cv::Rect bb = cv::boundingRect(c);
        std::cout << "  " << files[i].substr(files[i].find_last_of("/\\") + 1)
                  << "  bbox(" << bb.x << "," << bb.y << " " << bb.width << "x" << bb.height
                  << ")  spanX=" << 100.0 * bb.width / imgSize.width
                  << "%  spanY=" << 100.0 * bb.height / imgSize.height << "%\n";
    }
    std::cout << "  -> " << pts.size() << " views used\n\n";

    std::cout << "== 2. 四种畸变模型对比（同一批角点）==\n";
    struct Case { const char* name; int flags; };
    Case cases[] = {
        {"全部5参数自由", 0},
        {"切向置零(p1=p2=0)", cv::CALIB_ZERO_TANGENT_DIST},
        {"k3固定=0", cv::CALIB_FIX_K3},
        {"切向置零 + k3=0", cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_K3},
    };
    for (auto& cs : cases) {
        Fit f = fit(pts, imgSize, cs.flags);
        std::cout << "  [" << cs.name << "]\n";
        std::cout << std::setprecision(5);
        std::cout << "    k1=" << f.dist.at<double>(0) << "  k2=" << f.dist.at<double>(1)
                  << "  p1=" << f.dist.at<double>(2) << "  p2=" << f.dist.at<double>(3)
                  << "  k3=" << f.dist.at<double>(4) << "\n";
        std::cout << "    fx=" << f.K.at<double>(0, 0) << "  fy=" << f.K.at<double>(1, 1)
                  << "  cx=" << f.K.at<double>(0, 2) << "  cy=" << f.K.at<double>(1, 2)
                  << "  RMS=" << f.rms << "\n";
        std::cout << std::setprecision(2);
        std::cout << "    角落位移  TL=" << f.dTL << "  TR=" << f.dTR << "  BL=" << f.dBL
                  << "  BR=" << f.dBR << " px\n\n";
    }
    return 0;
}
