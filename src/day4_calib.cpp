// day4_calib.cpp —— Week 1 Day 4：相机标定（张正友棋盘格法）
//
// 用法:
//   day4_calib --gen [square_px]                  生成 9x6 内角点棋盘格 PNG（默认方格 40px）
//   day4_calib --capture [camera_index]            摄像头采标定图（空格存图，ESC 结束）
//   day4_calib --solve <square_mm> [img_dir]       求解内参 + 畸变 -> calib.yaml
//   day4_calib --synth                             合成数据自检：验证求解器能还原真值
//   day4_calib --undistort <image.png>             【教学版】单张校正对比，原生 undistort，左右拼接
//   day4_calib --undistort --live [camera_index]   摄像头实时畸变校正预览
//
// 棋盘格规格：9 列 x 6 行【内角点】= 10 x 7 个方格

#include <opencv2/opencv.hpp>

#include <cmath>
#include <cstdio>
#include <direct.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static const int kCols = 9;
static const int kRows = 6;

static void ensureDir(const std::string& p) { _mkdir(p.c_str()); }

// ---------------- 棋盘格纹理 ----------------
static cv::Mat makeBoardTexture(int sq) {
    cv::Mat img((kRows + 1) * sq, (kCols + 1) * sq, CV_8UC1, cv::Scalar(255));
    for (int r = 0; r <= kRows; ++r)
        for (int c = 0; c <= kCols; ++c)
            if ((r + c) % 2 == 0) img(cv::Rect(c * sq, r * sq, sq, sq)).setTo(0);
    return img;
}

// ---------------- 角点检测：粗定位 + 亚像素精修 ----------------
static bool findCorners(const cv::Mat& gray, std::vector<cv::Point2f>& corners) {
    bool ok = cv::findChessboardCorners(
        gray, cv::Size(kCols, kRows), corners,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
    if (!ok) return false;
    cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
    return true;
}

// ---------------- 求解器 ----------------
struct CalibResult {
    bool ok;
    double rms;
    cv::Mat K, dist;
    std::vector<double> viewErr;
    CalibResult() : ok(false), rms(0) {}
};

static CalibResult solveCalib(const std::vector<std::vector<cv::Point2f> >& imgPts,
                              double square_mm, cv::Size imgSize, bool verbose, int flags = 0) {
    CalibResult res;
    if (imgPts.size() < 3) {
        std::cerr << "[solve] need >= 3 valid views, got " << imgPts.size() << "\n";
        return res;
    }

    std::vector<cv::Point3f> objTemplate;
    for (int y = 0; y < kRows; ++y)
        for (int x = 0; x < kCols; ++x)
            objTemplate.push_back(cv::Point3f((float)(x * square_mm), (float)(y * square_mm), 0.f));

    std::vector<std::vector<cv::Point3f> > obj;
    for (size_t i = 0; i < imgPts.size(); ++i) obj.push_back(objTemplate);

    std::vector<cv::Mat> rvecs, tvecs;
    res.K = cv::Mat::eye(3, 3, CV_64F);
    res.dist = cv::Mat::zeros(5, 1, CV_64F);
    res.rms = cv::calibrateCamera(
        obj, imgPts, imgSize, res.K, res.dist, rvecs, tvecs, flags,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 1e-6));
    res.ok = true;

    double sum = 0;
    for (size_t i = 0; i < obj.size(); ++i) {
        std::vector<cv::Point2f> proj;
        cv::projectPoints(obj[i], rvecs[i], tvecs[i], res.K, res.dist, proj);
        double e = 0;
        for (size_t j = 0; j < proj.size(); ++j) e += cv::norm(proj[j] - imgPts[i][j]);
        e /= (double)proj.size();
        res.viewErr.push_back(e);
        sum += e;
        if (verbose)
            std::cout << "  view " << std::setw(2) << i << "  mean reproj err = "
                      << std::fixed << std::setprecision(3) << e << " px\n";
    }
    if (verbose)
        std::cout << "  mean over all views = " << std::fixed << std::setprecision(3)
                  << sum / (double)obj.size() << " px\n";
    std::cout.unsetf(std::ios::floatfield);
    return res;
}

static void printIntrinsics(const CalibResult& r) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  fx=" << r.K.at<double>(0, 0) << "  fy=" << r.K.at<double>(1, 1)
              << "  cx=" << r.K.at<double>(0, 2) << "  cy=" << r.K.at<double>(1, 2) << "\n";
    std::cout << "  k1=" << r.dist.at<double>(0) << "  k2=" << r.dist.at<double>(1)
              << "  p1=" << r.dist.at<double>(2) << "  p2=" << r.dist.at<double>(3);
    if (r.dist.total() > 4) std::cout << "  k3=" << r.dist.at<double>(4);
    std::cout << "\n  RMS = " << r.rms << " px\n";
    std::cout.unsetf(std::ios::floatfield);
}

// ---------------- 生成棋盘格 ----------------
static int runGen(int sqPx) {
    cv::Mat board = makeBoardTexture(sqPx);
    char name[128];
    sprintf(name, "chessboard_%dx%d_%dpx.png", kCols, kRows, sqPx);
    cv::imwrite(name, board);
    std::cout << "[gen] wrote " << name << "  size = " << board.cols << "x" << board.rows
              << " px  (" << (kCols + 1) << "x" << (kRows + 1) << " squares)\n";
    std::cout << "[gen] 用手机全屏显示它，再用尺子量【一个方格】的边长(mm)，"
                 "这个数就是 --solve 的 square_mm\n";
    return 0;
}

// ---------------- 采图 ----------------
static int runCapture(int index, bool manual_exp, double exposure) {
    cv::VideoCapture cap(index, cv::CAP_DSHOW);
    if (!cap.isOpened()) {
        std::cerr << "[capture] cannot open camera " << index << "\n";
        return 1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    if (manual_exp) {
        cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0);
        cap.set(cv::CAP_PROP_EXPOSURE, exposure);
        std::cout << "[capture] manual exposure = " << exposure << " (log2 s)\n";
    }

    const std::string dir = "calib_imgs";
    ensureDir(dir);
    int saved = 0;
    double armAt = 0;
    std::cout << "[capture] SPACE=存图  ESC=结束\n";
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "[capture] empty frame, stop\n";
            break;
        }
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> corners;
        bool found = findCorners(gray, corners);

        cv::Mat vis = frame.clone();
        if (found) {
            cv::rectangle(vis, cv::boundingRect(corners), cv::Scalar(255, 128, 0), 1);
            cv::drawChessboardCorners(vis, cv::Size(kCols, kRows), corners, found);
            cv::Rect bb = cv::boundingRect(corners);
            double spanX = 100.0 * bb.width / frame.cols;
            double spanY = 100.0 * bb.height / frame.rows;
            char cov[160];
            sprintf(cov, "spanX=%.0f%% spanY=%.0f%% need>70%%", spanX, spanY);
            cv::putText(vis, cov, cv::Point(10, 85), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                        spanX > 65.0 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255), 2);
        }
        char info[160];
        sprintf(info, "found=%d saved=%d bright=%.1f", (int)found, saved, cv::mean(gray)[0]);
        cv::putText(vis, info, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    found ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
        if (cv::mean(gray)[0] < 30.0)
            cv::putText(vis, "TOO DARK: add light", cv::Point(10, 55),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 165, 255), 2);
        if (armAt > 0) {
            double remain = (armAt - (double)cv::getTickCount()) / cv::getTickFrequency();
            char cd[96];
            sprintf(cd, "SHOT IN %.1fs  don't move", remain > 0 ? remain : 0.0);
            cv::putText(vis, cd, cv::Point(10, 115), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 255, 255), 2);
        }
        cv::imshow("calib", vis);

        int k = cv::waitKey(30);
        if (k == 27) break;
        if (k == ' ' && found && armAt == 0)
            armAt = (double)cv::getTickCount() + 2.0 * cv::getTickFrequency();
        if (armAt > 0 && (double)cv::getTickCount() >= armAt) {
            cv::Mat shot;
            cap >> shot;
            if (!shot.empty()) {
                char name[128];
                sprintf(name, "%s/view_%02d.png", dir.c_str(), saved);
                cv::imwrite(name, shot);
                ++saved;
                std::cout << "  saved " << name << "\n";
            }
            armAt = 0;
        }
    }
    std::cout << "[capture] total saved = " << saved << "  (目标 15~20 张)\n";
    return 0;
}

// ---------------- 抓一张图（给直线度检验用） ----------------
static int runSnap(const std::string& out, int index, bool manual_exp, double exposure)
{
    cv::VideoCapture cap(index, cv::CAP_DSHOW);
    if (!cap.isOpened()) { std::cerr << "[snap] cannot open camera " << index << "\n"; return 1; }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    if (manual_exp) {
        cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0);
        cap.set(cv::CAP_PROP_EXPOSURE, exposure);
    }
    std::cout << "[snap] " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
              << cap.get(cv::CAP_PROP_FRAME_HEIGHT)
              << "   SPACE = save to " << out << "   ESC = quit\n";
    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        cv::Mat vis = frame.clone();
        cv::putText(vis, "SPACE = save    ESC = quit", cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        char info[128];
        std::snprintf(info, sizeof(info), "bright=%.1f  exp=%.1f",
                      cv::mean(frame)[0], exposure);
        cv::putText(vis, info, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 255), 2);
        if (cv::mean(frame)[0] < 30.0)
            cv::putText(vis, "TOO DARK: add light", cv::Point(10, 75),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 165, 255), 2);
        cv::imshow("snap", vis);
        int k = cv::waitKey(30);
        if (k == 27) break;
        if (k == ' ') {
            cv::Mat shot;
            cap >> shot;
            if (!shot.empty()) {
                cv::imwrite(out, shot);
                std::cout << "[snap] saved " << out << "  " << shot.cols << "x"
                          << shot.rows << "\n";
            }
            break;
        }
    }
    cv::destroyAllWindows();
    return 0;
}

// ---------------- 求解真实采集 ----------------
static int runSolve(double square_mm, const std::string& dir, int flags = 0) {
    std::vector<cv::String> files;
    cv::glob(dir + "/*.png", files, false);
    std::vector<cv::String> jpg;
    cv::glob(dir + "/*.jpg", jpg, false);
    files.insert(files.end(), jpg.begin(), jpg.end());
    if (files.empty()) {
        std::cerr << "[solve] no images in " << dir << "\n";
        return 1;
    }

    std::vector<std::vector<cv::Point2f> > pts;
    cv::Size imgSize;
    for (size_t i = 0; i < files.size(); ++i) {
        cv::Mat img = cv::imread(files[i], cv::IMREAD_GRAYSCALE);
        if (img.empty()) continue;
        imgSize = img.size();
        std::vector<cv::Point2f> c;
        if (!findCorners(img, c)) {
            std::cout << "  [skip] " << files[i] << "  no board\n";
            continue;
        }
        pts.push_back(c);
    }
    std::cout << "[solve] detected " << pts.size() << " / " << files.size() << " views  model=" << (flags ? "simple(k1,k2)" : "full(k1,k2,p1,p2,k3)") << "\n";

    CalibResult r = solveCalib(pts, square_mm, imgSize, true, flags);
    if (!r.ok) return 1;
    printIntrinsics(r);

    // ---- 自检 0：四角反畸变位移是否对称、是否发散 ----
    // 桶形畸变的位移必须随半径单调、四角大致对称。
    // 如果某个角位移是 0，说明畸变多项式在那里已经跑飞，undistortPoints 直接
    // 放弃、原值返回 —— 这种模型 RMS 再低也不能用。
    {
        std::ios::fmtflags oldFlags = std::cout.flags();
        std::streamsize oldPrec = std::cout.precision();
        std::vector<cv::Point2f> cs, us;
        cs.push_back(cv::Point2f(0.0f, 0.0f));
        cs.push_back(cv::Point2f((float)imgSize.width - 1, 0.0f));
        cs.push_back(cv::Point2f(0.0f, (float)imgSize.height - 1));
        cs.push_back(cv::Point2f((float)imgSize.width - 1, (float)imgSize.height - 1));
        cv::undistortPoints(cs, us, r.K, r.dist, cv::noArray(), r.K);
        const char* nm[4] = { "TL", "TR", "BL", "BR" };
        double sh[4];
        bool diverged = false, asym = false;
        for (int i = 0; i < 4; ++i) {
            sh[i] = std::hypot(us[i].x - cs[i].x, us[i].y - cs[i].y);
            if (sh[i] < 0.01) diverged = true;
        }
        double mn = sh[0], mx = sh[0];
        for (int i = 1; i < 4; ++i) { mn = std::min(mn, sh[i]); mx = std::max(mx, sh[i]); }
        std::cout << "[check] corner shift";
        for (int i = 0; i < 4; ++i)
            std::cout << "  " << nm[i] << "=" << std::fixed << std::setprecision(1) << sh[i];
        std::cout << " px   symmetry " << std::setprecision(2) << (mn > 1e-6 ? mx / mn : 0.0) << "x\n";
        if (diverged) {
            std::cout << "[check] DIVERGED at a corner -> distortion model is broken, "
                         "use the simple model (k1,k2 only)\n";
            asym = true;
        }
        if (mn > 1e-6 && mx / mn > 2.0) {
            std::cout << "[check] asymmetric corner shift -> the model is absorbing "
                         "something that is not lens distortion (curved target, "
                         "lighting gradient, or too little coverage)\n";
            asym = true;
        }
        if (!asym)
            std::cout << "[check] corner correction symmetric -> physically sane\n";
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrec);
    }

    // ---- 自检 1：换"另一个"模型重解，fx 是否稳定 ----
    // 注意：必须换到对立的模型。如果主解已经是 simple，再拿 simple 去比对
    // 就是自己比自己，永远得到 0%，自检形同虚设。
    const int altFlags = (flags == 0) ? (cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_K3) : 0;
    CalibResult r2 = solveCalib(pts, square_mm, imgSize, false, altFlags);
    if (r2.ok) {
        const double f1 = r.K.at<double>(0, 0), f2 = r2.K.at<double>(0, 0);
        const double dfx = 100.0 * (f1 > f2 ? f1 - f2 : f2 - f1) / f1;
        std::cout << "[check] alt model (" << (altFlags ? "k1,k2 only" : "k1,k2,p1,p2,k3")
                  << ")  fx=" << f2 << "  RMS=" << r2.rms << "\n";
        std::cout << "[check] fx change " << dfx << "%  -> "
                  << (dfx < 5.0 ? "STABLE, params trustworthy"
                                : "UNSTABLE! not enough diversity, do not use this result")
                  << "\n";
    }

    // ---- 自检 2：采图距离是否单一 ----
    double sMin = 1e9, sMax = -1e9;
    for (size_t i = 0; i < pts.size(); ++i) {
        cv::Rect bb = cv::boundingRect(pts[i]);
        double s = 100.0 * bb.width / imgSize.width;
        if (s < sMin) sMin = s;
        if (s > sMax) sMax = s;
    }
    std::cout << "[check] spanX range " << sMin << "% ~ " << sMax << "%\n";
    if (sMax - sMin < 25.0)
        std::cout << "[check] distance too uniform! mix 40% / 60% / 85% views, "
                     "otherwise fx trades off against distance\n";

    cv::FileStorage fs("calib.yaml", cv::FileStorage::WRITE);
    fs << "image_width" << imgSize.width << "image_height" << imgSize.height;
    fs << "camera_matrix" << r.K;
    fs << "distortion_coefficients" << r.dist;
    fs << "rms" << r.rms;
    fs << "square_size_mm" << square_mm;
    fs.release();
    std::cout << "[solve] wrote calib.yaml\n";
    return 0;
}

// ---------------- 合成数据自检 ----------------
static int runSynth() {
    const int W = 640, H = 480;
    cv::Mat K0 = (cv::Mat_<double>(3, 3) << 600, 0, 320, 0, 600, 240, 0, 0, 1);
    cv::Mat D0 = (cv::Mat_<double>(1, 5) << -0.25, 0.10, 0.0008, -0.0005, 0.02);

    const int sqPx = 40;
    const double sq = 25.0;
    cv::Mat board = makeBoardTexture(sqPx);
    const double bw = (kCols + 1) * sq, bh = (kRows + 1) * sq;
    const cv::Point2d bc(bw * 0.5, bh * 0.5);

    std::vector<cv::Point2f> pix, und;
    pix.reserve(W * H);
    for (int v = 0; v < H; ++v)
        for (int u = 0; u < W; ++u) pix.push_back(cv::Point2f((float)u, (float)v));
    cv::undistortPoints(pix, und, K0, D0);

    ensureDir("calib_synth");
    std::vector<std::vector<cv::Point2f> > allPts;
    int made = 0;
    for (int k = 0; k < 12; ++k) {
        double rx = ((k % 4) - 1.5) * 0.12;
        double ry = ((k / 4) - 1.0) * 0.20;
        double rz = ((k % 3) - 1.0) * 0.10;
        double tz = 480.0 + (k % 3) * 60.0;
        cv::Mat R;
        cv::Rodrigues(cv::Vec3d(rx, ry, rz), R);
        cv::Mat cvec = (cv::Mat_<double>(3, 1) << bc.x, bc.y, 0.0);
        cv::Mat zer = (cv::Mat_<double>(3, 1) << 0, 0, tz);
        cv::Mat t = -R * cvec + zer;

        const double R00 = R.at<double>(0, 0), R01 = R.at<double>(0, 1);
        const double R10 = R.at<double>(1, 0), R11 = R.at<double>(1, 1);
        const double R20 = R.at<double>(2, 0), R21 = R.at<double>(2, 1);
        const double t0 = t.at<double>(0), t1 = t.at<double>(1), t2 = t.at<double>(2);

        cv::Mat mapx(H, W, CV_32F), mapy(H, W, CV_32F);
        for (int v = 0; v < H; ++v) {
            for (int u = 0; u < W; ++u) {
                const cv::Point2f& p = und[v * W + u];
                cv::Mat A = (cv::Mat_<double>(3, 3) << R00, R01, -p.x,
                                                      R10, R11, -p.y,
                                                      R20, R21, -1.0);
                cv::Mat b = (cv::Mat_<double>(3, 1) << -t0, -t1, -t2);
                cv::Mat sol;
                bool ok = cv::solve(A, b, sol);
                if (!ok) { mapx.at<float>(v, u) = -1.f; mapy.at<float>(v, u) = -1.f; continue; }
                double X = sol.at<double>(0), Y = sol.at<double>(1), s = sol.at<double>(2);
                double bu = X / sq * sqPx, bv = Y / sq * sqPx;
                if (s <= 0 || bu < 0 || bv < 0 || bu > board.cols - 1 || bv > board.rows - 1) {
                    mapx.at<float>(v, u) = -1.f;
                    mapy.at<float>(v, u) = -1.f;
                } else {
                    mapx.at<float>(v, u) = (float)bu;
                    mapy.at<float>(v, u) = (float)bv;
                }
            }
        }
        cv::Mat img;
        cv::remap(board, img, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(60));

        std::vector<cv::Point2f> c;
        if (!findCorners(img, c)) {
            std::cout << "  view " << k << ": detect FAILED (skipped)\n";
            continue;
        }
        allPts.push_back(c);
        ++made;
        char name[64];
        sprintf(name, "calib_synth/synth_%02d.png", k);
        cv::imwrite(name, img);
    }
    std::cout << "[synth] generated " << made << " views, detected "
              << allPts.size() << "\n";
    if (allPts.empty()) return 1;

    CalibResult r = solveCalib(allPts, sq, cv::Size(W, H), false);
    if (!r.ok) return 1;

    std::cout << "\n[truth ] fx=fy=600  cx=320  cy=240  k1=-0.25 k2=0.10 p1=0.0008 p2=-0.0005 k3=0.02\n";
    std::cout << "[solved]\n";
    printIntrinsics(r);

    double efx = std::fabs(r.K.at<double>(0, 0) - 600.0) / 600.0 * 100.0;
    double efy = std::fabs(r.K.at<double>(1, 1) - 600.0) / 600.0 * 100.0;
    double ecx = std::fabs(r.K.at<double>(0, 2) - 320.0);
    double ecy = std::fabs(r.K.at<double>(1, 2) - 240.0);
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n[error ] fx " << efx << "%  fy " << efy << "%  "
              << "cx " << ecx << "px  cy " << ecy << "px\n";
    std::cout.unsetf(std::ios::floatfield);
    std::cout << (efx < 1.0 && ecx < 3.0 ? "[PASS ] 求解器可信\n" : "[FAIL ] 求解器有问题\n");
    return 0;
}

// ---------------- 实时摄像头校正（预计算 map，高效 remap） ----------------
static int runUndistortVideo(const std::string& yaml_path, int cam_idx)
{
    cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "load yaml fail\n";
        return 1;
    }
    cv::Mat K, dist;
    int w, h;
    fs["image_width"] >> w;
    fs["image_height"] >> h;
    fs["camera_matrix"] >> K;
    fs["distortion_coefficients"] >> dist;
    fs.release();

    cv::VideoCapture cap(cam_idx, cv::CAP_DSHOW);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, w);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, h);
    if (!cap.isOpened())
    {
        std::cerr << "cam open fail\n";
        return 1;
    }

    cv::Mat map1, map2;
    cv::Size imgSize(w, h);
    cv::Mat newK = cv::getOptimalNewCameraMatrix(K, dist, imgSize, 1.0, imgSize);
    cv::initUndistortRectifyMap(
        K, dist,
        cv::Mat(),
        newK,
        imgSize,
        CV_32FC1,
        map1, map2
    );

    while (true)
    {
        cv::Mat frame, out;
        cap >> frame;
        if (frame.empty()) break;

        cv::remap(frame, out, map1, map2, cv::INTER_LINEAR);

        cv::Mat show;
        cv::hconcat(frame, out, show);
        cv::imshow("live: raw | undistort", show);

        int k = cv::waitKey(1);
        if (k == 27) break;
    }
    cap.release();
    return 0;
}

// ----------------【教学片段：原生 undistort，左右拼接对比】----------------
static int runUndistortTeach(const std::string& img_path, const std::string& yaml_path = "calib.yaml")
{
    cv::Mat K, dist;
    cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "无法打开 calib.yaml\n";
        return -1;
    }
    fs["camera_matrix"] >> K;
    fs["distortion_coefficients"] >> dist;
    fs.release();

    cv::Mat src = cv::imread(img_path);
    if (src.empty())
    {
        std::cerr << "图片读取失败\n";
        return -1;
    }

    cv::Mat undist;
    cv::undistort(src, undist, K, dist);   // 一次性校正（适合单张图）

    cv::Mat canvas;
    cv::hconcat(src, undist, canvas);

    cv::putText(canvas, "Original", {15, 30},
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(canvas, "Undistorted", {src.cols + 15, 30},
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    cv::imshow("Distort vs Undistort", canvas);
    std::cout << "提示：观察棋盘格外侧边框，看弯线是否拉直；按任意键关闭窗口\n";
    cv::waitKey(0);
    return 0;
}

// ---------------- main ----------------
int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "--help";
    if (mode == "--gen") {
        int sq = argc > 2 ? atoi(argv[2]) : 40;
        return runGen(sq);
    }
    if (mode == "--capture") {
        int idx = argc > 2 ? atoi(argv[2]) : 0;
        bool manual = argc > 3;
        double exposure = argc > 3 ? atof(argv[3]) : -5.0;
        return runCapture(idx, manual, exposure);
    }
    if (mode == "--snap") {
        std::string out = argc > 2 ? argv[2] : "straight.png";
        int idx = argc > 3 ? atoi(argv[3]) : 0;
        double expo = argc > 4 ? atof(argv[4]) : -6.0;
        return runSnap(out, idx, true, expo);
    }
    if (mode == "--solve") {
        if (argc < 3) { std::cout << "usage: day4_calib --solve <square_mm> [img_dir]\n"; return 1; }
        double sq = atof(argv[2]);
        std::string dir = argc > 3 ? argv[3] : "calib_imgs";
        int flags = 0;
        if (argc > 4 && std::string(argv[4]) == "simple")
            flags = cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_K3;
        return runSolve(sq, dir, flags);
    }
    if (mode == "--synth") {
        return runSynth();
    }
    if (mode == "--undistort")
    {
        if (argc == 3)
            return runUndistortTeach(argv[2]);
        else if (argc >= 4 && std::string(argv[2]) == "--live")
            return runUndistortVideo("calib.yaml", atoi(argv[3]));
        else
        {
            std::cout << "usage:\n"
                      << "  day4_calib --undistort image.png          教学版：undistort 单张左右对比\n"
                      << "  day4_calib --undistort --live [cam_index] 实时高效 remap 预览\n";
            return 1;
        }
    }

    std::cout << "day4_calib  (9x6 inner-corner chessboard)\n"
              << "  --gen [square_px]                  生成棋盘格 PNG\n"
              << "  --capture [camera_index]          采标定图（SPACE 存，ESC 退）\n"
              << "  --snap <out.png> [idx] [exp]      抓一张图（给直线度检验用，640x480）\n"
              << "  --solve <square_mm> [img_dir]     求内参 -> calib.yaml\n"
              << "  --synth                           合成数据自检\n"
              << "  --undistort image.png             【教学】单张校正对比（原生 undistort）\n"
              << "  --undistort --live [idx]          实时摄像头校正（remap 高性能）\n";
    return 0;
}
