#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

// 几何最小二乘圆拟合：正交距离最小，Gauss-Newton
bool fitGeometricCircleGN(const std::vector<cv::Point>& pts, cv::Point2f& center, float& radius, int maxIter = 20, double eps = 1e-4)
{
    int N = (int)pts.size();
    if (N < 6) return false;

    // 初值先用 minEnclosingCircle，保证初值靠谱
    cv::Point2f c0;
    float r0;
    cv::minEnclosingCircle(pts, c0, r0);
    double a = c0.x;
    double b = c0.y;
    double r = r0;

    for (int iter = 0; iter < maxIter; ++iter)
    {
        double J11 = 0, J12 = 0, J1r = 0;
        double J21 = 0, J22 = 0, J2r = 0;
        double Jr1 = 0, Jr2 = 0, Jrr = 0;
        double f1 = 0, f2 = 0, fr = 0;

        for (const auto& p : pts)
        {
            double x = p.x;
            double y = p.y;
            double dx = x - a;
            double dy = y - b;
            double d = sqrt(dx*dx + dy*dy);
            if (d < 1e-12) continue;

            double e = d - r;
            double dadx = dx / d;
            double dady = dy / d;

            J11 += dadx * dadx;
            J12 += dadx * dady;
            J1r -= dadx;

            J21 += dady * dadx;
            J22 += dady * dady;
            J2r -= dady;

            Jr1 -= dadx;
            Jr2 -= dady;
            Jrr += 1.0;

            f1 -= e * dadx;
            f2 -= e * dady;
            fr += e;
        }

        cv::Mat JtJ = (cv::Mat_<double>(3,3) <<
            J11, J12, J1r,
            J21, J22, J2r,
            Jr1, Jr2, Jrr);
        cv::Mat Jtf = (cv::Mat_<double>(3,1) << f1, f2, fr);

        cv::Mat delta;
        if (!cv::solve(JtJ, Jtf, delta, cv::DECOMP_SVD))
            break;

        double da = delta.at<double>(0);
        double db = delta.at<double>(1);
        double dr = delta.at<double>(2);

        a += da;
        b += db;
        r += dr;

        if (fabs(da) < eps && fabs(db) < eps && fabs(dr) < eps)
            break;
    }

    center = cv::Point2f((float)a, (float)b);
    radius = (float)r;
    return true;
}

int main()
{
    const int width = 400;
    const int height = 400;
    const int true_diameter_px = 200;
    const int true_radius_px = true_diameter_px / 2;
    const cv::Point center(width / 2, height / 2);

    cv::Mat src(height, width, CV_8UC3, cv::Scalar(200, 200, 200));
    cv::circle(src, center, true_radius_px, cv::Scalar(50, 50, 50), -1);

    cv::Mat noise(height, width, CV_8UC1);
    cv::randn(noise, 0, 8);
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    gray += noise;

    cv::Mat bin;
    cv::threshold(gray, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU | cv::THRESH_BINARY_INV);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    if (contours.empty())
    {
        std::cerr << "self test failed: no contour found\n";
        return -1;
    }

    size_t best_idx = 0;
    double max_area = cv::contourArea(contours[0]);
    for (size_t i = 1; i < contours.size(); ++i)
    {
        double a = cv::contourArea(contours[i]);
        if (a > max_area)
        {
            max_area = a;
            best_idx = i;
        }
    }

    const auto& cnt = contours[best_idx];

    float d_minEnc = 0, d_ellipse = 0, d_lsq_geo = 0;

    // minEnclosingCircle
    {
        cv::Point2f c;
        float r;
        cv::minEnclosingCircle(cnt, c, r);
        d_minEnc = 2.0f * r;
    }

    // fitEllipse
    {
        cv::RotatedRect ell = cv::fitEllipse(cnt);
        d_ellipse = ell.size.width;
    }

    // Geometric LS (Gauss-Newton)
    {
        cv::Point2f c;
        float r;
        if (fitGeometricCircleGN(cnt, c, r))
            d_lsq_geo = 2.0f * r;
        else
            d_lsq_geo = 0;
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Synthetic Self-Test ===\n";
    std::cout << "True diameter (px)        : " << true_diameter_px << "\n";
    std::cout << "minEnclosingCircle (px)   : " << d_minEnc
              << "   error = " << (d_minEnc - true_diameter_px) << " px\n";
    std::cout << "fitEllipse width (px)     : " << d_ellipse
              << "   error = " << (d_ellipse - true_diameter_px) << " px\n";
    std::cout << "Geometric LS Circle (px)  : " << d_lsq_geo
              << "   error = " << (d_lsq_geo - true_diameter_px) << " px\n";

    cv::Mat vis = src.clone();
    cv::drawContours(vis, contours, best_idx, cv::Scalar(0, 0, 255), 2);

    cv::Point2f c_tmp;
    float r_tmp;
    cv::minEnclosingCircle(cnt, c_tmp, r_tmp);
    cv::circle(vis, c_tmp, cvRound(r_tmp), cv::Scalar(0, 255, 0), 2);

    cv::imshow("self_test_bin", bin);
    cv::imshow("self_test_vis", vis);
    cv::waitKey(0);

    return 0;
}
