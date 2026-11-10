// probe_d6.cpp —— 在同一张合成圆（真值直径 200.0 px）上对比四种拟合
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <cstdio>

static bool gn(const std::vector<cv::Point>& pts, double& a, double& b, double& r,
               bool corrected, int iters = 20)
{
    for (int it = 0; it < iters; ++it)
    {
        double J11=0,J12=0,J1r=0,J22=0,J2r=0,Jrr=0,f1=0,f2=0,fr=0;
        for (const auto& p : pts)
        {
            double dx = p.x - a, dy = p.y - b;
            double d = std::sqrt(dx*dx + dy*dy);
            if (d < 1e-12) continue;
            double ux = dx/d, uy = dy/d, e = d - r;
            J11 += ux*ux; J12 += ux*uy; J22 += uy*uy; Jrr += 1.0;
            if (corrected) { J1r += ux; J2r += uy; f1 += e*ux; f2 += e*uy; }
            else           { J1r -= ux; J2r -= uy; f1 -= e*ux; f2 -= e*uy; }
            fr += e;
        }
        cv::Mat J = (cv::Mat_<double>(3,3) << J11,J12,J1r, J12,J22,J2r, J1r,J2r,Jrr);
        cv::Mat f = (cv::Mat_<double>(3,1) << f1,f2,fr);
        cv::Mat delta;
        if (!cv::solve(J, f, delta, cv::DECOMP_SVD)) return false;
        a += delta.at<double>(0); b += delta.at<double>(1); r += delta.at<double>(2);
        if (!std::isfinite(a) || !std::isfinite(r)) return false;
    }
    return true;
}

int main(int argc, char** argv)
{
    const char* file = argc > 1 ? argv[1] : "circle_test.png";
    cv::Mat src = cv::imread(file, cv::IMREAD_GRAYSCALE);
    if (src.empty()) { printf("read fail: %s\n", file); return 1; }

    cv::Mat bin;
    cv::threshold(src, bin, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    std::vector<std::vector<cv::Point>> cs;
    cv::findContours(bin, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    if (cs.empty()) { printf("no contour\n"); return 1; }

    size_t k = 0; double best = 0;
    for (size_t i = 0; i < cs.size(); ++i) {
        double a = cv::contourArea(cs[i]);
        if (a > best) { best = a; k = i; }
    }
    const auto& cnt = cs[k];
    printf("image: %s   points = %zu   area = %.1f\n", file, cnt.size(), best);
    printf("TRUE diameter = 200.0 px\n\n");
    printf("%-22s %12s %12s %12s\n", "method", "center", "radius", "diameter");
    printf("%-22s %12s %12s %12s\n", "", "", "", "");

    cv::Point2f c; float r;
    cv::minEnclosingCircle(cnt, c, r);
    printf("%-22s (%6.1f,%6.1f) %12.3f %12.3f\n", "minEnclosingCircle", c.x, c.y, r, 2.0*r);

    cv::RotatedRect e = cv::fitEllipse(cnt);
    printf("%-22s (%6.1f,%6.1f) %12.3f %12.3f\n", "fitEllipse", e.center.x, e.center.y,
           e.size.width/2.0, e.size.width);

    double a = c.x, b = c.y, rr = r;
    bool ok = gn(cnt, a, b, rr, false);
    printf("%-22s (%6.1f,%6.1f) %12.3f %12.3f   %s\n", "GN as written", a, b, rr, 2*rr,
           ok ? "" : "(diverged)");

    a = c.x; b = c.y; rr = r;
    ok = gn(cnt, a, b, rr, true);
    printf("%-22s (%6.1f,%6.1f) %12.3f %12.3f   %s\n", "GN signs corrected", a, b, rr, 2*rr,
           ok ? "" : "(diverged)");
    return 0;
}
