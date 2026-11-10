// probe_gn.cpp —— 对比你写的 Gauss-Newton 圆拟合和修正符号后的版本
// 输入：完美圆上的 120 个点，真值 center(320,240) r=100
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <cstdio>

struct Pt { double x, y; };

static void run(const std::vector<Pt>& pts, double& a, double& b, double& r,
                bool corrected, int iters = 15)
{
    for (int it = 0; it < iters; ++it)
    {
        double J11 = 0, J12 = 0, J1r = 0, J22 = 0, J2r = 0, Jrr = 0;
        double f1 = 0, f2 = 0, fr = 0;

        for (const auto& p : pts)
        {
            double dx = p.x - a, dy = p.y - b;
            double d = std::sqrt(dx * dx + dy * dy);
            if (d < 1e-12) continue;
            double ux = dx / d, uy = dy / d;
            double e = d - r;

            J11 += ux * ux;  J12 += ux * uy;  J22 += uy * uy;  Jrr += 1.0;
            if (corrected) { J1r += ux;  J2r += uy;  f1 += e * ux;  f2 += e * uy; }
            else           { J1r -= ux;  J2r -= uy;  f1 -= e * ux;  f2 -= e * uy; }
            fr += e;
        }

        cv::Mat J = (cv::Mat_<double>(3, 3) <<
                     J11, J12, J1r,
                     J12, J22, J2r,
                     J1r, J2r, Jrr);
        cv::Mat f = (cv::Mat_<double>(3, 1) << f1, f2, fr);

        cv::Mat delta;
        if (!cv::solve(J, f, delta, cv::DECOMP_SVD)) break;

        a += delta.at<double>(0);
        b += delta.at<double>(1);
        r += delta.at<double>(2);

        if (it < 5 || it == iters - 1)
            printf("   it%-2d center=(%10.2f,%10.2f)  r=%10.2f\n", it, a, b, r);
    }
}

int main()
{
    const double PI = 3.14159265358979323846;
    std::vector<Pt> pts;
    for (int k = 0; k < 120; ++k) {
        double t = 2 * PI * k / 120;
        pts.push_back({ 320 + 100 * cos(t), 240 + 100 * sin(t) });
    }

    printf("ground truth: center(320.000,240.000)  r=100.000  diam=200.000\n\n");

    double a = 305, b = 252, r = 92;
    printf("[AS WRITTEN IN day6_measure.cpp]  init (%.0f,%.0f) r=%.0f\n", a, b, r);
    run(pts, a, b, r, false);
    printf("   => center=(%.3f,%.3f) r=%.3f diam=%.3f\n\n", a, b, r, 2 * r);

    a = 305; b = 252; r = 92;
    printf("[SIGNS CORRECTED]                 init (%.0f,%.0f) r=%.0f\n", a, b, r);
    run(pts, a, b, r, true);
    printf("   => center=(%.3f,%.3f) r=%.3f diam=%.3f\n", a, b, r, 2 * r);
    return 0;
}
