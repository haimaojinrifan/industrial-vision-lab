// Day 2｜cv::Mat 内存模型实验
// 编译：把本文件加入 CMakeLists.txt，或单独：
//   g++ day2_mat.cpp -o day2_mat -I<opencv>/include -L<opencv>/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio
//
// 五个实验全部用一张自造的彩色图，不依赖外部图片，保证你能跑。
// 每个实验都打印「预期 vs 实际」，请在终端里逐条对照。

#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>

static void hr(const char* title)
{
    std::cout << "\n================ " << title << " ================\n";
}

// 打印 Mat 的关键布局参数
static void dump(const char* name, const cv::Mat& m)
{
    std::cout << std::left << std::setw(14) << name
              << " rows=" << m.rows
              << " cols=" << m.cols
              << " channels=" << m.channels()
              << " depth=" << m.depth()
              << " elemSize=" << m.elemSize()
              << " step[0]=" << m.step[0]
              << " total=" << m.total()
              << " u->refcount=" << (m.u ? m.u->refcount : 0)
              << "\n";
}

int main()
{
    // 自造一张 480x640 的 CV_8UC3 图：左半蓝、右半绿，中间一条红竖线
    cv::Mat img(480, 640, CV_8UC3, cv::Scalar(200, 120, 30)); // BGR
    img(cv::Rect(0, 0, 320, 480)).setTo(cv::Scalar(255, 0, 0));   // 左半蓝
    img(cv::Rect(320, 0, 320, 480)).setTo(cv::Scalar(0, 255, 0)); // 右半绿
    cv::line(img, cv::Point(320, 0), cv::Point(320, 479), cv::Scalar(0, 0, 255), 3);

    hr("0. 基准图");
    dump("img", img);
    std::cout << "预期：total=307200，elemSize=3，step[0]=1920，"
                 "total*elemSize = " << img.total() * img.elemSize() << "，"
                 "rows*step[0] = " << img.rows * img.step[0] << "\n";
    std::cout << ">>> 两者相等，因为这是「连续」Mat（isContinuous=" << img.isContinuous() << "）\n";

    // ---------------- 实验 1：浅拷贝 ----------------
    hr("1. 浅拷贝 Mat b = a");
    cv::Mat a = img;          // 浅拷贝
    cv::Mat b = a;            // 再浅拷贝
    dump("a", a);
    dump("b", b);
    std::cout << "预期：b 的 data 指针和 a 相同 = " << (a.data == b.data ? "是" : "否") << "\n";
    std::cout << "> 注意：refcount=" << (a.u ? a.u->refcount : 0)
              << " 而不是 2 —— 因为 img、a、b 三者都引用同一块数据（img 还没析构），"
                 "refcount 统计的是「所有活着的引用」，不是「a 和 b 两个」。\n";

    b.setTo(cv::Scalar(0, 0, 255)); // 通过 b 把整片染红
    std::cout << "对 b setTo 红色后，a 的 (10,10) 像素 BGR = "
              << a.at<cv::Vec3b>(10, 10) << "（预期 approx 0,0,255）\n";
    std::cout << ">>> 结论：改 b 影响了 a，因为共享同一块数据。这就是「写时复制」的反面——OpenCV 的 Mat 是引用语义，不会自动 COW。\n";

    // ---------------- 实验 2：深拷贝 clone ----------------
    hr("2. 深拷贝 a.clone()");
    cv::Mat c = a.clone();
    dump("c", c);
    std::cout << "预期：c.data 与 a.data 不同 = " << (c.data != a.data ? "是（不同）" : "否") << "\n";
    std::cout << "预期：c 的 refcount 独立，初始为 1\n";

    c.setTo(cv::Scalar(0, 255, 0)); // 把 c 染绿
    std::cout << "对 c setTo 绿色后，a 的 (10,10) 像素 BGR = "
              << a.at<cv::Vec3b>(10, 10) << "（预期仍为 0,0,255，没变）\n";
    std::cout << ">>> 结论：clone() 分配了新内存并拷贝数据，c 和 a 完全独立。\n";

    // ---------------- 实验 3：引用计数 ----------------
    hr("3. 引用计数变化");
    cv::Mat p = cv::Mat::zeros(10, 10, CV_8UC3);
    std::cout << "p 创建后：u->refcount = " << (p.u ? p.u->refcount : 0) << "（预期 1）\n";
    {
        cv::Mat q = p; // 浅拷贝
        std::cout << "q = p 之后：p.u->refcount = " << (p.u ? p.u->refcount : 0) << "（预期 2）\n";
    } // q 析构
    std::cout << "q 离开作用域后：p.u->refcount = " << (p.u ? p.u->refcount : 0) << "（预期 1）\n";
    std::cout << ">>> 结论：refcount 由 UMatData 持有，引用归零时才真正释放内存。\n";

    // ---------------- 实验 4：ROI ----------------
    hr("4. ROI 的 total*elemSize vs rows*step[0]");
    cv::Mat roi = img(cv::Rect(100, 100, 200, 200));
    dump("roi", roi);
    std::cout << "roi.total()*roi.elemSize() = " << roi.total() * roi.elemSize() << "\n";
    std::cout << "roi.rows * roi.step[0]     = " << roi.rows * roi.step[0] << "\n";
    std::cout << "roi.isContinuous()         = " << roi.isContinuous() << "（预期 0）\n";
    std::cout << ">>> 两个数不相等！因为 roi 的 step[0] 继承了父图的 1920，"
                 "它只是父图内存上的一个窗口，实际跨越 200*1920 字节。\n";

    // ---------------- 实验 5：copyTo vs clone ----------------
    hr("5. copyTo vs clone");
    cv::Mat d;
    a.copyTo(d);                // 目标为空，内部会分配
    dump("d(copyTo)", d);
    std::cout << "copyTo 后 d.data 与 a.data 不同 = " << (d.data != a.data ? "是" : "否") << "\n";

    cv::Mat e(480, 640, CV_8UC3); // 预分配一块内存（e.data 已非空）
    void* e_before = e.data;
    a.copyTo(e);                  // 目标尺寸/类型匹配且连续时，复用 e 已有内存
    std::cout << "copyTo 到预分配 Mat e 后，e.data 是否被复用 = "
              << (static_cast<void*>(e.data) == e_before ? "是（复用）" : "否（重新分配了）") << "\n";
    std::cout << ">> 为什么能复用：copyTo 只在「目标为空、尺寸/类型不匹配或非连续」时才重新分配。\n";

    cv::Mat g(100, 100, CV_8UC3); // 尺寸不匹配
    void* g_before = g.data;
    a.copyTo(g);
    std::cout << "copyTo 到尺寸不匹配的 g 后，g.data 是否重新分配 = "
              << (static_cast<void*>(g.data) != g_before ? "是（重新分配，因为尺寸不匹配）" : "否（复用）") << "\n";

    cv::Mat f = a.clone();        // clone 总是新分配
    std::cout << "clone 后 f.data 与 a.data 不同 = " << (f.data != a.data ? "是" : "否") << "\n";

    hr("全部完成");
    std::cout << "请在终端对照每一条「预期 vs 实际」，有出入的贴给我。\n";
    return 0;
}
