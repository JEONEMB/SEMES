#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    std::cout << "OpenCV version: "
              << CV_VERSION << std::endl;

    cv::Mat image(300, 500, CV_8UC3, cv::Scalar(255, 255, 255));

    cv::putText(
        image,
        "OpenCV 3.4.15",
        cv::Point(80, 150),
        cv::FONT_HERSHEY_SIMPLEX,
        1.0,
        cv::Scalar(0, 0, 0),
        2
    );

    cv::imshow("OpenCV Test", image);
    cv::waitKey(0);

    return 0;
}