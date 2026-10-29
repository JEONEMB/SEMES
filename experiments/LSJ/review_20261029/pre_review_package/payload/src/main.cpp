#include "theta_alignment.hpp"
#include "sequential_alignment.hpp"
#include <memory>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

int run(const std::vector<std::string>& args) {
    std::string referencePath, configPath, outputPath, listPath, yConfigPath;
    std::vector<std::string> images;
    for (size_t i = 1; i < args.size(); ++i) {
        const auto& flag = args[i];
        if (flag == "--reference" || flag == "--config" || flag == "--y-config" || flag == "--output" || flag == "--input-list") {
            if (++i == args.size()) throw std::invalid_argument("Missing value for " + flag);
            if (flag == "--reference") referencePath = args[i];
            else if (flag == "--config") configPath = args[i];
            else if (flag == "--y-config") yConfigPath = args[i];
            else if (flag == "--output") outputPath = args[i];
            else listPath = args[i];
        } else if (flag.rfind("--", 0) == 0) throw std::invalid_argument("Unknown option: " + flag);
        else images.push_back(flag);
    }
    if (referencePath.empty() || configPath.empty()) {
        std::cerr << "Usage: main --reference center.bmp --config configs/theta_alignment.json "
                  << "[--y-config configs/y_alignment.json] [--input-list paths.txt] [--output result.json] [image.bmp ...]\n";
        return 2;
    }
    if (!listPath.empty()) {
        std::string contents = semes::readTextUtf8(listPath);
        if (contents.compare(0, 3, "\xEF\xBB\xBF") == 0) contents.erase(0, 3);
        std::istringstream lines(contents); std::string path;
        while (std::getline(lines, path)) {
            if (!path.empty() && path.back() == '\r') path.pop_back();
            if (!path.empty()) images.push_back(path);
        }
    }
    if (images.empty()) throw std::invalid_argument("No input images");
    // Prevent accidentally replacing an input through an output argument.
    if (!outputPath.empty() && (outputPath == referencePath || outputPath == configPath || outputPath == listPath || outputPath == yConfigPath))
        throw std::invalid_argument("Output path must not equal an input path");
    for (const auto& path : images) if (outputPath == path) throw std::invalid_argument("Output equals image input");

    cv::setNumThreads(2);
    const auto config = semes::ThetaConfig::load(configPath);
    const cv::Mat reference = semes::readGrayUtf8(referencePath);
    std::unique_ptr<semes::ThetaEstimator> estimator;
    std::unique_ptr<semes::SequentialAligner> sequential;
    if (yConfigPath.empty()) estimator = std::make_unique<semes::ThetaEstimator>(reference, config);
    else sequential = std::make_unique<semes::SequentialAligner>(reference, config, semes::YConfig::load(yConfigPath));
    cv::FileStorage fs(".json", cv::FileStorage::WRITE | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
    fs << "schema_version" << 1 << "opencv_version" << CV_VERSION << "stage" << (sequential ? "theta_then_y" : "theta_only")
       << "reference" << referencePath << "config" << configPath << "y_config" << yConfigPath
       << "mode" << "independent_offline_images_no_machine_moves" << "results" << "[";
    int failed = 0;
    for (const auto& path : images) {
        fs << "{" << "file" << path;
        try {
            const cv::Mat current = semes::readGrayUtf8(path);
            if (sequential) {
                const auto result = sequential->estimate(current);
                fs << "valid" << 1;
                semes::writeAlignmentDecision(fs, result);
                if (!result.commandAvailable) ++failed;
            } else {
                const auto result = estimator->estimate(current);
                fs << "valid" << 1;
                semes::writeThetaResult(fs, result);
                if (!result.commandAvailable) ++failed;
            }
        } catch (const std::exception& error) {
            ++failed;
            fs << "valid" << 0 << "status" << "measurement_failed" << "error" << error.what();
        }
        fs << "}";
    }
    fs << "]";
    const std::string json = fs.releaseAndGetString();
    if (outputPath.empty()) std::cout << json;
    else { semes::writeTextUtf8(outputPath, json); std::cout << "Processed " << images.size() << ", unresolved/failed " << failed << "\n"; }
    return failed ? 1 : 0;
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i) {
        const int n = WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, nullptr, 0, nullptr, nullptr);
        std::string value(n, '\0');
        WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, &value[0], n, nullptr, nullptr);
        value.resize(n - 1); args.push_back(value);
    }
#else
int main(int argc, char** argv) {
    std::vector<std::string> args(argv, argv + argc);
#endif
    try { return run(args); }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
}
