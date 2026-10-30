// 실측 사진마다 채택 모듈의 중간 측정값과 추정값을 저장한다 (증강 검증용 실험 도구).
//   joint:   Observer::measure 의 102관측 (QR 정합 54 + 선 48)과 joint 추정 (y, t)
//   control: Tracker::measure 의 630점 변위 - center 변위 (Tracker::features 와 같은 차분)와 control 추정 (축, 값)
// 사용: dump_features <joint assets> <control assets> <photo_paths.txt: index\tpath> <out folder>
#include "observer_internal.hpp"
#include "control_internal.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

int main(int argc, char** argv) {
    if (argc != 5) { std::cerr << "usage: dump_features <joint assets> <control assets> <photo_paths.txt> <out>\n"; return 2; }
    namespace fs = std::filesystem;
    // 명령줄 인자는 시스템 코드페이지, 목록 파일 안의 경로는 UTF-8이다.
    fs::path jointAssets(argv[1]), controlAssets(argv[2]), list(argv[3]), out(argv[4]);
    try {
    fs::create_directories(out);
    alignment_detail::Observer observer(jointAssets);
    alignment_detail::JointModel jointModel; jointModel.load(jointAssets / "both_model.json");
    legacy_control::Tracker tracker(controlAssets);
    legacy_control::Model controlModel(controlAssets / "model.json");
    auto zero = tracker.measure(legacy_control::read_gray(controlAssets / "center.bmp")).displacement.clone();

    std::ifstream in(list);
    std::ofstream fj(out / "joint_obs.csv"), fc(out / "control_disp.csv"), fp(out / "predictions.csv");
    fj << std::setprecision(10); fc << std::setprecision(8); fp << std::setprecision(10);
    fp << "index,joint_y,joint_t,control_axis,control_branch,control_value,control_valid536\n";
    std::string line; int count = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto tab = line.find('\t'); std::string index = line.substr(0, tab); fs::path path = fs::u8path(line.substr(tab + 1));
        cv::Mat gray = alignment_detail::read_gray(path);
        auto obs = observer.measure(gray); auto j = jointModel.predict(obs.values);
        fj << index; for (int k = 0; k < 102; ++k) fj << ',' << obs.values.at<double>(k); fj << '\n';
        auto tracks = tracker.measure(gray);
        fc << index;
        for (int p = 0; p < 630; ++p) for (int c = 0; c < 2; ++c)
            fc << ',' << double(tracks.displacement.at<float>(p, c)) - double(zero.at<float>(p, c));
        for (int p = 0; p < 630; ++p) fc << ',' << int(tracks.valid[p]);
        fc << '\n';
        auto choice = controlModel.predict(tracker.features(tracks));
        fp << index << ',' << j.y << ',' << j.t << ',' << choice.axis << ',' << choice.branch << ',' << choice.value << ',' << tracker.valid_selected(tracks) << '\n';
        if (++count % 30 == 0) std::cout << "measured " << count << std::endl;
    }
    std::cout << "complete " << count << std::endl;
    } catch (const std::exception& e) { std::cerr << "error: " << e.what() << std::endl; return 1; }
    return 0;
}
