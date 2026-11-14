// 사진 → 학습, 평가용 표 (dataset/prepared/)
//
// 사용: prepare_dataset <dataset 폴더> [--data <data 폴더>]
//   <dataset 폴더>/photos/<회차>/…  에 SEMES 에서 받은 사진을 이름 그대로 넣는다 (dataset/README.md).
//   --data 를 안 주면 "data" (실사용과 같은 기준 사진으로 잰다).
//
// 하는 일
//   1. 회차 폴더마다 사진을 찾고 파일 이름에서 명령값을 읽는다 (못 읽는 이름은 건너뛰고 보고서에 적는다)
//   2. 실사용과 같은 함수(measure_alignment)로 QR, 나사 1~4 의 가로 이동을 잰다
//   3. 회차마다 "그날 로봇 0 이 절대 영점에서 얼마나 어긋났나"를 거꾸로 구한다
//        값마다  값 = aY·명령Y + aT·명령T + (2차 항) + 회차마다 다른 상수  를 모든 회차에 함께 맞춘 뒤,
//        회차 상수가 0 이 되는(= 모든 값이 0 인, 기준 화면과 같은) 명령을 그 회차의 영점 어긋남으로 삼는다
//   4. 학습용 정답 = 명령 − 그 회차의 영점 어긋남  (받은 명령값은 photos.tsv 에 그대로 남긴다)
//   5. 시험 나누기: 같은 회차의 같은 명령 자세는 한 묶음. 20% 를 마지막 확인용으로 떼고, 나머지를 5묶음 × 10번
#include "alignment.hpp"
#include "common/dataset.hpp"
#include "common/photo_files.hpp"
#include "common/tables.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>

namespace fs = std::filesystem;
using namespace tools;

namespace {

const double kTrainedRangeYmm = 2.0;       // 학습에 쓰는 명령 범위
const double kTrainedRangeTdeg = 1.0;
const double kFinalCheckShare = 0.2;       // 마지막 확인용으로 떼는 묶음 비율
const unsigned kSplitSeed = 20261114;      // 나누기 씨앗 (누가 돌려도 같은 나누기)
const int kOutlierRounds = 3;               // 명령값과 측정이 크게 어긋나는 사진을 빼고 다시 맞추는 횟수
const double kOutlierLimit = 10.0;          // 정상 흔들림의 몇 배를 넘으면 "명령값이 의심스러운 사진"으로 보나

struct PreparedPhoto {
    std::string session;
    PhotoFile file;
    bool measured = false;
    Values values{};
    bool all_screws = false;
    bool in_range = false;
    bool command_mismatch = false;   // 명령값과 측정이 크게 어긋남 (이름의 명령값이 틀렸을 수 있음)
    bool usable() const { return measured && all_screws && in_range && !command_mismatch; }
};

std::map<std::string, std::pair<double, double>> fit_session_zeros(const std::vector<const PreparedPhoto*>& used, const std::vector<std::string>& sessions, cv::Mat& residual);

double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// 3. 회차마다 영점 어긋남을 거꾸로 구한다.
//    명령값과 측정이 크게 어긋나는 사진(이름의 명령값이 틀렸을 수 있는 사진)은 맞추기에서 빼고 suspicious 에 표시한다.
std::map<std::string, std::pair<double, double>> estimate_session_zeros(const std::vector<PreparedPhoto>& photos, const std::vector<std::string>& sessions, std::vector<int>& suspicious) {
    suspicious.assign(photos.size(), 0);
    std::map<std::string, std::pair<double, double>> zeros;
    for (int round = 0; round < kOutlierRounds + 1; ++round) {
        std::vector<int> used_index;
        for (int i = 0; i < int(photos.size()); ++i) if (photos[i].usable() && !suspicious[i]) used_index.push_back(i);
        std::vector<const PreparedPhoto*> used;
        for (int i : used_index) used.push_back(&photos[i]);
        cv::Mat residual;
        zeros = fit_session_zeros(used, sessions, residual);
        if (round == kOutlierRounds) break;
        // 값마다 가운데값 기반 흔들림으로 나눈 어긋남이 kOutlierLimit 을 넘는 사진을 뺀다
        bool removed = false;
        std::vector<double> spread(kValueCount);
        for (int k = 0; k < kValueCount; ++k) {
            std::vector<double> column;
            for (int i = 0; i < residual.rows; ++i) column.push_back(std::abs(residual.at<double>(i, k)));
            spread[k] = std::max(1.4826 * median_of(column), 1e-9);
        }
        for (int i = 0; i < residual.rows; ++i)
            for (int k = 0; k < kValueCount; ++k)
                if (std::abs(residual.at<double>(i, k)) / spread[k] > kOutlierLimit) { suspicious[used_index[i]] = 1; removed = true; break; }
        if (!removed) break;
    }
    return zeros;
}

std::map<std::string, std::pair<double, double>> fit_session_zeros(const std::vector<const PreparedPhoto*>& used, const std::vector<std::string>& sessions, cv::Mat& residual) {
    const int n = int(used.size()), s_count = int(sessions.size());
    const int pose_terms = 5;   // Y, T, Y², YT, T²
    cv::Mat design = cv::Mat::zeros(n, pose_terms + s_count, CV_64F);
    for (int i = 0; i < n; ++i) {
        const double y = used[i]->file.command_y_mm, t = used[i]->file.command_t_deg;
        const double terms[pose_terms] = {y, t, y * y, y * t, t * t};
        for (int q = 0; q < pose_terms; ++q) design.at<double>(i, q) = terms[q];
        const int s = int(std::find(sessions.begin(), sessions.end(), used[i]->session) - sessions.begin());
        design.at<double>(i, pose_terms + s) = 1;
    }
    cv::Mat slope(kValueCount, 2, CV_64F), constant(kValueCount, s_count, CV_64F);
    residual = cv::Mat(n, kValueCount, CV_64F);
    for (int k = 0; k < kValueCount; ++k) {
        cv::Mat value(n, 1, CV_64F), fitted;
        for (int i = 0; i < n; ++i) value.at<double>(i) = used[i]->values[k];
        cv::solve(design, value, fitted, cv::DECOMP_SVD);
        const cv::Mat predicted = design * fitted;
        for (int i = 0; i < n; ++i) residual.at<double>(i, k) = value.at<double>(i) - predicted.at<double>(i);
        slope.at<double>(k, 0) = fitted.at<double>(0);
        slope.at<double>(k, 1) = fitted.at<double>(1);
        for (int s = 0; s < s_count; ++s) constant.at<double>(k, s) = fitted.at<double>(pose_terms + s);
    }
    std::map<std::string, std::pair<double, double>> zeros;
    for (int s = 0; s < s_count; ++s) {
        cv::Mat zero;   // slope · zero = −constant  (그 명령에서 모든 값이 0)
        cv::solve(slope, -constant.col(s), zero, cv::DECOMP_SVD);
        zeros[sessions[s]] = {zero.at<double>(0), zero.at<double>(1)};
    }
    return zeros;
}

// 5. 시험 나누기
void make_splits(const std::vector<PreparedPhoto>& photos, std::vector<int>& final_check, std::vector<std::array<int, kRepeatCount>>& folds) {
    std::map<std::string, std::vector<int>> groups;   // "회차|Y|T" → 사진 번호
    for (int i = 0; i < int(photos.size()); ++i) {
        if (!photos[i].usable()) continue;
        groups[photos[i].session + "|" + format_number(photos[i].file.command_y_mm) + "|" + format_number(photos[i].file.command_t_deg)].push_back(i);
    }
    std::vector<std::vector<int>> group_list;
    for (auto& g : groups) group_list.push_back(g.second);

    final_check.assign(photos.size(), 0);
    folds.assign(photos.size(), {});
    for (auto& f : folds) f.fill(-1);

    std::mt19937 random(kSplitSeed);
    std::vector<int> order(group_list.size());
    for (int g = 0; g < int(order.size()); ++g) order[g] = g;
    std::shuffle(order.begin(), order.end(), random);
    const int final_groups = int(std::round(kFinalCheckShare * order.size()));
    std::vector<int> development;
    for (int g = 0; g < int(order.size()); ++g) {
        if (g < final_groups) for (int i : group_list[order[g]]) final_check[i] = 1;
        else development.push_back(order[g]);
    }
    for (int r = 0; r < kRepeatCount; ++r) {
        std::mt19937 repeat_random(kSplitSeed + 1 + r);
        std::vector<int> shuffled = development;
        std::shuffle(shuffled.begin(), shuffled.end(), repeat_random);
        for (int g = 0; g < int(shuffled.size()); ++g)
            for (int i : group_list[shuffled[g]]) folds[i][r] = g % kFoldCount;
    }
}

}   // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "사용: prepare_dataset <dataset 폴더> [--data <data 폴더>]\n"; return 2; }
    const fs::path dataset = fs::u8path(argv[1]);
    std::string data_folder = "data";
    for (int a = 2; a + 1 < argc; ++a) if (std::string(argv[a]) == "--data") data_folder = argv[a + 1];

    const fs::path photo_root = dataset / "photos", prepared = dataset / "prepared";
    std::string error;
    if (!make_empty_folder(prepared, error)) { std::cerr << error << "\n"; return 1; }
    if (alignment::initialize(data_folder).status == alignment::Status::Failed) return 1;

    // 1, 2. 사진 찾기, 재기
    const std::vector<std::string> sessions = list_subfolders(photo_root);
    if (sessions.empty()) { std::cerr << "사진 폴더가 없습니다: " << photo_root.u8string() << " (회차마다 폴더를 만들어 넣으세요)\n"; return 1; }
    std::vector<PreparedPhoto> photos;
    std::vector<std::string> skipped;
    for (const std::string& session : sessions) {
        for (const PhotoFile& file : find_photos(photo_root / fs::u8path(session))) {
            if (!file.has_command) { skipped.push_back(session + "/" + file.name + "  (이름에 명령값이 없음)"); continue; }
            PreparedPhoto photo;
            photo.session = session;
            photo.file = file;
            photo.in_range = std::abs(file.command_y_mm) <= kTrainedRangeYmm + 1e-9 && std::abs(file.command_t_deg) <= kTrainedRangeTdeg + 1e-9;
            alignment::clear_zero();
            const alignment::Result r = alignment::measure_alignment(read_gray_photo(file.path));
            photo.measured = r.status != alignment::Status::Failed;
            photo.values[0] = r.qr_shift_px;
            photo.all_screws = photo.measured;
            for (int k = 0; k < 4; ++k) {
                photo.values[k + 1] = r.screw_shift_px[k];
                if (!std::isfinite(r.screw_shift_px[k])) photo.all_screws = false;
            }
            photos.push_back(photo);
        }
        std::cerr << "  " << session << " 측정 끝\n";
    }

    // 3, 4, 5
    std::vector<int> mismatch;
    const auto zeros = estimate_session_zeros(photos, sessions, mismatch);
    std::vector<std::string> mismatch_names;
    for (int i = 0; i < int(photos.size()); ++i)
        if (mismatch[i]) { photos[i].command_mismatch = true; mismatch_names.push_back(photos[i].session + "/" + photos[i].file.name); }
    std::vector<int> final_check;
    std::vector<std::array<int, kRepeatCount>> folds;
    make_splits(photos, final_check, folds);

    // 표 쓰기
    std::ofstream photo_table(prepared / "photos.tsv", std::ios::binary), feature_table(prepared / "features.tsv", std::ios::binary),
        label_table(prepared / "labels.tsv", std::ios::binary), split_table(prepared / "splits.tsv", std::ios::binary),
        session_table(prepared / "sessions.tsv", std::ios::binary);
    photo_table << "id\tsession\tpath\tcommand_y_mm\tcommand_t_deg\n";
    feature_table << "id\tmeasured\tqr\tscrew1\tscrew2\tscrew3\tscrew4\n";
    label_table << "id\tlabel_y_mm\tlabel_t_deg\n";
    split_table << "id\tusable\tfinal_check";
    for (int r = 0; r < kRepeatCount; ++r) split_table << "\tcv" << r;
    split_table << "\n";
    for (int i = 0; i < int(photos.size()); ++i) {
        const PreparedPhoto& p = photos[i];
        const auto& zero = zeros.at(p.session);
        photo_table << i << '\t' << p.session << '\t' << p.file.path.u8string() << '\t' << format_number(p.file.command_y_mm) << '\t' << format_number(p.file.command_t_deg) << '\n';
        feature_table << i << '\t' << (p.measured ? 1 : 0);
        for (double v : p.values) feature_table << '\t' << (std::isfinite(v) ? format_number(v, 12) : "nan");
        feature_table << '\n';
        label_table << i << '\t' << format_number(p.file.command_y_mm - zero.first, 12) << '\t' << format_number(p.file.command_t_deg - zero.second, 12) << '\n';
        split_table << i << '\t' << (p.usable() ? 1 : 0) << '\t' << final_check[i];
        for (int r = 0; r < kRepeatCount; ++r) split_table << '\t' << folds[i][r];
        split_table << '\n';
    }
    session_table << "session\tzero_y_mm\tzero_t_deg\tphotos\n";
    for (const std::string& s : sessions) {
        int count = 0;
        for (const PreparedPhoto& p : photos) count += p.session == s;
        session_table << s << '\t' << format_number(zeros.at(s).first, 8) << '\t' << format_number(zeros.at(s).second, 8) << '\t' << count << '\n';
    }

    // 보고서
    std::ofstream report(prepared / "prepare_report.txt", std::ios::binary);
    int usable = 0, failed = 0, missing_screw = 0, out_of_range = 0, final_count = 0;
    for (int i = 0; i < int(photos.size()); ++i) {
        const PreparedPhoto& p = photos[i];
        usable += p.usable();
        failed += !p.measured;
        missing_screw += p.measured && !p.all_screws;
        out_of_range += !p.in_range;
        final_count += final_check[i];
    }
    report << "사진 " << photos.size() << "장 (회차 " << sessions.size() << "개)\n"
           << "  학습, 평가에 쓰는 사진 " << usable << "장 (그중 마지막 확인용 " << final_count << "장)\n"
           << "  빼는 사진: 측정 실패 " << failed << ", 나사를 다 못 찾음 " << missing_screw << ", 명령이 학습 범위(|Y| <= 2, |T| <= 1) 밖 " << out_of_range
           << ", 명령값과 측정이 크게 어긋남 " << mismatch_names.size() << "\n\n"
           << "회차별 영점 어긋남 (명령 − 학습 정답. 그날 로봇 0 과 절대 영점의 차이)\n";
    for (const std::string& s : sessions) report << "  " << s << "  Y " << format_number(zeros.at(s).first, 5) << " mm, T " << format_number(zeros.at(s).second, 5) << " 도\n";
    report << "\n명령값과 측정이 크게 어긋나 뺀 사진 (이름의 명령값이 틀렸을 수 있다. 사진을 확인하세요) " << mismatch_names.size() << "장\n";
    for (const std::string& s : mismatch_names) report << "  " << s << "\n";
    report << "\n건너뛴 파일 " << skipped.size() << "개\n";
    for (const std::string& s : skipped) report << "  " << s << "\n";
    report.close();
    std::ifstream echo(prepared / "prepare_report.txt", std::ios::binary);
    std::cout << echo.rdbuf();
    return 0;
}
