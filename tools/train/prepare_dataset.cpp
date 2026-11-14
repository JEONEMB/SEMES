// 사진 → 학습, 평가용 표 (data/new_photos/tables/)
//
// 사용: prepare_dataset <사진 폴더> [--data <model 폴더>]
//   <사진 폴더>/<회차>/…  에 SEMES 에서 받은 사진을 이름 그대로 넣는다 (data/new_photos/README.md). 표는 <사진 폴더>/tables/ 에 쓴다.
//   --data 를 안 주면 "data/model" (실사용과 같은 기준 사진으로 잰다).
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
const int kOutlierRounds = 3;               // 값이 식에서 크게 벗어난 사진을 영점 맞추기에서 빼고 다시 맞추는 횟수
const double kOutlierLimit = 10.0;          // 정상 흔들림의 몇 배를 넘으면 영점 맞추기에서 빼나
const double kCommandErrorYmm = 0.2;        // 사진이 보여 주는 자세와 이름의 명령이 이보다 크게 다르면 "명령값이 틀린 사진"
const double kCommandErrorTdeg = 0.1;       //   (통과 기준 0.045 mm, 0.01 도보다 훨씬 커서 측정 오차로는 생기지 않는 크기)
const int kPoseTerms = 5;                   // 영점 맞추기 식의 자세 항: Y, T, Y², YT, T²
const char* const kTableFolder = "tables";   // 사진 폴더 안에 표를 쓰는 폴더 이름

struct PreparedPhoto {
    std::string session;
    PhotoFile file;
    bool measured = false;
    Values values{};
    bool all_screws = false;
    bool in_range = false;
    bool command_mismatch = false;   // 사진이 보여 주는 자세가 이름의 명령과 크게 다름 (이름의 명령값이 틀렸을 수 있음)
    bool usable() const { return measured && all_screws && in_range && !command_mismatch; }
};

double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// 모든 회차를 함께 맞춘 식.  값 = 자세 항 5개(Y, T, Y², YT, T²) + 회차마다 다른 상수
struct ZeroFit {
    cv::Mat coefficients;                                      // 값마다 한 줄, (자세 항 5 + 회차 수) 칸
    std::map<std::string, std::pair<double, double>> zeros;    // 회차 → 영점 어긋남 (Y mm, T 도)
};

cv::Mat fit_terms(const PreparedPhoto& photo, const std::vector<std::string>& sessions) {
    const double y = photo.file.command_y_mm, t = photo.file.command_t_deg;
    const double pose[kPoseTerms] = {y, t, y * y, y * t, t * t};
    cv::Mat terms = cv::Mat::zeros(1, kPoseTerms + int(sessions.size()), CV_64F);
    for (int q = 0; q < kPoseTerms; ++q) terms.at<double>(q) = pose[q];
    terms.at<double>(kPoseTerms + int(std::find(sessions.begin(), sessions.end(), photo.session) - sessions.begin())) = 1;
    return terms;
}

ZeroFit fit_session_zeros(const std::vector<const PreparedPhoto*>& used, const std::vector<std::string>& sessions) {
    const int n = int(used.size()), s_count = int(sessions.size());
    cv::Mat design(n, kPoseTerms + s_count, CV_64F);
    for (int i = 0; i < n; ++i) fit_terms(*used[i], sessions).copyTo(design.row(i));
    ZeroFit fit;
    fit.coefficients = cv::Mat(kValueCount, kPoseTerms + s_count, CV_64F);
    for (int k = 0; k < kValueCount; ++k) {
        cv::Mat value(n, 1, CV_64F), fitted;
        for (int i = 0; i < n; ++i) value.at<double>(i) = used[i]->values[k];
        cv::solve(design, value, fitted, cv::DECOMP_SVD);
        cv::Mat(fitted.t()).copyTo(fit.coefficients.row(k));
    }
    const cv::Mat slope = fit.coefficients.colRange(0, 2);   // 값이 Y, T 에 따라 변하는 정도
    for (int s = 0; s < s_count; ++s) {
        cv::Mat zero;   // slope · zero = −회차 상수  (그 명령에서 모든 값이 0)
        cv::solve(slope, -fit.coefficients.col(kPoseTerms + s), zero, cv::DECOMP_SVD);
        fit.zeros[sessions[s]] = {zero.at<double>(0), zero.at<double>(1)};
    }
    return fit;
}

// 사진의 값 − 식이 말하는 값 (값마다 하나)
cv::Mat residual_of(const ZeroFit& fit, const PreparedPhoto& photo, const std::vector<std::string>& sessions) {
    const cv::Mat predicted = fit.coefficients * fit_terms(photo, sessions).t();
    cv::Mat residual(kValueCount, 1, CV_64F);
    for (int k = 0; k < kValueCount; ++k) residual.at<double>(k) = photo.values[k] - predicted.at<double>(k);
    return residual;
}

// 사진이 보여 주는 자세가 이름의 명령과 얼마나 다른가 (Y mm, T 도). 값의 어긋남을 자세로 바꾼 것
std::pair<double, double> command_error(const ZeroFit& fit, const PreparedPhoto& photo, const std::vector<std::string>& sessions) {
    cv::Mat pose;
    cv::solve(fit.coefficients.colRange(0, 2), residual_of(fit, photo, sessions), pose, cv::DECOMP_SVD);
    return {pose.at<double>(0), pose.at<double>(1)};
}

// 3. 회차마다 영점 어긋남을 거꾸로 구한다.
//    값이 식에서 크게 벗어난 사진은 영점 맞추기에서만 빼고 left_out 에 표시한다 (학습, 평가에는 그대로 쓴다).
ZeroFit estimate_session_zeros(const std::vector<PreparedPhoto>& photos, const std::vector<std::string>& sessions, std::vector<int>& left_out) {
    left_out.assign(photos.size(), 0);
    ZeroFit fit;
    for (int round = 0; round < kOutlierRounds + 1; ++round) {
        std::vector<int> used_index;
        for (int i = 0; i < int(photos.size()); ++i) if (photos[i].usable() && !left_out[i]) used_index.push_back(i);
        std::vector<const PreparedPhoto*> used;
        for (int i : used_index) used.push_back(&photos[i]);
        fit = fit_session_zeros(used, sessions);
        if (round == kOutlierRounds) break;
        // 값마다 가운데값 기반 흔들림으로 나눈 어긋남이 kOutlierLimit 을 넘는 사진을 뺀다
        std::vector<cv::Mat> residual;
        for (const PreparedPhoto* p : used) residual.push_back(residual_of(fit, *p, sessions));
        bool removed = false;
        for (int k = 0; k < kValueCount; ++k) {
            std::vector<double> column;
            for (const cv::Mat& r : residual) column.push_back(std::abs(r.at<double>(k)));
            const double spread = std::max(1.4826 * median_of(column), 1e-9);
            for (size_t i = 0; i < used.size(); ++i)
                if (column[i] / spread > kOutlierLimit) { left_out[used_index[i]] = 1; removed = true; }
        }
        if (!removed) break;
    }
    return fit;
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

// 1, 2. 회차 폴더마다 사진을 찾아 실사용과 같은 함수로 잰다. 이름에 명령값이 없는 파일은 skipped 에 적는다.
std::vector<PreparedPhoto> measure_all_photos(const fs::path& photo_root, const std::vector<std::string>& sessions, std::vector<std::string>& skipped) {
    std::vector<PreparedPhoto> photos;
    for (const std::string& session : sessions) {
        for (const PhotoFile& file : find_photos(photo_root / fs::u8path(session))) {
            if (!file.has_command) { skipped.push_back(session + "/" + file.name + "  (이름에 명령값이 없음)"); continue; }
            PreparedPhoto photo;
            photo.session = session;
            photo.file = file;
            photo.in_range = std::abs(file.command_y_mm) <= kTrainedRangeYmm + 1e-9 && std::abs(file.command_t_deg) <= kTrainedRangeTdeg + 1e-9;
            alignment::clear_zero();
            const alignment::Result r = alignment::measure_alignment(alignment::read_photo(file.path.u8string()));
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
    return photos;
}

// 영점 맞추기에서 빠진 사진을 설명하는 줄 (보고서용)
struct LeftOutNotes {
    std::vector<std::string> mismatch;   // 명령이 틀려 보여 학습, 평가에서도 뺀 사진
    std::vector<std::string> kept;       // 영점 맞추기에서만 빼고 학습, 평가에는 쓰는 사진
};

// 영점 맞추기에서 빠진 사진 가운데 명령이 정말 틀려 보이는 것만 학습, 평가에서 뺀다 (command_mismatch 표시).
LeftOutNotes mark_command_mismatch(std::vector<PreparedPhoto>& photos, const ZeroFit& fit, const std::vector<std::string>& sessions, const std::vector<int>& left_out) {
    LeftOutNotes notes;
    for (int i = 0; i < int(photos.size()); ++i) {
        if (!left_out[i]) continue;
        const auto error = command_error(fit, photos[i], sessions);
        const std::string line = photos[i].session + "/" + photos[i].file.name + "  (사진이 보여 주는 자세 − 명령: Y " + format_number(error.first, 3) + " mm, T " +
                                 format_number(error.second, 3) + " 도)";
        photos[i].command_mismatch = std::abs(error.first) > kCommandErrorYmm || std::abs(error.second) > kCommandErrorTdeg;
        (photos[i].command_mismatch ? notes.mismatch : notes.kept).push_back(line);
    }
    return notes;
}

using SessionZeros = std::map<std::string, std::pair<double, double>>;
using Folds = std::vector<std::array<int, kRepeatCount>>;

// 표 5개를 쓴다 (형식은 common/dataset.hpp).
void write_tables(const fs::path& tables, const std::vector<PreparedPhoto>& photos, const std::vector<std::string>& sessions, const SessionZeros& zeros,
                  const std::vector<int>& final_check, const Folds& folds) {
    std::ofstream photo_table(tables / "photos.tsv", std::ios::binary), feature_table(tables / "features.tsv", std::ios::binary),
        label_table(tables / "labels.tsv", std::ios::binary), split_table(tables / "splits.tsv", std::ios::binary),
        session_table(tables / "sessions.tsv", std::ios::binary);
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
}

// 사진 수, 뺀 사진과 이유, 회차별 영점 어긋남을 사람이 읽는 글로 쓴다.
void write_report(const fs::path& path, const std::vector<PreparedPhoto>& photos, const std::vector<std::string>& sessions, const SessionZeros& zeros,
                  const std::vector<int>& final_check, const LeftOutNotes& notes, const std::vector<std::string>& skipped) {
    int usable = 0, failed = 0, missing_screw = 0, out_of_range = 0, final_count = 0;
    for (int i = 0; i < int(photos.size()); ++i) {
        const PreparedPhoto& p = photos[i];
        usable += p.usable();
        failed += !p.measured;
        missing_screw += p.measured && !p.all_screws;
        out_of_range += !p.in_range;
        final_count += final_check[i];
    }
    std::ofstream report(path, std::ios::binary);
    report << "사진 " << photos.size() << "장 (회차 " << sessions.size() << "개)\n"
           << "  학습, 평가에 쓰는 사진 " << usable << "장 (그중 마지막 확인용 " << final_count << "장)\n"
           << "  빼는 사진: 측정 실패 " << failed << ", 나사를 다 못 찾음 " << missing_screw << ", 명령이 학습 범위(|Y| <= 2, |T| <= 1) 밖 " << out_of_range
           << ", 명령값이 틀려 보임 " << notes.mismatch.size() << "\n\n"
           << "회차별 영점 어긋남 (명령 − 학습 정답. 그날 로봇 0 과 절대 영점의 차이)\n";
    for (const std::string& s : sessions) report << "  " << s << "  Y " << format_number(zeros.at(s).first, 5) << " mm, T " << format_number(zeros.at(s).second, 5) << " 도\n";
    report << "\n명령값이 틀려 보여 뺀 사진 (자세 차이가 Y " << format_number(kCommandErrorYmm) << " mm 또는 T " << format_number(kCommandErrorTdeg)
           << " 도를 넘음. 사진을 확인하세요) " << notes.mismatch.size() << "장\n";
    for (const std::string& s : notes.mismatch) report << "  " << s << "\n";
    report << "\n영점 맞추기에서만 뺀 사진 (값이 식에서 벗어났지만 명령은 맞음. 학습, 평가에는 씀) " << notes.kept.size() << "장\n";
    for (const std::string& s : notes.kept) report << "  " << s << "\n";
    report << "\n건너뛴 파일 " << skipped.size() << "개\n";
    for (const std::string& s : skipped) report << "  " << s << "\n";
}

}   // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "사용: prepare_dataset <사진 폴더> [--data <model 폴더>]\n"; return 2; }
    const fs::path photo_root = fs::u8path(argv[1]), tables = photo_root / kTableFolder;
    std::string model_folder = "data/model";
    for (int a = 2; a + 1 < argc; ++a)
        if (std::string(argv[a]) == "--data") model_folder = argv[a + 1];

    std::string error;
    if (!make_empty_folder(tables, error)) { std::cerr << error << "\n"; return 1; }
    if (alignment::initialize(model_folder).status == alignment::Status::Failed) return 1;
    std::vector<std::string> sessions = list_subfolders(photo_root);
    sessions.erase(std::remove(sessions.begin(), sessions.end(), std::string(kTableFolder)), sessions.end());   // 표 폴더는 회차가 아니다
    if (sessions.empty()) { std::cerr << "사진 폴더가 없습니다: " << photo_root.u8string() << " (회차마다 폴더를 만들어 넣으세요)\n"; return 1; }

    std::vector<std::string> skipped;
    std::vector<PreparedPhoto> photos = measure_all_photos(photo_root, sessions, skipped);   // 1, 2
    std::vector<int> left_out;
    const ZeroFit fit = estimate_session_zeros(photos, sessions, left_out);                  // 3, 4
    const LeftOutNotes notes = mark_command_mismatch(photos, fit, sessions, left_out);
    std::vector<int> final_check;
    Folds folds;
    make_splits(photos, final_check, folds);                                                 // 5

    write_tables(tables, photos, sessions, fit.zeros, final_check, folds);
    write_report(tables / "prepare_report.txt", photos, sessions, fit.zeros, final_check, notes, skipped);
    std::ifstream echo(tables / "prepare_report.txt", std::ios::binary);
    std::cout << echo.rdbuf();
    return 0;
}
