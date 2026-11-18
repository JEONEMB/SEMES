#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Fields = std::vector<std::string>;
using Pose = std::pair<int, int>;

std::vector<Fields> readTable(const fs::path& path, char delim) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot read " + path.string());
    std::string line;
    std::getline(input, line); // header
    std::vector<Fields> rows;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        Fields fields;
        std::string field;
        bool quoted = false;
        for (char c : line) {
            if (c == '"') quoted = !quoted;
            else if (c == delim && !quoted) {
                fields.push_back(field);
                field.clear();
            } else field += c;
        }
        if (quoted) throw std::runtime_error("Unclosed field in " + path.string());
        fields.push_back(field);
        rows.push_back(std::move(fields));
    }
    return rows;
}

Pose pose(const std::string& y, const std::string& t) {
    return {std::lround(100 * std::stod(y)), std::lround(100 * std::stod(t))};
}

struct Shot {
    std::string date, path, group;
    Pose yt;
    bool center;
    int fold;
};

void auditSplit(std::ostream& out, const std::string& name, int fold,
                const std::vector<Shot>& train, const std::vector<Shot>& test) {
    std::set<Pose> trainPoses, sharedPoses;
    std::set<std::string> trainPaths, trainGroups, sharedGroups;
    for (const Shot& shot : train) {
        trainPoses.insert(shot.yt);
        trainPaths.insert(shot.path);
        trainGroups.insert(shot.group);
    }
    int samePaths = 0, testAtSharedPose = 0;
    for (const Shot& shot : test) {
        samePaths += trainPaths.count(shot.path);
        if (trainPoses.count(shot.yt)) {
            sharedPoses.insert(shot.yt);
            ++testAtSharedPose;
        }
        if (trainGroups.count(shot.group)) sharedGroups.insert(shot.group);
    }
    out << name << '\t' << fold << '\t' << train.size() << '\t' << test.size()
        << '\t' << samePaths << '\t' << sharedGroups.size() << '\t' << sharedPoses.size()
        << '\t' << testAtSharedPose << '\n';
}

void auditFolds(std::ostream& out, const std::string& name, const std::vector<Shot>& shots) {
    std::set<int> folds;
    for (const Shot& shot : shots) if (shot.fold >= 0) folds.insert(shot.fold);
    for (int fold : folds) {
        std::vector<Shot> train, test;
        for (const Shot& shot : shots) {
            if (shot.fold >= 0) (shot.fold == fold ? test : train).push_back(shot);
        }
        auditSplit(out, name, fold, train, test);
    }
}

int main(int argc, char** argv) try {
    if (argc != 4) throw std::runtime_error("Usage: audit REPO CLEAN OUTPUT");
    const fs::path repo = argv[1], clean = argv[2], output = argv[3];
    const fs::path hye = repo / "experiments/hye";
    const fs::path metadata = repo / "data/fivefold_seed20261104/_metadata";
    const fs::path a0 = hye / fs::u8path(u8"20261102_124309_A0_분할측정고정_계층분할과점측정캐시/결과/cache/shots.csv");
    std::ofstream ledger(output / "shots.tsv");
    std::ofstream splits(output / "splits.tsv");
    std::ofstream summary(output / "coverage.tsv");
    if (!ledger || !splits || !summary) throw std::runtime_error("Cannot create output tables");
    ledger << "date\tpath_from_clean\tY_x100\tT_x100\trepeat_group\tcenter\tfirst_341\tgroup_5fold\n";
    splits << "split\tfold\ttrain_n\ttest_n\tsame_files\tshared_repeat_groups\tshared_YT_poses\ttest_images_at_shared_pose\n";
    summary << "date\timages\tcenters\tposes\tmixed_images\tmixed_poses\tQ1_images\tQ2_images\tQ3_images\tQ4_images\n";

    std::vector<Shot> shots;
    std::map<std::string, Shot> byPath, controlByName;
    for (const bool center : {false, true}) {
        for (const auto& row : readTable(metadata / (center ? "centers.csv" : "images.csv"), ',')) {
            const std::string date = row.at(0);
            const std::string path = "clean_" + date + "/" + row.at(2);
            const Shot shot{date, path, row.at(6), pose(row.at(4), row.at(5)), center,
                            center ? -1 : std::stoi(row.at(10))};
            if (!fs::is_regular_file(clean / fs::u8path(path)) || !byPath.emplace(path, shot).second)
                throw std::runtime_error("Missing or duplicate image " + path);
            shots.push_back(shot);
            if (date == "20261014" && !center) controlByName.emplace(row.at(8), shot);
            ledger << date << '\t' << path << '\t' << shot.yt.first << '\t' << shot.yt.second
                   << '\t' << shot.group << '\t' << center << '\t' << (date != "20261029")
                   << '\t' << shot.fold << '\n';
        }
    }
    int onDisk = 0;
    for (const auto& entry : fs::recursive_directory_iterator(clean))
        if (entry.is_regular_file() && entry.path().extension() == ".bmp") ++onDisk;
    if (onDisk != static_cast<int>(shots.size())) throw std::runtime_error("Image count differs from manifest");

    for (const std::string date : {"20261014", "20261020", "20261029"}) {
        int images = 0, centers = 0, mixed = 0, quadrant[4] = {};
        std::set<Pose> poses, mixedPoses;
        for (const Shot& shot : shots) if (shot.date == date) {
            ++images;
            centers += shot.center;
            poses.insert(shot.yt);
            const auto [y, t] = shot.yt;
            if (!y || !t) continue;
            ++mixed;
            mixedPoses.insert(shot.yt);
            ++quadrant[y > 0 ? (t > 0 ? 0 : 3) : (t > 0 ? 1 : 2)];
        }
        summary << date << '\t' << images << '\t' << centers << '\t' << poses.size()
                << '\t' << mixed << '\t' << mixedPoses.size();
        for (const int n : quadrant) summary << '\t' << n;
        summary << '\n';

        std::vector<Shot> train, test, only1014;
        for (const Shot& shot : shots) if (!shot.center) {
            (shot.date == date ? test : train).push_back(shot);
            if (shot.date == "20261014") only1014.push_back(shot);
        }
        auditSplit(splits, "leave_date_out", std::stoi(date), train, test);
        if (date != "20261014") auditSplit(splits, "train_1014", std::stoi(date), only1014, test);
    }
    auditFolds(splits, "date_pose_group_5fold", shots);
    const fs::path a17 = hye / fs::u8path(u8"20261104_091825_A17_날짜층화5fold_joint재평가/결과/results/folds5.csv");
    std::vector<Shot> stratified;
    for (const auto& row : readTable(a17, ',')) {
        Shot shot = byPath.at(row.at(1) + "/" + row.at(2));
        shot.fold = std::stoi(row.at(5));
        stratified.push_back(shot);
    }
    auditFolds(splits, "A17_date_stratified_5fold", stratified);

    std::vector<Shot> joint;
    int labelDifferences = 0, otherCenters = 0;
    std::ofstream labelAudit(output / "label_differences.tsv");
    if (!labelAudit) throw std::runtime_error("Cannot write label audit");
    labelAudit << "path\tmanifest_Y_x100\tmanifest_T_x100\tA0_Y_x100\tA0_T_x100\n";
    for (const auto& row : readTable(a0, ',')) {
        Shot shot = byPath.at(row.at(1) + "/" + row.at(2));
        const Pose yt = pose(row.at(3), row.at(4));
        if (shot.yt != yt) {
            ++labelDifferences;
            labelAudit << shot.path << '\t' << shot.yt.first << '\t' << shot.yt.second
                       << '\t' << yt.first << '\t' << yt.second << '\n';
        }
        shot.yt = yt;
        shot.group = shot.date + "/" + row.at(7);
        shot.fold = std::stoi(row.at(8));
        otherCenters += shot.center && shot.fold >= 0;
        joint.push_back(shot);
    }
    if (joint.size() != shots.size()) throw std::runtime_error("A0 count differs from manifest");
    auditFolds(splits, "A0_joint_4fold_as_stored", joint);

    std::ifstream control(hye / "00000_final_Y_T control/validation/splits.json");
    const std::string json((std::istreambuf_iterator<char>(control)), {});
    const std::regex foldPattern(R"re("fold"\s*:\s*(\d+)\s*,\s*"train"\s*:\s*\[([^\]]*)\]\s*,\s*"test"\s*:\s*\[([^\]]*)\])re");
    const std::regex filename(R"re("([^"]+)")re");
    const auto names = [&](const std::string& list) {
        std::vector<Shot> found;
        for (auto m = std::sregex_iterator(list.begin(), list.end(), filename); m != std::sregex_iterator(); ++m)
            found.push_back(controlByName.at((*m)[1]));
        return found;
    };
    int nFolds = 0;
    for (auto m = std::sregex_iterator(json.begin(), json.end(), foldPattern); m != std::sregex_iterator(); ++m) {
        auditSplit(splits, "control_4fold", std::stoi((*m)[1]), names((*m)[2]), names((*m)[3]));
        ++nFolds;
    }
    if (nFolds != 4) throw std::runtime_error("Expected four control folds");

    const fs::path mixed = hye / "20261105_090000_mixed_learning_curve/results";
    std::map<int, Shot> mixedIndex;
    for (const auto& row : readTable(mixed / "dataset.tsv", '\t')) {
        const std::string prefix = "data/clean/";
        if (row.at(2).compare(0, prefix.size(), prefix) != 0) throw std::runtime_error("Unexpected dataset path");
        mixedIndex.emplace(std::stoi(row.at(0)), byPath.at(row.at(2).substr(prefix.size())));
    }
    std::map<std::string, std::vector<Shot>> bySeed;
    for (const auto& row : readTable(mixed / "test_folds.tsv", '\t')) {
        Shot shot = mixedIndex.at(std::stoi(row.at(2)));
        shot.fold = std::stoi(row.at(1));
        bySeed[row.at(0)].push_back(shot);
    }
    for (const auto& [seed, foldShots] : bySeed) auditFolds(splits, "random_images_" + seed, foldShots);
    std::cout << "Manifest images: " << shots.size() << "\nA0 label differences: " << labelDifferences
              << "\nCenter images assigned to A0 test folds: " << otherCenters << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
