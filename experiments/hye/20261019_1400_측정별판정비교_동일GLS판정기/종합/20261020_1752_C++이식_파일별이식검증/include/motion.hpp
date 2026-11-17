#pragma once
#include "numeric.hpp"
#include <optional>
namespace semes::motion {
struct Settings {
    double probe_y=.05,probe_t=.01,max_step_y=.25,max_step_t=.25,gain=.7,minimum_response_snr=1.5,maximum_condition=1000,noise_rms=1,stop_feature_rms=2.5,stop_command=.03;
    int probe_count=3,max_turns=20,history_frames=9;
    void validate()const;
    Json json()const;
    static Settings from_json(const Json&);
};
struct Frame{cv::Vec2d cumulative;cv::Mat observation;};
class Controller {
    Settings cfg_;
    cv::Mat prior_,J_,s_,goal_;
    std::vector<Frame> history_;
    std::optional<cv::Vec2d> pending_;
    cv::Vec2d cumulative_{0,0},correction_;
    int turn_=0,streak_=0,updates_=0;
    double trust_=1,feature_rms_=std::numeric_limits<double>::infinity();
    std::string status_="not_started",kind_;
    Json diagnostics_=Json::dict(),events_=Json::list();
    void assess();
    bool identify();
public:
    explicit Controller(Settings settings={},const cv::Mat& prior={});
    Controller& start(const cv::Mat& observation,const cv::Mat& fixed_goal);
    std::optional<cv::Vec2d> next_action();
    void observe(const cv::Mat& observation,const cv::Vec2d& acknowledged_increment);
    Json snapshot()const;
    Json state()const;
    static Controller from_state(const Json&);
    const Json& events()const{return events_;}
    const cv::Mat& response()const{return J_;}
    const std::string& status()const{return status_;}
    int turns()const{return turn_;}
    cv::Vec2d correction()const{return correction_;}
};
cv::Vec2d choose_probe_amplitudes(const cv::Mat& prior,cv::Vec2d minimum,cv::Vec2d maximum,double noise_rms);
cv::Mat vector_from_json(const Json&);
Json vector_json(const cv::Mat&);
}
