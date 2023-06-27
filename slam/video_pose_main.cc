#include "openvslam/system.h"
#include "openvslam/config.h"
#ifdef USE_PANGOLIN_VIEWER
#include "pangolin_viewer/viewer.h"
#endif

#include <iostream>
#include <chrono>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <spdlog/spdlog.h>
#include <popl.hpp>

void mono_localizing(const std::shared_ptr<openvslam::config>& cfg, const std::string& vocab_file_path, const std::string& video_file_path,
  const std::string& mask_img_path, int frame_limit, const std::string& map_path, const std::string& traj_path) {

    const cv::Mat mask = mask_img_path.empty() ? cv::Mat{} : cv::imread(mask_img_path, cv::IMREAD_GRAYSCALE);
    // build a SLAM system
    openvslam::system SLAM(cfg, vocab_file_path);

    SLAM.load_map_database(map_path);


    // startup the SLAM process
    SLAM.startup(false);

    SLAM.enable_mapping_module();

    #ifdef USE_PANGOLIN_VIEWER
        pangolin_viewer::viewer viewer(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
    #endif

    cv::VideoCapture video(video_file_path, cv::CAP_FFMPEG);
    unsigned int total_num = video.get(CV_CAP_PROP_FRAME_COUNT);
    std::vector<double> track_times;
    track_times.reserve(total_num);

    cv::Mat frame;
    double timestamp = 0.0;

    unsigned int num_frame = 0;

    // run the viewer in another thread
#ifdef USE_PANGOLIN_VIEWER
   std::thread thread([&]() {
        viewer.run();
    });
#endif


    while (video.read(frame)) {


        const auto tp_1 = std::chrono::steady_clock::now();

        // input the current frame and estimate the camera pose
        /*****************************************************************************************************
         * Get camera pose here
        *******************************************************************************************************/
        Eigen::Matrix4d pose = SLAM.feed_monocular_frame(frame, timestamp, mask);

        if(SLAM.is_tracking())
        {
            std::cout << num_frame << std::endl << pose << std::endl;
        }

        const auto tp_2 = std::chrono::steady_clock::now();

        const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
        track_times.push_back(track_time);

        timestamp += 1.0 / cfg->camera_->fps_;

        // check if the termination of SLAM system is requested or not
        if (SLAM.terminate_is_requested()) {
            break;
        }
        if (num_frame > frame_limit){
            break;
        }
        ++num_frame;
    }






    // wait until the loop BA is finished
    while (SLAM.loop_BA_is_running()) {
        std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }

#ifdef USE_PANGOLIN_VIEWER
    // automatically close the viewer
    viewer.request_terminate();
#endif

    thread.join();

    // shutdown the SLAM process
    SLAM.shutdown();
    if (!traj_path.empty()) {
        // output the map database
        SLAM.save_frame_trajectory(traj_path, "KITTI");

    }

  }

void mono_tracking(const std::shared_ptr<openvslam::config>& cfg, const std::string& vocab_file_path, const std::string& video_file_path,
  const std::string& mask_img_path, int frame_limit, const std::string& map_path) {
    // load the mask image
    const cv::Mat mask = mask_img_path.empty() ? cv::Mat{} : cv::imread(mask_img_path, cv::IMREAD_GRAYSCALE);

    // build a SLAM system
    openvslam::system SLAM(cfg, vocab_file_path);
    // startup the SLAM process
    SLAM.startup();

    // create a viewer object
    // and pass the frame_publisher and the map_publisher
#ifdef USE_PANGOLIN_VIEWER
    pangolin_viewer::viewer viewer(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#endif

    cv::VideoCapture video(video_file_path, cv::CAP_FFMPEG);
    std::vector<double> track_times;
    unsigned int total_num = video.get(CV_CAP_PROP_FRAME_COUNT);
    track_times.resize(total_num * 2);

    cv::Mat frame;
    double timestamp = 0.0;

    unsigned int num_frame = 0;

    // run the viewer in another thread
#ifdef USE_PANGOLIN_VIEWER
   std::thread thread([&]() {
        viewer.run();
    });
#endif


    while (video.read(frame)) {
        
        std::cout << "Frame: " << num_frame << std::endl;
        const auto tp_1 = std::chrono::steady_clock::now();

        // input the current frame and estimate the camera pose
        /*****************************************************************************************************
         * Get camera pose here
        *******************************************************************************************************/
        Eigen::Matrix4d pose = SLAM.feed_monocular_frame(frame, timestamp, mask);

        if(SLAM.is_tracking())
        {
            std::cout << "Pose: " << pose << std::endl;
        }

        const auto tp_2 = std::chrono::steady_clock::now();

        const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
        track_times.push_back(track_time);

        timestamp += 1.0 / cfg->camera_->fps_;

        // check if the termination of SLAM system is requested or not
        if (SLAM.terminate_is_requested()) {
            break;
        }
        if (num_frame > frame_limit){
            break;
        }
        ++num_frame;
    }

    // wait until the loop BA is finished
    while (SLAM.loop_BA_is_running()) {
        std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }

#ifdef USE_PANGOLIN_VIEWER
    // automatically close the viewer
    viewer.request_terminate();
#endif

    thread.join();

    // shutdown the SLAM process
    SLAM.shutdown();
    if (!map_path.empty()) {
        // output the map database
        SLAM.save_map_database(map_path);
    }

}

int main(int argc, char* argv[]) {
     // create options
    popl::OptionParser op("Allowed options");

    std::string video_folder("build/360_test/");
    std::string vocab_folder("build/orb_vocab/");
    std::string mask_file("");
    std::string map_db_path("");
    std::string trajectory_db_path("");
    int frame_limit = -1;
    int mode = -1;
    auto help = op.add<popl::Switch>("h", "help", "produce help message");
    auto vocab_folder_path = op.add<popl::Value<std::string>>("o", "vocab", "vocabulary folder path");
    auto video_folder_path = op.add<popl::Value<std::string>>("v", "video", "video and config folder path");
    auto mask_file_path = op.add<popl::Value<std::string>>("m", "mask", "mask file name in the video folder");
    auto video_frame_limit = op.add<popl::Value<int>>("f", "frame", "frame limit number");
    auto map_path = op.add<popl::Value<std::string>>("p", "map", "map db path");
    auto trajectory_path = op.add<popl::Value<std::string>>("t", "trajectory", "trajectory path");
    auto mode_int = op.add<popl::Value<int>>("n", "mode", "mapping or localizing 0 for mapping 1 for localizing");

    try {
        op.parse(argc, argv);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    if (help->is_set()) {
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    if(vocab_folder_path->is_set()){
        vocab_folder = vocab_folder_path->value();
    }

    if(video_folder_path->is_set()){
        video_folder = video_folder_path->value();
    }

    if(mask_file_path->is_set()){
        mask_file = video_folder + mask_file_path->value();
    }

    if(video_frame_limit->is_set()){
        frame_limit = video_frame_limit->value();
    }

    if(mode_int->is_set()){
        mode = mode_int->value();
    }

    if(map_path->is_set()){
      map_db_path=map_path->value();
    }

    if(trajectory_path->is_set()){
      trajectory_db_path=trajectory_path->value();
    }

    // load configuration
    std::shared_ptr<openvslam::config> cfg;
    try {
        cfg = std::make_shared<openvslam::config>(video_folder + "config.yaml");
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // run tracking
    if (cfg->camera_->setup_type_ == openvslam::camera::setup_type_t::Monocular) {
      if(mode == 0){
        printf("mapping...\n");
        mono_tracking(cfg, vocab_folder + "orb_vocab.dbow2", video_folder + "video.mp4", mask_file, frame_limit, map_db_path);
      }
      else if(mode == 1){
        printf("localizaing...\n");
        mono_localizing(cfg, vocab_folder + "orb_vocab.dbow2", video_folder + "video.mp4", mask_file, frame_limit, map_db_path, trajectory_db_path);
      }


    }
    else {
        throw std::runtime_error("Invalid setup type: " + cfg->camera_->get_setup_type_string());
    }

    return EXIT_SUCCESS;
}
