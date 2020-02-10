/** ****************************************************************************
 *  @file    FaceHeadPoseOpencv.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2019/04
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef FACE_HEAD_POSE_OPENCV_HPP
#define FACE_HEAD_POSE_OPENCV_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <FaceHeadPose.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>
using namespace cv::dnn;
using namespace cv;
namespace upm {

/** ****************************************************************************
 * @class FaceHeadPoseOpencv
 * @brief Class used for facial feature point detection.
 ******************************************************************************/
class FaceHeadPoseOpencv: public FaceHeadPose
{
public:
  FaceHeadPoseOpencv(std::string path) : _path(path) {};

  ~FaceHeadPoseOpencv() {};
  
  void detectFaceOpenCVDNN(const cv::Mat& frameOpenCVDNN,  std::vector<FaceAnnotation> &faces,
    Net net);

  void
  parseOptions
    (
    int argc,
    char **argv
    );
  
   void
  train
    (
    const std::vector<FaceAnnotation> &anns_train,
    const std::vector<FaceAnnotation> &anns_valid
    );

  void
  load();

  void
  process
    (
    cv::Mat frame,
    std::vector<FaceAnnotation> &faces,
    const FaceAnnotation &ann
    );
  
private:
  std::string _path;

};

} // namespace upm

#endif /* FACE__PRL19_HPP */
